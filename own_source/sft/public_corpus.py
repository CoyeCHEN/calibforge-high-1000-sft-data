"""Build independent, source-linked corpora from first qualified teacher successes.

This module consumes admission evidence; it never decides reward, alters a task,
retries a teacher, or rebuilds unseen history from larger raw tool outputs.
"""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .qwen35 import digest
from .teacher_segments import make_segments, project_decision

CAPS = (131072, 262144)
CLASSES = ("thinking", "thinking_boundary", "body", "action", "ending")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(index, count):
    return ("early", "middle", "late")[min(2, 3 * index // count)]


def coverage(decisions, encoded):
    """Count target coverage separately from rows and whole-task retention."""
    by_id = {d.call_id: (index, d) for index, d in enumerate(decisions)}
    retained = [
        m["call_id"]
        for sample in encoded["samples"]
        for m in sample["evidence"]["decision_mapping"]
    ]
    if len(retained) != len(set(retained)):
        raise ValueError("Duplicate supervised target")
    classes = Counter({name: 0 for name in CLASSES})
    stages = Counter({name: 0 for name in ("early", "middle", "late")})
    for ident in retained:
        index, decision = by_id[ident]
        classes.update(decision.evidence["supervision_token_classes"])
        stages[stage(index, len(decisions))] += 1
    return {
        "original_targets": len(decisions),
        "retained_targets": len(retained),
        "segments": len(encoded["samples"]),
        "multi_assistant_samples": sum(
            len(s["evidence"]["decision_mapping"]) > 1 for s in encoded["samples"]
        ),
        "supervised_tokens": sum(classes.values()),
        "supervision_token_classes": dict(classes),
        "stage_target_counts": dict(stages),
        "all_targets_retained": len(retained) == len(decisions),
        "closing_target_retained": bool(
            decisions and decisions[-1].call_id in retained
        ),
        "input_plus_target_lengths": [
            s["evidence"]["input_tokens"] for s in encoded["samples"]
        ],
    }


def uniform_segment_mean_weights(mappings):
    """Analytical row-mean weights, distinct from raw supervised-token shares.

    Applies to one uniform visit per segment, microbatch1 and a mean over each
    segment's supervised tokens. Distributed sampler padding and actual updates
    belong to the training batch audit, not this corpus-level calculation.
    """
    counts = Counter(m["task_id"] for m in mappings)
    tokens = Counter()
    for mapping in mappings:
        if mapping["supervised_tokens"] <= 0:
            raise ValueError("A supervised segment must contain target tokens")
        tokens[mapping["task_id"]] += mapping["supervised_tokens"]
    size = len(mappings)
    total = sum(tokens.values())
    return {
        "task_segment_share": {task: count / size for task, count in counts.items()},
        "task_supervised_token_share": {
            task: count / total for task, count in tokens.items()
        },
        "per_token_coefficients": [
            1 / (size * m["supervised_tokens"]) for m in mappings
        ],
    }


def admitted_records(records, pool):
    """Require the native earliest-success index; never choose by path length."""
    expected = {}
    for record in records:
        if record.get("qualified_success"):
            key = (record["started_at"], record["trial_id"])
            task = record["task_id"]
            if task not in expected or key < expected[task]:
                expected[task] = key
    selected = []
    seen = set()
    families = {}
    for record in records:
        if record["pool"] != pool:
            raise ValueError("Independent corpus received another task pool")
        if not record["is_primary_trajectory"]:
            continue
        if (
            not record["qualified_success"]
            or record["classification"] != "QUALIFIED_SUCCESS"
        ):
            raise ValueError("Primary trajectory lacks qualified admission")
        task = record["task_id"]
        if expected.get(task) != (record["started_at"], record["trial_id"]):
            raise ValueError("Primary is not the chronological first qualified success")
        if task in seen:
            raise ValueError("Two primary trajectories for one independent task")
        seen.add(task)
        split = record["split"]
        if split not in ("train", "validation"):
            raise ValueError("Unknown frozen task split")
        family = record["family_id"]
        if families.setdefault(family, split) != split:
            raise ValueError("Task family crosses frozen splits")
        proof = record["student_proof"]
        required = (
            "semantic_projection_passed",
            "native_loader_and_collator_passed",
            "native_streaming_and_nonstreaming_parsers_passed",
            "all_original_targets_once",
        )
        if not all(proof.get(key) is True for key in required):
            raise ValueError("Student admission evidence incomplete")
        selected.append(record)
    if seen != set(expected):
        raise ValueError("Qualified task lacks a primary trajectory")
    return sorted(selected, key=lambda x: x["task_id"])


def project_trial(tokenizer, record, raw_path):
    proof = record["student_proof"]
    if file_sha(raw_path) != proof["raw_export_sha256"]:
        raise ValueError("Raw canonical export identity changed")
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        calls = [json.loads(line) for line in stream]
    if [call["call_id"] for call in calls] != proof["original_call_ids"]:
        raise ValueError("Canonical targets differ from audited actual calls")
    decisions = []
    for call in calls:
        if call["trial_id"] != record["trial_id"]:
            raise ValueError("Wrong source Trial identity")
        if call["source_journal_sha256"] != record["source_journal_sha256"]:
            raise ValueError("Wrong raw Recorder journal")
        if not call.get("raw_sse_response_semantics_verified"):
            raise ValueError("Unverified raw response projection")
        if call.get("stop_reason") not in ("stop", "toolUse"):
            raise ValueError("Non-normal model target termination")
        decisions.append(project_decision(tokenizer, call))
    if not decisions:
        raise ValueError("No actual assistant decisions")
    return decisions


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return {"relative_path": path.name, "sha256": file_sha(path)}


def build_corpus(tokenizer, *, pool, success_records, exports, output, provenance):
    """Write an auditable staging build; final acceptance requires full Loader audit.

    ``exports`` maps exact Trial UUIDs to canonical request/response exports.
    No source from an unqualified trial is loaded. Existing builds are immutable.
    """
    if pool not in ("calibforge", "tmax"):
        raise ValueError("Unknown fresh task pool")
    records = admitted_records(success_records, pool)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    caches = {cap: {split: [] for split in ("train", "validation")} for cap in CAPS}
    mappings = {cap: [] for cap in CAPS}
    tasks = []
    quarantines = []
    family_splits = {}
    for record in records:
        task, trial, split = record["task_id"], record["trial_id"], record["split"]
        family_splits[record["family_id"]] = split
        source = {
            "pool": pool,
            "task_id": task,
            "trial_id": trial,
            "split": split,
            "family_id": record["family_id"],
            "source_revision": record["source_revision"],
            "source_journal_sha256": record["source_journal_sha256"],
            "result_sha256": record["result_sha256"],
            "raw_export_sha256": record["student_proof"]["raw_export_sha256"],
        }
        try:
            decisions = project_trial(tokenizer, record, exports[trial])
        except (ValueError, AssertionError, TypeError, KeyError) as error:
            # Preserve the whole trace; never admit an apparently usable text prefix.
            quarantines.append(
                dict(
                    **source,
                    quarantine="WHOLE_TRACE_STRUCTURE_OR_SEMANTICS",
                    reason=str(error),
                )
            )
            continue
        views = {}
        for cap in CAPS:
            encoded = make_segments(decisions, cutoff=cap, processor_tokens=66)
            views[str(cap)] = coverage(decisions, encoded)
            for item in encoded["quarantine"]:
                quarantines.append(
                    dict(**source, quarantine="OVERLONG_DECISION", **item)
                )
            for sample in encoded["samples"]:
                row_index = len(caches[cap][split])
                caches[cap][split].append(sample["row"])
                mappings[cap].append(
                    dict(
                        **source,
                        row_index=row_index,
                        sample_sha256=digest(sample["row"]),
                        **sample["evidence"],
                    )
                )
        targets = [
            dict(
                **source,
                ordinal=i,
                stage=stage(i, len(decisions)),
                closing_target=i == len(decisions) - 1,
                **d.evidence,
            )
            for i, d in enumerate(decisions)
        ]
        target_file = output / (trial + "-targets.json")
        target_identity = _write_json(target_file, targets)
        tasks.append(
            dict(
                **source,
                original_targets=len(decisions),
                views=views,
                target_mapping=target_identity,
            )
        )
    # Arrow caches use the existing LF tokenized_path contract, not a new loader.
    from datasets import Dataset, DatasetDict, Features, Sequence, Value

    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "images": Sequence(Value("string")),
            "videos": Sequence(Value("string")),
            "audios": Sequence(Value("string")),
        }
    )
    artifacts = {}
    summaries = {}
    for cap in CAPS:
        name = str(cap)
        cache_path = output / ("cache-" + name)
        DatasetDict(
            {
                split: Dataset.from_dict(
                    {key: [row[key] for row in rows] for key in features},
                    features=features,
                )
                for split, rows in caches[cap].items()
            }
        ).save_to_disk(str(cache_path))
        map_file = output / ("SAMPLE_MAPPING-" + name + ".jsonl")
        for split in ("train", "validation"):
            subset = [m for m in mappings[cap] if m["split"] == split]
            weights = uniform_segment_mean_weights(subset)
            for mapping, coefficient in zip(subset, weights["per_token_coefficients"]):
                mapping["uniform_segment_mean_per_supervised_token_coefficient"] = (
                    coefficient
                )
        with map_file.open("w") as stream:
            for mapping in mappings[cap]:
                stream.write(json.dumps(mapping, ensure_ascii=False) + "\n")
        artifacts[map_file.name] = file_sha(map_file)
        summaries[name] = {}
        for split in ("train", "validation"):
            members = [t for t in tasks if t["split"] == split]
            stats = [t["views"][name] for t in members]
            supervised = sum(s["supervised_tokens"] for s in stats)
            weights = uniform_segment_mean_weights(
                [m for m in mappings[cap] if m["split"] == split]
            )
            summaries[name][split] = {
                "qualified_tasks": sum(r["split"] == split for r in records),
                "semantically_projected_tasks": len(members),
                "retained_independent_tasks": sum(
                    s["retained_targets"] > 0 for s in stats
                ),
                "complete_tasks": sum(s["all_targets_retained"] for s in stats),
                "closing_tasks": sum(s["closing_target_retained"] for s in stats),
                "original_targets": sum(s["original_targets"] for s in stats),
                "retained_targets": sum(s["retained_targets"] for s in stats),
                "segments": sum(s["segments"] for s in stats),
                "multi_assistant_samples": sum(
                    s["multi_assistant_samples"] for s in stats
                ),
                "supervised_tokens": supervised,
                "supervision_token_classes": {
                    k: sum(s["supervision_token_classes"][k] for s in stats)
                    for k in CLASSES
                },
                "stage_target_counts": {
                    k: sum(s["stage_target_counts"][k] for s in stats)
                    for k in ("early", "middle", "late")
                },
                "task_segment_share": {
                    t["task_id"]: weights["task_segment_share"].get(t["task_id"], 0)
                    for t in members
                },
                "task_supervised_token_share": {
                    t["task_id"]: (
                        t["views"][name]["supervised_tokens"] / supervised
                        if supervised
                        else 0
                    )
                    for t in members
                },
                "weighting_policy": "Uniform segment-mean analytical shares and distinct supervised-token shares; no reweighting or resampling. Microbatch1 sample-mean basis excludes distributed tail padding and requires matching the frozen native training batch contract.",
            }
        for file in sorted(cache_path.rglob("*")):
            if file.is_file():
                artifacts[str(file.relative_to(output))] = file_sha(file)
    artifacts["TASK_AUDIT.json"] = _write_json(output / "TASK_AUDIT.json", tasks)[
        "sha256"
    ]
    artifacts["QUARANTINE.json"] = _write_json(output / "QUARANTINE.json", quarantines)[
        "sha256"
    ]
    manifest = {
        "status": "STAGING_PENDING_FULL_CACHE_LOADER_AND_COLLATOR_AUDIT",
        "pool": pool,
        "provenance": provenance,
        "qualified_independent_tasks": len(records),
        "primary_trajectories": len(records),
        "semantically_projected_tasks": len(tasks),
        "qualified_success_is_not_a_length_retention_count": True,
        "length_views": summaries,
        "family_splits": family_splits,
        "artifacts": artifacts,
        "target_scope": "Native agent decisions; auxiliary summaries retained in raw evidence and actual visible history, not supervised as agent decisions",
        "stage_policy": "Chronological decision-index thirds, early/middle/late; not inferred semantic quality stages",
        "multi_assistant_sample_definition": "A segment containing two or more newly supervised Assistant targets",
        "processor_extra_tokens": 66,
        "pad_to_multiple_of": 8,
        "cap_includes_input_target_processor_and_padding": True,
        "historical_thinking_policy": "Pinned official Qwen Jinja last-real-user policy, identical at serving",
        "target_policy": "Full Thinking+Action and all new Assistant targets exactly once; overlapping history input only",
        "reasoning_projection": "DeepSeek reasoning_content to official Qwen field; provider-private signatures omitted",
        "observation_basis": "Actual provider payload after native toolResultBudget; never restored from larger raw output",
        "unsupported_multimodal_policy": "Whole-trace quarantine; never remove images and retain image-dependent targets",
        "truncation": False,
        "released": False,
    }
    _write_json(output / "DATA_MANIFEST.json", manifest)
    return manifest
