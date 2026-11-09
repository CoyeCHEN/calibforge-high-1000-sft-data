"""Group-disjoint selection and source-preserving extraction, before encoding."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from huggingface_hub import HfFileSystem
from inventory import read_shard, digest, REVISION, DATASET


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def ngrams(text):
    words = normalized(text).split()
    return {tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))}


def source_keys(row):
    keys = {"notebook:" + row["notebook_id"]}
    if row.get("kaggle_dataset_name"):
        keys.add("declared-kaggle:" + row["kaggle_dataset_name"].casefold())
    generic = {"train.csv", "test.csv", "data.csv", "dataset.csv", "sample_submission.csv"}
    for filename in row.get("files_used") or []:
        p = Path(filename)
        if p.name.casefold() not in generic and len(p.name) >= 7:
            keys.add("specific-filename:" + p.name.casefold())
        match = re.search(r"(?:kaggle/)?input/([^/]+)/[^/]+", filename)
        if match:
            keys.add("input-dataset:" + match[1].casefold())
    return keys


def check_trace(row):
    messages = row["messages"]
    if not messages or messages[0]["role"] not in ("user", "system"):
        return "invalid_start"
    pending = 0
    python_calls = observations = assistant = 0
    final = None
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        role = message["role"]
        if role == "tool":
            if pending != 1 or not message.get("content"):
                return "orphan_or_empty_observation"
            pending = 0
            observations += 1
        elif role == "assistant":
            if pending:
                return "missing_observation"
            assistant += 1
            if len(calls) != 1:
                return "not_sequential_single_call"
            function = calls[0]["function"]
            name = function["name"]
            args = function["arguments"]
            if name == "add_and_execute_jupyter_code_cell":
                if not isinstance(args.get("code"), str) or not args["code"].strip():
                    return "empty_code"
                python_calls += 1
                pending = 1
            elif name == "final_answer":
                if index != len(messages) - 1 or not isinstance(args.get("answer"), str):
                    return "invalid_final_answer"
                final = args["answer"]
            else:
                return "unknown_source_tool"
        elif pending:
            return "role_interrupted_tool_pair"
    if pending or final is None or python_calls < 2 or observations != python_calls:
        return "incomplete_or_not_multiturn"
    # No reference answer is synthesized or substituted. Disagreements are
    # reported separately; E2B provenance establishes execution, not correctness.
    return dict(assistant_decisions=assistant, python_calls=python_calls,
                observations=observations, final_answer=final,
                reference_answer_verbatim_in_final=normalized(row["answer"]) in normalized(final))


def assign_source_groups(rows):
    # Connected components prevent a notebook, declared Kaggle source, or
    # specific shared input filename from crossing the split, even when the
    # upstream source mapping disagrees between notebooks.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        keys = sorted(source_keys(row))
        for key in keys[1:]:
            union(keys[0], key)
    for row in rows:
        row["task_group"] = digest(find(sorted(source_keys(row))[0]))
        row["split"] = "validation" if int(row["task_group"][:8], 16) % 10 == 0 else "train"


def evaluation_overlap_checker(root):
    root = Path(root)
    eval_questions = []
    for path in sorted((root / "data/cibench_dataset").rglob("*.ipynb")):
        if "/datasources/" in str(path):
            continue
        notebook = json.loads(path.read_text())
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "markdown":
                content = cell.get("source", "")
                if isinstance(content, list):
                    content = "".join(content)
                eval_questions.append(content.split("\n\nThought: ")[0])
    for path in (root / "data/dabstep").glob("*.jsonl"):
        for line in path.open():
            item = json.loads(line)
            question = item.get("question", item.get("task", ""))
            if question:
                eval_questions.append(question)
    eval_normal = {normalized(q) for q in eval_questions}
    eval_ngrams = [ngrams(q) for q in eval_questions if len(normalized(q).split()) >= 8]
    inverted = defaultdict(set)
    for i, grams in enumerate(eval_ngrams):
        for gram in grams:
            inverted[gram].add(i)

    def overlap(question):
        normal = normalized(question)
        if normal in eval_normal:
            return "exact_eval_question"
        grams = ngrams(question)
        candidates = set()
        for gram in grams:
            candidates.update(inverted.get(gram, ()))
        for i in candidates:
            other = eval_ngrams[i]
            if grams and len(grams & other) / len(grams | other) >= .8:
                return "near_eval_question_4gram_jaccard_ge_0.8"
        return None

    return overlap, len(eval_questions)


def main(root):
    root = Path(root)
    rows = [json.loads(line) for p in sorted((root / "data/inventory-shards").glob("*.jsonl"))
            for line in p.open()]
    assert len(list((root / "data/inventory-shards").glob("*.jsonl"))) == 103
    assign_source_groups(rows)
    overlap, eval_question_count = evaluation_overlap_checker(root)

    counts = Counter()
    selected = {"train": [], "validation": []}
    question_seen = set()
    trace_seen = set()
    all_index = {r["id"]: r for r in rows}
    fs = HfFileSystem()
    raw_dir = root / "data/raw-shards"
    raw_dir.mkdir(exist_ok=True)
    # The upstream release is shuffled. Take shards in order; record every
    # exclusion and stop only after obtaining the pre-registered two quotas.
    for shard in range(103):
        raw_file = raw_dir / f"{shard:03d}.jsonl"
        if not raw_file.exists():
            path = raw_file.with_suffix(".tmp")
            with path.open("w") as handle:
                for source in read_shard(fs, shard, ["id", "messages", "tools", "question", "answer",
                                                     "executor_type", "files_used", "packages_used",
                                                     "kaggle_dataset_name", "edu_score"]):
                    handle.write(json.dumps(source, ensure_ascii=False) + "\n")
            path.replace(raw_file)
        for line in raw_file.open():
            source = json.loads(line)
            meta = all_index[source["id"]]
            split = meta["split"]
            quota = 2000 if split == "train" else 200
            if len(selected[split]) >= quota:
                counts["quota_already_met"] += 1
                continue
            if source["executor_type"] != "e2b":
                counts["simulated_executor_excluded"] += 1
                continue
            if not normalized(source["answer"]) or not normalized(source["question"]):
                counts["empty_question_or_reference"] += 1
                continue
            reason = overlap(source["question"])
            if reason:
                counts[reason] += 1
                continue
            question_key = normalized(source["question"])
            trace_hash = digest(dict(messages=source["messages"], tools=source["tools"]))
            if question_key in question_seen or trace_hash in trace_seen:
                counts["duplicate_question_or_trace"] += 1
                continue
            quality = check_trace(source)
            if isinstance(quality, str):
                counts[quality] += 1
                continue
            # A positive-control corpus must not knowingly teach a final answer
            # that contradicts its released reference. This conservative check
            # does not claim all retained answers are semantically verified.
            if not quality["reference_answer_verbatim_in_final"]:
                counts["reference_answer_not_verbatim_in_final"] += 1
                continue
            question_seen.add(question_key)
            trace_seen.add(trace_hash)
            selected[split].append(dict(source=source, provenance=meta, quality=quality,
                                        source_trace_sha256=trace_hash))
        print(json.dumps(dict(shard=shard, train=len(selected["train"]),
                              validation=len(selected["validation"]), exclusions=dict(counts))), flush=True)
        if len(selected["train"]) == 2000 and len(selected["validation"]) == 200:
            break
    for split, values in selected.items():
        path = root / "data" / (split + "-raw.jsonl")
        with path.open("x") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    a = {r["provenance"]["task_group"] for r in selected["train"]}
    b = {r["provenance"]["task_group"] for r in selected["validation"]}
    assert not a & b
    result = dict(dataset=DATASET, revision=REVISION, split="non_thinking",
                  requested_train_trajectories=2000, requested_validation_trajectories=200,
                  excluded=dict(counts), eval_questions_screened=eval_question_count,
                  source_group_disjoint=True, grouping="connected notebook/declared Kaggle/specific filename/input dataset",
                  caveat="Kaggle mapping is upstream metadata, sometimes inconsistent; conservative connected grouping used. Verbatim answer agreement is not an independent verifier.",
                  executor_policy="e2b only, no simulated results renamed, no replayed rows",
                  cohorts={s: dict(trajectories=len(v), questions=len({normalized(r['source']['question']) for r in v}),
                                  notebooks=len({r['provenance']['notebook_id'] for r in v}),
                                  source_groups=len({r['provenance']['task_group'] for r in v}),
                                  declared_kaggle_datasets=len({r['source']['kaggle_dataset_name'] for r in v}),
                                  assistant_decisions=sum(r['quality']['assistant_decisions'] for r in v),
                                  python_calls=sum(r['quality']['python_calls'] for r in v),
                                  raw_sha256=hashlib.sha256((root/'data'/(s+'-raw.jsonl')).read_bytes()).hexdigest())
                           for s, v in selected.items()})
    (root / "checks/selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
