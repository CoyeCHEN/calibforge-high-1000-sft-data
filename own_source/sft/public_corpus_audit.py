"""Full-cache native LF audit; CPU loss reference, with no training admission side effect."""

import copy
import inspect
import json
from pathlib import Path

from .public_corpus import CAPS, file_sha, uniform_segment_mean_weights
from .qwen35 import TEMPLATE_SHA, digest


def _numerical_reference(collator, windows):
    if not windows:
        return {
            "passed": None,
            "status": "NOT_APPLICABLE_NO_RETAINED_ROWS",
            "scope": "Empty corpus retained honestly; no synthetic targets introduced for a numerical reference",
        }
    import torch
    import torch.nn.functional as F
    from transformers import LlamaConfig, LlamaForCausalLM

    # Vocabulary remapping is limited to diagnostic windows; never write them to a corpus.
    batch = collator(copy.deepcopy(windows))
    ids = batch["input_ids"] % 126 + 1
    labels = batch["labels"].clone()
    mask = labels != -100
    labels[mask] = labels[mask] % 126 + 1
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        attention_dropout=0,
    )
    cfg._attn_implementation = "eager"
    native = LlamaForCausalLM(cfg)
    reference = copy.deepcopy(native)
    result = native(
        input_ids=ids,
        attention_mask=batch["attention_mask"],
        labels=labels,
        use_cache=False,
    )
    result.loss.backward()
    logits = reference(
        input_ids=ids, attention_mask=batch["attention_mask"], use_cache=False
    ).logits
    manual = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, 128),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    manual.backward()
    errors = []
    for (name, a), (other, b) in zip(
        native.named_parameters(), reference.named_parameters()
    ):
        assert name == other
        if a.grad is not None:
            errors.append(float((a.grad - b.grad).abs().max()))
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=3e-5)
    assert abs(float(result.loss.detach() - manual.detach())) < 2e-6
    return {
        "native_loss": float(result.loss.detach()),
        "manual_shift_once_loss": float(manual.detach()),
        "maximum_gradient_error": max(errors),
        "passed": True,
        "scope": "Tiny random CPU Llama, actual final-cache label windows; shift/mask/gradient reference only, no Qwen 4B or GPU kernel equivalence claim",
    }


def audit_corpus(root, *, model_path):
    import torch
    from llamafactory.data import (
        SFTDataCollatorWith4DAttentionMask,
        get_dataset,
        get_template_and_fix_tokenizer,
    )
    from llamafactory.hparams import DataArguments, ModelArguments, TrainingArguments
    from llamafactory.model import load_tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = Path(root)
    manifest_path = root / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["artifacts"].items():
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or file_sha(root / path) != expected
        ):
            raise ValueError("Final-cache artifact identity changed")
    model_args = ModelArguments(model_name_or_path=str(model_path))
    module = load_tokenizer(model_args)
    tokenizer = module["tokenizer"]
    if digest(tokenizer.chat_template) != TEMPLATE_SHA:
        raise ValueError("Official template differs")
    template = get_template_and_fix_tokenizer(
        tokenizer,
        DataArguments(template="qwen3_5", enable_thinking=True, preserve_thinking=True),
    )
    assert digest(tokenizer.chat_template) == TEMPLATE_SHA
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        tokenizer=tokenizer,
        processor=module["processor"],
        model=None,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        block_diag_attn=False,
        neat_packing=False,
        attn_implementation="sdpa",
        compute_dtype=torch.float32,
    )
    records = []
    windows = []
    for cap in CAPS:
        mappings = [
            json.loads(line) for line in (root / f"SAMPLE_MAPPING-{cap}.jsonl").open()
        ]
        identities = {(m["split"], m["row_index"]): m for m in mappings}
        if len(identities) != len(mappings):
            raise ValueError("Duplicate cache row identity")
        args = DataArguments(
            template="qwen3_5",
            tokenized_path=str(root / f"cache-{cap}"),
            cutoff_len=cap,
            enable_thinking=True,
            preserve_thinking=True,
            train_on_prompt=False,
            packing=False,
        )
        training_args = TrainingArguments(
            output_dir=str(root / f"audit-output-{cap}"),
            use_cpu=True,
            do_train=True,
            do_eval=True,
            report_to="none",
            dataloader_num_workers=0,
        )
        loaded = get_dataset(
            template,
            model_args,
            args,
            training_args,
            "sft",
            tokenizer,
            module["processor"],
        )
        seen = set()
        for split, key in [("train", "train_dataset"), ("validation", "eval_dataset")]:
            subset = [m for m in mappings if m["split"] == split]
            weights = uniform_segment_mean_weights(subset)
            for mapping, coefficient in zip(subset, weights["per_token_coefficients"]):
                assert (
                    mapping["uniform_segment_mean_per_supervised_token_coefficient"]
                    == coefficient
                )
            declared = manifest["length_views"][str(cap)][split]["task_segment_share"]
            assert set(weights["task_segment_share"]) <= set(declared)
            assert declared == {
                task: weights["task_segment_share"].get(task, 0) for task in declared
            }
            data = loaded[key]
            assert len(data) == manifest["length_views"][str(cap)][split]["segments"]
            for index, row in enumerate(data):
                mapping = identities[(split, index)]
                assert digest(row) == mapping["sample_sha256"]
                assert digest(row["input_ids"]) == mapping["input_sha256"]
                assert digest(row["labels"]) == mapping["labels_sha256"]
                expected = [-100] * len(row["input_ids"])
                for target in mapping["decision_mapping"]:
                    left, right = target["supervised_range"]
                    assert left > 0 and right <= len(expected)
                    assert all(x == -100 for x in expected[left:right])
                    expected[left:right] = row["input_ids"][left:right]
                    # Representative exact-mask windows cover beginnings and endings.
                    if cap == 262144 and len(windows) < 8:
                        for centre in (left, right - 1):
                            start = max(0, centre - 24)
                            end = min(len(expected), centre + 32)
                            windows.append(
                                {
                                    "input_ids": row["input_ids"][start:end],
                                    "attention_mask": [1] * (end - start),
                                    "labels": row["labels"][start:end],
                                    "images": [],
                                    "videos": [],
                                    "audios": [],
                                }
                            )
                assert row["labels"] == expected
                batch = collator([copy.deepcopy(row)])
                length = len(expected)
                assert batch["input_ids"][0, :length].tolist() == row["input_ids"]
                assert batch["labels"][0, :length].tolist() == expected
                assert bool((batch["labels"][0, length:] == -100).all())
                assert batch["input_ids"].shape[-1] <= cap
                assert (
                    int((batch["labels"] != -100).sum()) == mapping["supervised_tokens"]
                )
                seen.add((split, index))
                records.append(
                    {
                        "cutoff": cap,
                        "split": split,
                        "row_index": index,
                        "task_id": mapping["task_id"],
                        "trial_id": mapping["trial_id"],
                        "sample_sha256": mapping["sample_sha256"],
                        "all_labels_verified": True,
                        "added_processor_and_padding_positions_masked": True,
                        "collated_tokens": batch["input_ids"].shape[-1],
                    }
                )
        assert seen == set(identities)
    numerical = _numerical_reference(collator, windows)
    assert not torch.cuda.is_initialized()
    runtime_files = {
        str(Path(inspect.getfile(obj))): file_sha(inspect.getfile(obj))
        for obj in (get_dataset, SFTDataCollatorWith4DAttentionMask)
    }
    report = {
        "passed": True,
        "pool": manifest["pool"],
        "manifest_sha256": file_sha(manifest_path),
        "all_caches_and_both_fixed_splits_read_through_formal_loader": True,
        "all_row_labels_checked": True,
        "native_runtime_files": runtime_files,
        "rows": records,
        "official_template_sha256": TEMPLATE_SHA,
        "model_loaded_for_training": False,
        "numerical_reference": numerical,
        "empty_corpus": not records,
        "sample_mean_weight_metadata_checked": True,
        "cuda_initialized": False,
        "release_admission": False,
    }
    target = root / "NATIVE_CACHE_AUDIT.json"
    if target.exists():
        raise FileExistsError("Audit identity already frozen")
    target.write_text(json.dumps(report, indent=2))
    return report
