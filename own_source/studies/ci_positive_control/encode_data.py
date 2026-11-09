"""Use the production Qwen codec/decision merger; preserve full trajectories."""
from collections import Counter
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
import numpy as np
from sft.qwen35 import load_tokenizer, digest
from sft.teacher_segments import project_decision, make_segments
from tool_contract import schemas, messages


def main(root):
    root = Path(root)
    tok = load_tokenizer()
    vocab_size = len(tok)
    summary = {}
    cache = {}
    projection_path = root / "configs/sglang-template-tools.json"
    projection = json.loads((root / "checks/actual-server-preprocessor.json").read_text())
    if (projection.get("processing_entry") != "_process_messages"
            or projection.get("tool_projection_file_sha256") != hashlib.sha256(projection_path.read_bytes()).hexdigest()):
        raise ValueError("Capture fresh tools through the full serving request entry before encoding")
    served_tools = json.loads(projection_path.read_text())
    tokenizer_path = root / 'data/tokenizer'
    tok.save_pretrained(tokenizer_path)
    def semantic_tools(tools):
        import copy
        result=copy.deepcopy(tools)
        for tool in result:
            if tool["function"].get("strict") is False:
                del tool["function"]["strict"]
        return result
    for split in ["train", "validation"]:
        rows, evidence, lengths = [], [], []
        counts = Counter()
        for line in (root / "data" / (split + "-raw.jsonl")).open():
            item = json.loads(line)
            source = item["source"]
            native = messages(source["messages"], digest(source["id"])[:16])
            logical_tools = schemas(root, source["tools"])
            if semantic_tools(served_tools) != logical_tools:
                raise ValueError("Published tool semantics differ from the frozen serving schema")
            tools = served_tools
            decision_count=sum(m['role']=='assistant'for m in native)
            decisions=[]
            for index,response in enumerate(native):
                if response['role']!='assistant':continue
                packet=dict(call_id=source['id']+f':{index}',context_fidelity='published_messages',
                    provider_payload=dict(messages=native[:index],tools=tools),assistant_response=response)
                decisions.append(project_decision(tok,packet,enable_thinking=False,projection_source='published_messages'))
            merged=make_segments(decisions,cutoff=262144)
            counts['newly_encoded_trajectories']+=1
            if merged["quarantine"]:
                raise ValueError("Overlength trace requires an explicit decision, never truncation")
            assert merged["all_targets_exactly_once"]
            for sample in merged["samples"]:
                row = sample["row"]
                assert row["labels"][0] == -100
                assert all(a == -100 or a == b for a, b in zip(row["labels"], row["input_ids"]))
                assert all(0 <= i < vocab_size for i in row["input_ids"])
                counts["supervised_tokens"] += sum(i != -100 for i in row["labels"][1:])
                lengths.append(len(row["input_ids"]))
                rows.append(row)
                evidence.append(dict(trace_id=source["id"], task_group=item["provenance"]["task_group"],
                                     original_trace_sha256=item["source_trace_sha256"],
                                     tools_sha256=digest(tools), **sample["evidence"]))
            counts["trajectories"] += 1
            counts["assistant_decisions"] += decision_count
            counts["python_calls"] += item["quality"]["python_calls"]
            if counts["trajectories"] % 100 == 0:
                print(split, dict(counts), flush=True)
        cache[split] = Dataset.from_list(rows)
        with (root / "data" / (split + "-encoding-index-r2.jsonl")).open("x") as handle:
            for item in evidence:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[split] = dict(**counts, training_rows=len(rows),
                             length_percentiles={str(q): float(np.percentile(lengths, q))
                                                 for q in [50, 75, 90, 95, 99, 100]},
                             input_sha256=digest([r["input_ids"] for r in rows]),
                             labels_sha256=digest([r["labels"] for r in rows]),
                             all_targets_exactly_once=True, truncation=False)
    DatasetDict(cache).save_to_disk(str(root / "data/cache-262144-r2"), max_shard_size="256MB")
    result = dict(model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                  template_sha256=digest(tok.chat_template), thinking=False, compaction=False,
                  cutoff_len=262144, global_loss="sum shifted supervised CE / global update supervised token count",
                  **summary)
    result.update(conditioning_source="Published ChatML messages; original teacher HTTP payload not released",
                  serving_tool_projection="Actual full SGLang _process_messages tools passed to tokenizer; exclude_unset and field order",
                  original_message_and_code_unchanged=True)
    result['tokenizer'] = dict(path='data/tokenizer',
        runtime_class=type(tok).__module__ + '.' + type(tok).__qualname__,
        transformers_version=version('transformers'), tokenizers_version=version('tokenizers'),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(tokenizer_path.iterdir()) if p.is_file()})
    (root / "checks/encoding-r2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument('root')
    args=parser.parse_args();main(args.root)
