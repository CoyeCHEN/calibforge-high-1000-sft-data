"""Full real LF collator check; readable per-token evidence for real traces."""
import copy
import json
from pathlib import Path
import sys

import torch
from datasets import load_from_disk
from transformers import AutoProcessor
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from sft.qwen35 import load_tokenizer, digest


def main(root):
    root = Path(root)
    torch.set_num_threads(1)
    tok = load_tokenizer()
    processor = AutoProcessor.from_pretrained("/data/models/Qwen/Qwen3.5-4B",local_files_only=True)
    template = get_template_and_fix_tokenizer(tok,DataArguments(template="qwen3_5",enable_thinking=False))
    collator = SFTDataCollatorWith4DAttentionMask(template=template, tokenizer=tok, processor=processor,
        pad_to_multiple_of=8, label_pad_token_id=-100, block_diag_attn=False,
        attn_implementation="flash_attention_2", compute_dtype=torch.bfloat16)
    data = load_from_disk(str(root/"data/cache-262144-r2"))
    counts = {}
    for split in ["train","validation"]:
        total = 0
        index = [json.loads(line) for line in (root/"data"/(split+"-encoding-index-r2.jsonl")).open()]
        for n, row in enumerate(data[split]):
            batch = collator([copy.deepcopy(row)])
            keep = batch["attention_mask"][0].bool()
            assert batch["input_ids"][0][keep].tolist() == row["input_ids"]
            assert batch["labels"][0][keep].tolist() == row["labels"]
            assert bool((batch["labels"][0][~keep] == -100).all())
            assert sum(x!=-100 for x in row["labels"][1:]) == index[n]["supervised_tokens"]
            total += index[n]["supervised_tokens"]
        counts[split] = dict(rows=len(data[split]), shifted_supervised_tokens=total)
    # Match three complete public Iris traces to their actual full-sequence rows.
    chosen = [json.loads(line) for line in (root/"data/replay-traces.jsonl").open()]
    by_trace = {item["trace_id"]:n for n,item in enumerate(
        json.loads(line) for line in (root/"data/train-encoding-index-r2.jsonl").open())}
    mask_path = root/"reports/TOKEN_LABEL_EXAMPLES-r2.md"
    with mask_path.open("w") as doc:
        doc.write("# Actual CI token/label evidence\n\n"
                  "Training and inference are Off. The official empty closed thinking prefix is supplied as input, not generated. "
                  "Only assistant completion bodies, native tool calls and im_end EOS carry labels. User and real environment observations are -100. "
                  "Causal loss compares logits at i-1 to label at i, exactly once. Every assistant target is assigned once.\n\n")
        forward_rows = []
        for item in chosen:
            ident = item["source"]["id"]
            n = by_trace[ident]
            row = data["train"][n]
            forward_rows.append(dict(index=n, trace_id=ident))
            doc.write(f"## {ident}\n\nFull input and supervision SHA: `{digest(row['input_ids'])}` / `{digest(row['labels'])}`.\n\n")
            spans = []
            mask = row["labels"]
            for i in range(1,len(mask)):
                if (mask[i]!=-100)!=(mask[i-1]!=-100):spans.append(i)
            positions = sorted({i for boundary in spans for i in range(max(0,boundary-5),min(len(mask),boundary+9))})
            doc.write("| Position | Input token ID | Decoded token | Label | Predicted from |\n|---:|---:|---|---:|---:|\n")
            for i in positions:
                token = repr(tok.decode([row['input_ids'][i]],skip_special_tokens=False)).replace('|','\\|')
                doc.write(f"| {i} | {row['input_ids'][i]} | {token} | {mask[i]} | {i-1 if mask[i]!=-100 else 'masked'} |\n")
            doc.write("\n")
    (root/"checks/forward-rows-r2.json").write_text(json.dumps(forward_rows,indent=2)+"\n")
    result = dict(status="PASSED", collator="actual LF SFTDataCollatorWith4DAttentionMask",
                  counts=counts, all_unpadded_ids_labels_exact=True, padding_masked=True,
                  cuda_initialized=torch.cuda.is_initialized(), readable_masks=str(mask_path))
    (root/"checks/loader-r2.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main(sys.argv[1])
