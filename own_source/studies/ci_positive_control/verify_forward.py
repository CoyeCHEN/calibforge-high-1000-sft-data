"""Three real cached traces, actual LF collator/model and shifted CE reference."""
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer


def main(root):
    root = Path(root)
    cfg = json.loads((root/"configs/train-r2.json").read_text())["LF_arguments"]
    probe = cfg | dict(deepspeed=None, do_train=False, do_eval=False)
    ma, da, ta, fa, ga = get_train_args(probe)
    module = load_tokenizer(ma)
    tok = module["tokenizer"]
    template = get_template_and_fix_tokenizer(tok,da)
    model = load_model(tok,ma,fa,is_trainable=True).to("cuda").eval()
    collator = SFTDataCollatorWith4DAttentionMask(template=template,model=model,
        tokenizer=tok,processor=module["processor"],pad_to_multiple_of=8,label_pad_token_id=-100,
        block_diag_attn=False,attn_implementation="flash_attention_2",compute_dtype=torch.bfloat16)
    data = load_from_disk(cfg["tokenized_path"])["train"]
    results=[]
    for item in json.loads((root/"checks/forward-rows-r2.json").read_text()):
        row=data[item["index"]]
        batch=collator([copy.deepcopy(row)])
        batch={k:v.to("cuda") if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(**batch,skip_logits=False)
        labels=batch["labels"][:,1:].reshape(-1)
        reference=F.cross_entropy(output.logits[:,:-1].reshape(-1,output.logits.shape[-1]).float(),
                                  labels,ignore_index=-100,reduction="sum")/labels.ne(-100).sum()
        delta=float(abs(output.loss.float()-reference))
        assert bool(torch.isfinite(reference)) and delta<.003,(delta,float(output.loss),float(reference))
        results.append(dict(**item,loss=float(output.loss),independent_shifted_token_ce=float(reference),
                            absolute_difference=delta,input_tokens=len(row['input_ids']),
                            shifted_supervision=int(labels.ne(-100).sum())))
        del output,reference,batch
    report=dict(status="PASSED",rows=results,scope="Actual LF forward vs independent CE; no optimizer update",
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    (root/"checks/actual-forward-r2.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main(sys.argv[1])
