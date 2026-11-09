"""Actual LF model-aware collator/mRoPE, then bounded real M0 CPU forward."""
import copy
import hashlib
import json
from pathlib import Path
import time
import torch

A=Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
X=Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')
M='/data/models/Qwen/Qwen3.5-4B'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    torch.set_num_threads(1)
    from transformers import AutoConfig,AutoModelForImageTextToText,AutoTokenizer,AutoProcessor
    from datasets import load_from_disk
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask,get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    config=AutoConfig.from_pretrained(M,local_files_only=True)
    with torch.device('meta'):meta=AutoModelForImageTextToText.from_config(config,attn_implementation='eager')
    tok=AutoTokenizer.from_pretrained(M,local_files_only=True);processor=AutoProcessor.from_pretrained(M,local_files_only=True)
    template=get_template_and_fix_tokenizer(tok,DataArguments(template='qwen3_5',enable_thinking=False))
    def collate(model):return SFTDataCollatorWith4DAttentionMask(template=template,model=model,tokenizer=tok,processor=processor,
        pad_to_multiple_of=8,label_pad_token_id=-100,block_diag_attn=False,attn_implementation='flash_attention_2',compute_dtype=torch.bfloat16)
    collator=collate(meta);data=load_from_disk(str(X/'data/cache-262144-r2'));report={}
    for split in ['train','validation']:
        count=0;extra=set();shapes=set();deltas=set()
        for row in data[split]:
            b=collator([copy.deepcopy(row)]);keep=b['attention_mask'][0].bool();pos=b['position_ids']
            assert b['input_ids'][0][keep].tolist()==row['input_ids']
            assert b['labels'][0][keep].tolist()==row['labels']
            assert (b['labels'][0][~keep]==-100).all()
            assert torch.equal(pos[:,0,keep],torch.arange(len(row['input_ids'])).expand(pos.shape[0],-1)),pos[:,0,keep]
            extra.add(int((~keep).sum()));shapes.add(tuple(pos.shape[:2]));deltas.update(b.get('rope_deltas',torch.zeros(1)).reshape(-1).tolist());count+=1
        report[split]=dict(rows=count,ids_labels_equal=True,visible_positions_contiguous_from_zero=True,
            position_leading_shapes=sorted(shapes),masked_extra_token_counts=sorted(extra),rope_deltas=sorted(deltas))
    (A/'MODEL_AWARE_COLLATOR_ALL.json').write_text(json.dumps(report,indent=2)+'\n');print('MODEL_AWARE_COLLATOR_ALL',report,flush=True)
    frozen=torch.load(A/'M0_REAL_HIDDEN.pt',weights_only=True);rowno=frozen['proof'][0]['row'];row=data['train'][rowno]
    model=AutoModelForImageTextToText.from_pretrained(M,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    b=collate(model)([copy.deepcopy(row)]);torch.save(b,A/'ACTUAL_MODEL_AWARE_FORWARD_BATCH.pt')
    received={};hidden={}
    def pre(module,args,kwargs):
        received.update({k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in kwargs.items()if torch.is_tensor(v)})
    def post(module,args,output):hidden['x']=output.last_hidden_state
    model.register_forward_pre_hook(pre,with_kwargs=True);model.model.register_forward_hook(post)
    t=time.time()
    # This is a CPU eager reference, not the production FA2/Liger kernel.
    with torch.no_grad():
        out=model(**b,use_cache=False)
        ref=torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]).float(),b['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='sum')/b['labels'][:,1:].ne(-100).sum()
        positions=torch.tensor(frozen['proof'][0]['selected_logit_positions'])
        expected=frozen['features'][0][0];actual=hidden['x'][0,positions]
        diff=actual-expected
    result=dict(scope='actual M0 CPU eager/FP32, actual model-aware LF collator, real original row, no truncation',
        row=rowno,trace_id=frozen['proof'][0]['trace_id'],received_forward_kwargs=received,
        cache_input_sha256=digest(row['input_ids']),cache_labels_sha256=digest(row['labels']),
        actual_forward_loss=float(out.loss),independent_shifted_ce=float(ref),loss_abs_difference=float(abs(out.loss-ref)),
        hidden_difference_vs_no_dummy_collator=dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/expected.norm())),
        elapsed_seconds=time.time()-t,cuda_initialized=torch.cuda.is_initialized())
    (A/'M0_ACTUAL_COLLATOR_FORWARD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
