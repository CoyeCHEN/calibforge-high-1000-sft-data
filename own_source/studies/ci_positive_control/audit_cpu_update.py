"""Bounded CPU M0 forward and full-vocabulary LM-head gradient/Adam control.

This deliberately does NOT claim to execute CUDA Liger or four-rank ZeRO-3.
The four-rank average is explicitly emulated; project normalization is executed.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
import sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F

A=Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
X=Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')

def metric(a,b):
    dot=aa=bb=err=0.
    for x,y in zip(a.reshape(-1).split(1000000),b.reshape(-1).split(1000000)):
        x=x.double();y=y.double();dot+=float((x*y).sum());aa+=float(x.square().sum());bb+=float(y.square().sum());err+=float((x-y).square().sum())
    return dict(norm_reference=aa**.5,norm_actual=bb**.5,norm_ratio=(bb/aa)**.5 if aa else None,
                cosine=dot/(aa*bb)**.5 if aa*bb else None,relative_l2=(err/aa)**.5 if aa else None,max_abs=float((a-b).abs().max()))

def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    from datasets import load_from_disk
    from transformers import AutoModelForImageTextToText,AutoTokenizer,Seq2SeqTrainingArguments
    from transformers.modeling_outputs import CausalLMOutput
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from llamafactory.hparams import FinetuningArguments,TrainingArguments
    from normalization import install
    data=load_from_disk(str(X/'data/cache-262144-r2'))['train']
    index=[json.loads(l)for l in (X/'data/train-encoding-index-r2.jsonl').open()]
    selected=sorted(range(len(index)),key=lambda i:len(data[i]['input_ids']))[:2]
    if '--reuse-hidden' in sys.argv:
        from safetensors import safe_open
        frozen=torch.load(A/'M0_REAL_HIDDEN.pt',weights_only=True);features=frozen['features'];proof=frozen['proof']
        root=Path('/data/models/Qwen/Qwen3.5-4B');mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        assert json.loads((root/'config.json').read_text())['tie_word_embeddings'] is True
        key='model.language_model.embed_tokens.weight'
        with safe_open(root/mapping[key],framework='pt',device='cpu')as f:weight=f.get_tensor(key).float()
        print('Reusing completed real M0 forwards; no repeated backbone computation',flush=True)
    else:
        print('M0 actual CPU forward rows',selected,flush=True)
        model=AutoModelForImageTextToText.from_pretrained('/data/models/Qwen/Qwen3.5-4B',local_files_only=True,
          dtype=torch.float32,attn_implementation='eager').eval()
        features=[];proof=[]
        with torch.no_grad():
            for i in selected:
                row=data[i];ids=torch.tensor([row['input_ids']]);labels=torch.tensor([row['labels']]);t=time.time()
                out=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                hidden=out.last_hidden_state if hasattr(out,'last_hidden_state')else out[0]
                positions=torch.nonzero(labels[0,1:]!=-100).flatten()[:64]
                features.append((hidden[0,positions].detach().clone(),labels[0,positions+1].clone()))
                proof.append(dict(row=i,trace_id=index[i]['trace_id'],full_input_tokens=len(row['input_ids']),
                  selected_logit_positions=positions.tolist(),target_positions=(positions+1).tolist(),
                  input_sha256=index[i]['input_sha256'],labels_sha256=index[i]['labels_sha256'],seconds=time.time()-t))
                print('M0 features',proof[-1],flush=True)
        weight=model.lm_head.weight.detach().clone();del model
        torch.save(dict(features=features,proof=proof),A/'M0_REAL_HIDDEN.pt')
    # Full M0 vocabulary and head. All backbone states are real, frozen M0 outputs.
    class Head(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(weight.clone())
        def forward(self,input_ids,labels,attention_mask=None,**kwargs):
            # Diagnostic input_ids holds a feature selector; the selected target is from the real cached row.
            which=int(input_ids[0,0]);n=labels.shape[1]-1;h=features[which][0][:n]
            logits=F.linear(h,self.weight).unsqueeze(0)
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')
            return CausalLMOutput(loss=loss,logits=logits)
    tok=AutoTokenizer.from_pretrained('/data/models/Qwen/Qwen3.5-4B',local_files_only=True)
    install(CustomSeq2SeqTrainer,'token')
    results=[]
    for ga in [8,2]:
        # Unequal supervision lengths; same tensor windows on both sides. DP average is emulated, not claimed as a distributed run.
        batches=[]
        for rank in range(4):
            rankrows=[]
            for micro in range(ga):
                which=(rank+micro)%2;n=min(len(features[which][1]),[1,3,17,64][(rank+micro)%4])
                label=torch.cat([torch.tensor([-100]),features[which][1][:n]]).view(1,-1)
                rankrows.append(dict(input_ids=torch.full_like(label,which),attention_mask=torch.ones_like(label),labels=label))
            batches.append(rankrows)
        N=sum(int(b['labels'][:,1:].ne(-100).sum())for rows in batches for b in rows)
        head=Head();opt=torch.optim.AdamW(head.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
        ref_loss=0.
        for rows in batches:
            for b in rows:
                out=head(**b);n=int(b['labels'][:,1:].ne(-100).sum());loss=out.loss*n/N;ref_loss+=float(loss.detach());loss.backward()
        refgrad=head.weight.grad.detach().clone()
        ref_norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.0));opt.step()
        refparam=head.weight.detach().clone();refm=opt.state[head.weight]['exp_avg'].clone();refv=opt.state[head.weight]['exp_avg_sq'].clone()
        del head,opt
        head=Head()
        trainer=CustomSeq2SeqTrainer(finetuning_args=FinetuningArguments(finetuning_type='full'),processor=None,
          args=TrainingArguments(output_dir=str(A/f'cpu-head-ga{ga}'),use_cpu=True,report_to=[],gradient_accumulation_steps=ga),
          model=head,tokenizer=tok)
        trainer.model_accepts_loss_kwargs=False;trainer.is_in_train=True;trainer.current_gradient_accumulation_steps=ga
        opt=torch.optim.AdamW(head.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
        actual_loss=0.;factors=[]
        for rank,rows in enumerate(batches):
            trainer._ab_window=dict(step=1,counts=[int(b['labels'][:,1:].ne(-100).sum())for b in rows],global_tokens=N,ga=ga,dp=4,cursor=0)
            for b in rows:
                n=int(b['labels'][:,1:].ne(-100).sum());factors.append(4*ga*n/N)
                loss=trainer.compute_loss(head,copy.deepcopy(b))/ga/4
                actual_loss+=float(loss.detach());loss.backward()
        gradients=metric(refgrad,head.weight.grad);actual_norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.0));opt.step()
        row=dict(ga=ga,dp=4,global_supervised_tokens=N,counts=[[int(b['labels'][:,1:].ne(-100).sum())for b in rows]for rows in batches],
          normalization_factors=factors,reference_loss=ref_loss,actual_loss=actual_loss,gradient=gradients,
          clip_norms=[ref_norm,actual_norm],adam_m=metric(refm,opt.state[head.weight]['exp_avg']),adam_v=metric(refv,opt.state[head.weight]['exp_avg_sq']),
          master_update=metric(refparam-weight,head.weight.detach()-weight))
        results.append(row);print('CPU NUMERIC',json.dumps(row),flush=True)
        (A/'CPU_M0_HEAD_NUMERICS.json').write_text(json.dumps(dict(scope='real M0 full-context hidden states, full-vocabulary FP32 LM-head; production LF compute_loss and normalization wrapper; manual GA/DP factors',
            not_tested=['backbone gradients','CUDA Liger fused CE','actual distributed reduce-scatter','ZeRO-3 Adam partitions'],proof=proof,results=results),indent=2)+'\n')
        del trainer,head,opt,refgrad,refparam,refm,refv

if __name__=='__main__':main()
