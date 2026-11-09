"""Independent published-message reference. Never imports the project codec/merger."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

A=Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
X=Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')
MODEL='/data/models/Qwen/Qwen3.5-4B'

def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x)for k,x in v.items()if x is not None}
    if isinstance(v,list):return [clean(x)for x in v]
    return v
def save(n,v):(A/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    import torch
    from transformers import AutoTokenizer,AutoProcessor
    from datasets import load_from_disk
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask,get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tools=json.loads((X/'configs/sglang-template-tools.json').read_text())
    # Supervision contract is frozen before inspecting cache values.
    save('INDEPENDENT_SUPERVISION_RULE.json',dict(source='original messages/tools, no project converter imports',
      template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),thinking=False,
      inputs='all source history, native published arguments parsed as JSON object, stable sequential call IDs',
      labels='all assistant generated suffixes once including im_end; generation opener and post-EOS newline masked',
      reference='independently render each source assistant with official tokenizer, verify compatible full prefix; merge by absolute positions',
      causal_shift='logit[i-1] predicts label[i]; no pre-shift in cache',
      tools='frozen 8 declarations; terminal declarations are conditioning, native Python and final_answer targets'))
    cache=load_from_disk(str(X/'data/cache-262144-r2'))
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    collator_tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    template=get_template_and_fix_tokenizer(collator_tok,DataArguments(template='qwen3_5',enable_thinking=False))
    collator=SFTDataCollatorWith4DAttentionMask(template=template,tokenizer=collator_tok,processor=processor,
      pad_to_multiple_of=8,label_pad_token_id=-100,block_diag_attn=False,attn_implementation='flash_attention_2',compute_dtype=torch.bfloat16)
    live=json.loads((A/'ACTUAL_FORWARD_AND_SAVE.json').read_text())
    livecounts=collections.Counter((b['input_sha256'],b['labels_sha256'])for rr in live.values()for b in rr['forward_batches'])
    stats={};selected={};allkeys={};differences=[]
    for split in ['train','validation']:
        idx=[json.loads(l)for l in (X/f'data/{split}-encoding-index-r2.jsonl').open()]
        mapping={r['trace_id']:i for i,r in enumerate(idx)}
        c=collections.Counter();totalstart=time.time()
        with (A/f'{split}-independent-rows.jsonl').open('w') as out:
            for ordinal,line in enumerate((X/f'data/{split}-raw.jsonl').open()):
                raw=json.loads(line);src=raw['source'];messages=clean(src['messages']);ranges=[];ids=None
                pending=None;codeproof=[];observations=[]
                for mi,m in enumerate(messages):
                    if m.get('content') is None:m['content']=''
                    if m['role']=='assistant':
                        for tc in m.get('tool_calls',[]):
                            tc['type']='function';tc['id']=f'audit-{ordinal}-{mi}';pending=tc['id']
                            arg=tc['function']['arguments']
                            if isinstance(arg,str):arg=json.loads(arg)
                            tc['function']['arguments']=arg
                            if tc['function']['name']=='add_and_execute_jupyter_code_cell':
                                codeproof.append({'message_index':mi,'arguments':copy.deepcopy(arg)})
                        opts=dict(tools=tools,enable_thinking=False)
                        prefix=tok.apply_chat_template(messages[:mi],tokenize=False,add_generation_prompt=True,**opts)
                        rendered=tok.apply_chat_template(messages[:mi+1],tokenize=False,add_generation_prompt=False,**opts)
                        assert rendered.startswith(prefix)
                        p=tok.apply_chat_template(messages[:mi],tokenize=True,return_dict=False,add_generation_prompt=True,**opts)
                        suffix=rendered[len(prefix):];s=tok.encode(suffix,add_special_tokens=False)
                        assert tok.decode(s,skip_special_tokens=False,clean_up_tokenization_spaces=False)==suffix
                        nxt=p+s
                        if ids is not None:assert nxt[:len(ids)]==ids,('history changed',src['id'],mi)
                        ids=nxt
                        assert ids[-2:]==[tok.convert_tokens_to_ids('<|im_end|>'),198]
                        ranges.append((len(p),len(ids)-1,mi))
                    elif m['role']=='tool':
                        m['tool_call_id']=pending;pending=None;observations.append(m['content'])
                labels=[-100]*len(ids)
                for left,right,mi in ranges:
                    assert all(x==-100 for x in labels[left:right]);labels[left:right]=ids[left:right]
                n=mapping[src['id']];actual=cache[split][n]
                eqids=ids==actual['input_ids'];eqlabels=labels==actual['labels']
                key=(digest(ids),digest(labels));allkeys[src['id']]=key
                result=dict(trace_id=src['id'],row=n,input_sha256=key[0],labels_sha256=key[1],
                    input_match=eqids,labels_match=eqlabels,targets=len(ranges),supervised_tokens=sum(x!=-100 for x in labels),
                    actual_run_exposure=livecounts[key]if split=='train'else None)
                if not eqids or not eqlabels:
                    for field,ref in [('input_ids',ids),('labels',labels)]:
                        real=actual[field]
                        pos=next((i for i,(a,b)in enumerate(zip(ref,real))if a!=b),min(len(ref),len(real)))
                        if ref!=real:differences.append(dict(trace_id=src['id'],field=field,first_difference=pos,expected=ref[max(0,pos-20):pos+30],actual=real[max(0,pos-20):pos+30]))
                batch=collator([copy.deepcopy(actual)]);keep=batch['attention_mask'][0].bool()
                assert batch['input_ids'][0][keep].tolist()==actual['input_ids']
                assert batch['labels'][0][keep].tolist()==actual['labels']
                assert batch['labels'][0][~keep].eq(-100).all()
                assert all(x==-100 or x==ids[i]for i,x in enumerate(labels))
                c.update(rows=1,targets=len(ranges),tokens=result['supervised_tokens'],input_matches=eqids,label_matches=eqlabels,collator_matches=1)
                if split=='train':c['matched_actual_forward_exposures']+=livecounts[key]
                kinds=[]
                if len(codeproof)>2:kinds.append('multi_cell')
                if any('Traceback' in x for x in observations):kinds.append('error_feedback')
                if codeproof and max(len(json.dumps(p['arguments']))for p in codeproof)>2500:kinds.append('long_code')
                if any(m.get('tool_calls',[{}])[0].get('function',{}).get('name')=='final_answer'for m in messages if m.get('tool_calls')):kinds.append('final_answer')
                for kind in kinds:
                    if kind not in selected and split=='train':
                        selected[kind]=src['id'];stem='sample-'+kind
                        save(stem+'.json',dict(raw=raw,reference_messages=messages,reference_tools=tools,ranges=ranges,
                          expected_input_ids=ids,expected_labels=labels,actual_cache=actual,
                          collator={k:v.tolist()if torch.is_tensor(v)else v for k,v in batch.items()},python_arguments=codeproof,result=result))
                        with (A/(stem+'-tokens.tsv')).open('w')as f:
                            w=csv.writer(f,delimiter='\t');w.writerow(['position','token_id','text','label','predicted_from_logit'])
                            for i,(token,label)in enumerate(zip(ids,labels)):w.writerow([i,token,tok.decode([token]),label,i-1 if label!=-100 else 'masked'])
                        (A/(stem+'-rendered.txt')).write_text(tok.decode(ids,skip_special_tokens=False))
                out.write(json.dumps(result)+'\n')
                if (ordinal+1)%250==0:print(split,ordinal+1,dict(c),'seconds',round(time.time()-totalstart,1),flush=True)
        stats[split]=dict(c)
    save('INDEPENDENT_DATA_RESULT.json',dict(stats=stats,differences=differences,selected=selected,
        actual_training_forward_count=sum(livecounts.values()),unmatched_actual_forward_keys=[list(k)for k in livecounts if k not in set(allkeys.values())]))
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
