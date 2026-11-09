"""Unique-row, globally token-weighted teacher-forced NLL via the real server."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request

from datasets import load_from_disk


def main(root,label):
    root=Path(root);data=load_from_disk(str(root/'data/cache-262144-r2'))['validation']
    meta=[json.loads(line) for line in (root/'data/validation-encoding-index-r2.jsonl').open()]
    output=root/'checks'/('validation-nll-'+label+'.jsonl')
    if output.exists():raise FileExistsError('NLL evidence exists; no implicit replay')
    numerator=0.;denominator=0;task={};start=time.time()
    with output.open('x') as handle:
        for i,(row,entry) in enumerate(zip(data,meta)):
            ids=row['input_ids'];labels=row['labels']
            payload=dict(input_ids=ids,sampling_params=dict(max_new_tokens=0,temperature=0,top_p=1,
                top_k=-1,min_p=0,repetition_penalty=1,presence_penalty=0,skip_special_tokens=False),
                return_logprob=True,logprob_start_len=0,stream=False)
            request=urllib.request.Request('http://127.0.0.1:18641/generate',data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=180) as response:
                raw=json.load(response)
            values=raw['meta_info']['input_token_logprobs'];offset=len(ids)-len(values)
            assert offset in (0,1) and all(y==-100 for y in labels[:offset])
            losses=[]
            for j,pair in enumerate(values):
                p=offset+j
                assert pair[1]==ids[p],(i,p,pair[1],ids[p])
                if labels[p]!=-100:
                    assert pair[0] is not None and math.isfinite(pair[0])
                    losses.append(-pair[0])
            assert len(losses)==entry['supervised_tokens']
            total=sum(losses);n=len(losses);numerator+=total;denominator+=n
            result=dict(label=label,index=i,trace_id=entry['trace_id'],task_group=entry['task_group'],
                supervised_tokens=n,nll_sum=total,token_nll=total/n,input_sha256=entry['input_sha256'],
                labels_sha256=entry['labels_sha256'])
            handle.write(json.dumps(result)+'\n');handle.flush()
            old=task.setdefault(entry['task_group'],[0.,0]);old[0]+=total;old[1]+=n
            if (i+1)%20==0:print(label,i+1,len(data),'NLL',numerator/denominator,flush=True)
    assert denominator==273159 and len(data)==194
    result=dict(status='COMPLETE',label=label,unique_rows=len(data),task_groups=len(task),
                nll_sum=numerator,shifted_supervised_tokens=denominator,
                global_token_nll=numerator/denominator,seconds=time.time()-start,
                task_group_token_nll={k:v[0]/v[1] for k,v in task.items()},
                duplicated_distributed_rows=0,source='Explicit frozen IDs/labels; official serving input logprobs')
    (root/'checks'/('validation-nll-'+label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('label');a=p.parse_args();main(a.root,a.label)
