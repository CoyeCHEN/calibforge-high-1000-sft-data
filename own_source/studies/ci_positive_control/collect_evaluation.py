"""Check the whole official scope and retain per-notebook execution evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from opencompass.datasets.cibench import load_experiment


def main(root,label,edition):
    root=Path(root)
    protocol=json.loads((root/f'configs/evaluation-protocol-{edition}.json').read_text())
    runs=list((root/f'evaluation/{label}-{edition}').glob('20*'))
    if len(runs)!=1:raise ValueError('Explicit single evaluation identity required')
    run=runs[0];groups={}
    for entry in protocol['entries']:
        parts=Path(entry['task_id']).parts
        groups.setdefault((parts[0],parts[1]),[]).append(entry)
    questions=0;records=[];official={};artifacts=[]
    for (kind,lib),entries in groups.items():
        pred=run/f'predictions/{label}/{kind}/{lib}.json'
        score=run/f'results/{label}/{kind}/{lib}.json'
        raw=json.loads(pred.read_text());metrics=json.loads(score.read_text())
        if 'error' in metrics:raise ValueError('Official scoring failed: '+str(score))
        assert set(raw)=={str(i)for i in range(len(entries))}
        official[kind+'/'+lib]=metrics
        for path in (pred,score):
            artifacts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i,entry in enumerate(entries):
            result=raw[str(i)]
            expected=load_experiment(str(root/'data/cibench_dataset'/entry['task_id']))
            prompts=[m['content']for m in expected['questions'] if m['role']=='user']
            assert result['origin_prompt']==prompts,(entry['task_id'],'prompt identity')
            assert len(result['prediction'])==len(result['steps'])==len(prompts)==entry['questions']
            questions+=len(prompts)
            errors=Counter();repeated=[];previous=None
            for j,steps in enumerate(result['steps']):
                for action in steps:
                    if action['state']!=0:errors[str(action['state'])]+=1
                    key=json.dumps(dict(type=action['type'],args=action['args'],
                        result=action['result'],errmsg=action['errmsg']),sort_keys=True)
                    if key==previous:repeated.append(j)
                    previous=key
            records.append(dict(task_id=entry['task_id'],questions=len(prompts),
                action_errors=dict(errors),adjacent_identical_action_and_feedback_questions=repeated))
    assert len(records)==307 and questions==1818 and len(official)==26
    destination=root/f'checks/cibench-{label}-{edition}-completion.json'
    destination.write_text(json.dumps(dict(status='COMPLETE',label=label,edition=edition,
        notebooks=len(records),questions=questions,library_scopes=len(official),run=str(run),
        official_metrics=official,execution_records=records,artifacts=artifacts,
        repetition_caveat='Exact adjacent action+feedback diagnostic only; question identities differ. '
            'A repeated action is not automatically a loop; use full request histories for causal review.'),indent=2)+'\n')
    print(destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('label');p.add_argument('edition')
    a=p.parse_args();main(a.root,a.label,a.edition)
