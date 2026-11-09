"""Audited CI expansion; original messages/results and the holdout stay immutable."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from inventory import DATASET, REVISION, digest
from select_data import assign_source_groups, check_trace, evaluation_overlap_checker, normalized

TOKENS = re.compile(r'(?P<number>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?)|(?P<word>[^\W\d]\w*)|(?P<unit>[$€£])')
GRAMMAR = {'an','the','is','are','was','were','has','have','with','for','of','and','in','to','at','by','from','as','be','it','that','which'}
NONANSWER = re.compile(r'\b(?:not applicable|not explicitly|not available|not provided|cannot be|could not be|unable to|unknown|insufficient data|not enough data)\b', re.I)
EXECUTION_ERROR = re.compile(r'Traceback \(most recent call last\)|(?:^|\n)(?:[A-Za-z]+Error|Error):|Execution failed', re.M)
NEGATION = re.compile(r'\b(?:not|never|unable|cannot|unavailable|unsure)\b', re.I)


def tokens(text):
    result=[]
    for match in TOKENS.finditer(unicodedata.normalize('NFKC',text)):
        if match.lastgroup=='number':
            value=match.group();number=Decimal(value.rstrip('%').replace(',',''))
            if value.endswith('%'):number/=100
            result.append(('number',str(number.normalize())))
        else:result.append((match.lastgroup,match.group().casefold()))
    return result


def contains(sequence,part):
    return bool(part) and any(sequence[i:i+len(part)]==part for i in range(len(sequence)-len(part)+1))


def ordered(sequence,part):
    iterator=iter(sequence)
    return bool(part) and all(any(value==wanted for value in iterator)for wanted in part)


def approval(reference,answer,observations):
    if NONANSWER.search(reference)or NONANSWER.search(answer):return 'QUARANTINE_NONANSWER'
    if normalized(reference)in {'none','null','nan','n a','no answer','not found'}:return 'QUARANTINE_NONANSWER'
    if not observations or EXECUTION_ERROR.search(observations[-1]):return 'QUARANTINE_LAST_TOOL_ERROR'
    ref,final=tokens(reference),tokens(answer)
    if not ref or not final:return 'QUARANTINE_EMPTY_ANSWER'
    if normalized(reference)in {'yes','no'}:
        return 'APPROVED_EXACT_BINARY'if final[0]==ref[0]else 'QUARANTINE_BINARY_CONTRADICTION'
    if NEGATION.search(answer)and not NEGATION.search(reference):return 'QUARANTINE_UNREVIEWED_NEGATION'
    # A short category mentioned inside a comparison is not proof of selecting
    # that category (e.g. "K-means ... compared to Hierarchical").
    if len(ref)<=2 and not any(k=='number'for k,v in ref) and re.search(r'\b(?:than|versus|vs|compared|instead)\b',answer,re.I):
        return 'QUARANTINE_CATEGORY_COMPARISON'
    if contains(final,ref):return 'APPROVED_EXACT_ANSWER_TOKENS'
    salient=lambda values:[(k,v)for k,v in values if k!='word'or v not in GRAMMAR]
    content=salient(ref)
    if not ordered(salient(final),content):return 'QUARANTINE_REFERENCE_DISAGREEMENT'
    if not any(ordered(salient(tokens(obs)),content)for obs in observations if not EXECUTION_ERROR.search(obs)):
        return 'QUARANTINE_MISSING_RESULT_SUPPORT'
    return 'APPROVED_REFERENCE_AND_REAL_OBSERVATION'


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main(root,output,target):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'data').mkdir();(output/'checks').mkdir();(output/'configs').mkdir();(output/'logs').mkdir()
    rows=[json.loads(l)for p in sorted((root/'data/inventory-shards').glob('*.jsonl'))for l in p.open()]
    assign_source_groups(rows);index={x['id']:x for x in rows}
    frozen_train=[json.loads(l)for l in (root/'data/train-raw.jsonl').open()]
    validation=[json.loads(l)for l in (root/'data/validation-raw.jsonl').open()]
    base_ids={x['source']['id']for x in frozen_train}
    seen_q={normalized(x['source']['question'])for x in validation}
    seen_trace={x['source_trace_sha256']for x in validation}
    overlap,screened=evaluation_overlap_checker(root)
    counts=Counter();selected=[];base_rejections=[]
    with (output/'checks/eligibility-ledger.jsonl').open('x')as ledger:
        for shard in sorted((root/'data/raw-shards').glob('*.jsonl')):
            for line in shard.open():
                source=json.loads(line);meta=index[source['id']]
                if meta['split']!='train' or source['executor_type']!='e2b':continue
                trace=digest(dict(messages=source['messages'],tools=source['tools']))
                decision=overlap(source['question'])
                if not normalized(source['question']):decision='QUARANTINE_EMPTY_QUESTION'
                quality=check_trace(source)
                if not decision and isinstance(quality,str):decision=quality
                observations=[m['content']for m in source['messages']if m['role']=='tool']
                if not decision:decision=approval(source['answer'],quality['final_answer'],observations)
                question=normalized(source['question'])
                if decision.startswith('APPROVED')and(question in seen_q or trace in seen_trace):decision='QUARANTINE_DUPLICATE'
                if decision.startswith('APPROVED')and len(selected)>=target:decision='QUALIFIED_OVER_QUOTA'
                record=dict(id=source['id'],task_group=meta['task_group'],decision=decision,
                    source_trace_sha256=trace,source_shard=str(shard),original_2000=source['id']in base_ids,
                    reference=source['answer'],final_answer=quality['final_answer']if isinstance(quality,dict)else None,
                    last_observation_sha256=digest(observations[-1])if observations else None)
                ledger.write(json.dumps(record,ensure_ascii=False)+'\n');counts[decision]+=1
                if decision.startswith('APPROVED'):
                    seen_q.add(question);seen_trace.add(trace)
                    selected.append(dict(source=source,provenance=meta,quality=quality,
                        source_trace_sha256=trace,approval=record))
                elif source['id']in base_ids:base_rejections.append(record)
    assert not {x['provenance']['task_group']for x in selected}&{x['provenance']['task_group']for x in validation}
    with (output/'data/train-raw.jsonl').open('x')as handle:
        for row in selected:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output/'data/validation-raw.jsonl').write_bytes((root/'data/validation-raw.jsonl').read_bytes())
    (output/'checks/original-2000-rejections.json').write_text(json.dumps(base_rejections,ensure_ascii=False,indent=2)+'\n')
    result=dict(dataset=DATASET,revision=REVISION,target_training_trajectories=target,
        actual_training_trajectories=len(selected),target_met=len(selected)==target,
        validation_trajectories=len(validation),validation_identical=True,source_group_disjoint=True,
        decisions=dict(counts),original_2000_rejected=len(base_rejections),eval_questions_screened=screened,
        assistant_decisions=sum(x['quality']['assistant_decisions']for x in selected),
        python_calls=sum(x['quality']['python_calls']for x in selected),
        notebooks=len({x['provenance']['notebook_id']for x in selected}),
        source_groups=len({x['provenance']['task_group']for x in selected}),
        train_raw_sha256=sha(output/'data/train-raw.jsonl'),validation_raw_sha256=sha(output/'data/validation-raw.jsonl'),
        training_started=False,encoded=False,
        evidence_scope='Published E2B execution + exact reference/boundary checks, or ordered reference content supported by a real observation. No new results or answers synthesized. This is a transparent silver-data filter, not independent full-corpus execution verification.',
        changed_factors_vs_2000='Both corpus size and stricter quality filtering; not a pure equal-quality data-size ablation.')
    (output/'checks/selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');p.add_argument('--target',type=int,default=10000)
    a=p.parse_args();main(a.root,a.output,a.target)
