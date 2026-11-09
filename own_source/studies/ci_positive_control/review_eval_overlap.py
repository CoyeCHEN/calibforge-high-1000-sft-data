"""Incremental lexical overlap review of selected QA against both held-out sets."""
import json
from pathlib import Path
import re
import sys
from rapidfuzz import process,fuzz
from select_data import normalized,ngrams


def main(root):
    root=Path(root);queries=[]
    for file in (root/'data/cibench_dataset').rglob('*.ipynb'):
        if '/datasources/' in str(file):continue
        for i,cell in enumerate(json.loads(file.read_text()).get('cells',[])):
            if cell['cell_type']!='markdown':continue
            question=''.join(cell['source']).split('\n\nThought: ')[0]
            question=re.sub(r'^File Path:.*?\n\n','',question,flags=re.S).strip()
            queries.append(dict(source=str(file.relative_to(root/'data/cibench_dataset')),cell=i,question=question))
    dab=[json.loads(x)for x in (root/'data/dabstep/all.jsonl').open()]
    dab_normal={normalized(x['question'])for x in dab};dab_grams=[ngrams(x['question'])for x in dab]
    exact=near=0;matches=[];choices=[normalized(q['question'])for q in queries]
    out=root/'checks/cibench-token-sort-candidates.jsonl'
    with out.open('x') as handle:
        for split in ['train','validation']:
            for n,line in enumerate((root/f'data/{split}-raw.jsonl').open()):
                source=json.loads(line)['source'];question=source['question'];gram=ngrams(question)
                exact+=int(normalized(question) in dab_normal)
                near+=int(any(gram and other and len(gram&other)/len(gram|other)>=.8 for other in dab_grams))
                match=process.extractOne(normalized(question),choices,
                    scorer=fuzz.token_sort_ratio,score_cutoff=50)
                if match:
                    _,score,j=match
                    record=dict(id=source['id'],split=split,question=question,
                        kaggle=source['kaggle_dataset_name'],score=score,match=queries[j])
                    handle.write(json.dumps(record)+'\n');handle.flush();matches.append(record)
                if (n+1)%200==0:print(split,n+1,'candidates',len(matches),flush=True)
    matches.sort(key=lambda v:-v['score'])
    (root/'checks/cibench-token-sort-review.json').write_text(json.dumps(dict(
        candidates=matches[:80],dabstep_all450_exact=exact,dabstep_all450_near_4gram=near,
        scope='All selected 2194 rows; lexical screening is not proof of universal semantic non-overlap'),indent=2)+'\n')
    print('OVERLAP_REVIEW_COMPLETE',len(matches),exact,near,flush=True)


if __name__=='__main__':main(sys.argv[1])
