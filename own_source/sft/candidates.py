"""Materialize frozen, audited provider-payload candidates. Never approve a dataset."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from . import qwen35 as codec
from .runtime.checkpoint_archive import sha256,atomic_json
from adapters.openharness_v2 import _resolve


def materialize(exports,index,proof,output,classification,inventory):
    from datasets import Dataset,DatasetDict,Features,Sequence,Value
    if classification not in ('TB21_SAME_TASK_ADAPTATION','CLEAN_CANDIDATE'):
        raise ValueError('Explicit experimental use classification required')
    exports,index,proof,output=map(Path,(exports,index,proof,output))
    if output.exists():raise FileExistsError('Candidates preserved; choose a new output')
    audit=json.loads(proof.read_text())
    if audit.get('issues')or audit.get('result_session_manifest_matches')!=audit.get('expected_rows'):
        raise ValueError('Incomplete original/native identity verification')
    inventory=Path(inventory);export_files=json.loads(inventory.read_text())['rows']
    originals={r['trial_id']:r for r in audit['rows']}
    expected={(r['identity'],r['call_id']):r for r in map(json.loads,index.open())}
    trials=[json.loads(line)for line in (exports/'final/trials.jsonl').open()]
    candidates=[t for t in trials if t['sft_candidate']]
    groups=sorted({t['task_id']for t in candidates},key=lambda s:hashlib.sha256(('42:'+s).encode()).hexdigest())
    validation=set(groups[:max(1,round(len(groups)*.05))]);counts=collections.Counter();tool_dependence=collections.Counter();seen=set()
    tok=codec.load_tokenizer();output.mkdir(parents=True)
    manifest=dict(classification=classification,formal_training_eligible=False,tool_contract='UNDECIDED',
        model_revision=codec.REVISION,template_sha256=codec.TEMPLATE_SHA,cutoff=131072,text_budget=131006,
        lf_visual_dummy_overhead=66,no_truncation=True,supervision_scope='current',supervision='Full Thinking + Action',
        export_inventory_sha256=sha256(inventory),source_index_sha256=sha256(index),original_identity_proof_sha256=sha256(proof),source_trials_sha256=sha256(exports/'final/trials.jsonl'),
        source_root=str(exports),raw_original_root=audit['original_root'],raw_original_host=audit['original_host'],
        validation_task_groups=sorted(validation),task_groups_disjoint=True,
        blockers=['TOOL_CONTRACT_UNDECIDED','M1_FINAL_HF_MISSING','CURRENT_ENTRY_128K_GA4_NOT_VERIFIED'],
        eligibility_note='Candidate only. Task overlap is an experimental-use label, not a structural rejection. CLEAN requires independent source-gate approval.')
    def generator(split):
        def rows():
            with (output/f'{split}-rows.jsonl').open('x')as metadata:
                for t in candidates:
                    if ('validation'if t['task_id']in validation else'train')!=split:continue
                    identity=originals[t['trial_id']]
                    if identity['session_id']!=t['session_id']or identity['result_sha256']!=t['result_sha256']:raise ValueError('Original session changed')
                    if not((t['reward'].get('reward')if isinstance(t['reward'],dict)else t['reward'])==1 and t['valid_score']and t['normal_close']and t['trajectory_complete']):raise ValueError('Candidate status changed')
                    root=exports/'final/exports'/t['trial_name']
                    calls_sha=sha256(root/'calls.jsonl');manifest_sha=sha256(root/'manifest.json')
                    pinned=export_files[t['trial_id']]
                    if pinned['trial_name']!=t['trial_name']or any(sha256(root/name)!=digest for name,digest in pinned['files'].items()):raise ValueError('Normalized export changed')
                    exported=json.loads((root/'manifest.json').read_text())
                    if not exported['integrity']['complete']or exported['integrity']['issues']or t['session_id']not in {s['session_id']for s in exported['sessions']}:raise ValueError('Export identity/integrity mismatch')
                    for line in (root/'calls.jsonl').open():
                        raw=json.loads(line);key=(t['trial_id'],raw['call_id'])
                        if key not in expected:continue  # Excludes recorded compaction auxiliary calls.
                        prior=expected[key];counts['candidate_decisions']+=1
                        if prior['error']:
                            counts['structure_rejected']+=1;continue
                        if prior['text_tokens']>131006:
                            counts['overlength_rejected']+=1;continue
                        payload=_resolve(raw['payload'],root);response=_resolve(raw['response'],root)
                        row,meta=codec.encode(tok,payload['messages']+[response],payload.get('tools'),scope='current',cutoff=131006)
                        if any(meta[k]!=prior[k]for k in ('input_sha256','labels_sha256','text_tokens','loss_tokens')):
                            raise ValueError('Frozen token/label identity changed')
                        if meta['input_sha256']in seen:raise ValueError('Duplicate encoded candidate')
                        seen.add(meta['input_sha256'])
                        record=dict(task_id=t['task_id'],task_group=t['task_id'],trial_id=t['trial_id'],call_id=raw['call_id'],session_id=t['session_id'],
                            split=split,classification=classification,reward=1,valid_reward=True,normal_close=True,recorder_complete=True,
                            original_result_sha256=identity['result_sha256'],original_manifest_sha256=identity['manifest_sha256'],export_manifest_sha256=manifest_sha,export_calls_sha256=calls_sha,
                            source_export=str(root),source_tools=[x['function']['name']for x in payload.get('tools',[])],**meta)
                        for c in response.get('content',[]):
                            if c.get('type')=='toolCall':tool_dependence[c['name']]+=1
                        metadata.write(json.dumps(record)+'\n');counts[split+'_rows']+=1;counts[split+'_loss_tokens']+=meta['loss_tokens']
                        yield row
        return rows
    features=Features({k:Sequence(Value('int32'))for k in ('input_ids','attention_mask','labels')}|{k:Sequence(Value('string'))for k in ('images','videos','audios')})
    data=DatasetDict({s:Dataset.from_generator(generator(s),features=features,cache_dir=str(output/'build-cache'),writer_batch_size=1)for s in ('train','validation')})
    if counts['candidate_decisions']!=len(expected):raise ValueError('Frozen candidate index did not fully join original calls')
    data.save_to_disk(str(output/'tokenized'),num_proc=1,max_shard_size='400MB')
    manifest.update(counts=dict(counts),target_tool_dependence=dict(tool_dependence),
        artifacts=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha256(p))for p in sorted((output/'tokenized').rglob('*'))if p.is_file()])
    atomic_json(output/'candidate-manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exports',required=True);p.add_argument('--index',required=True);p.add_argument('--identity-proof',required=True);p.add_argument('--output',required=True);p.add_argument('--classification',required=True);p.add_argument('--export-inventory',required=True)
    a=p.parse_args();print(json.dumps(materialize(a.exports,a.index,a.identity_proof,a.output,a.classification,a.export_inventory),indent=2))
