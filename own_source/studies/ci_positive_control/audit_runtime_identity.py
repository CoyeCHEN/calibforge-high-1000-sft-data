"""Historical CI8197 wrapper reconstruction, not the native-token candidate.

Run against the frozen 7491e0e source/package payload only. This script and
normalization.py remain historical audit controls; train_ci no longer installs
those wrappers. Do not use this reconstruction to identify a new candidate run.
"""
import difflib
import functools
import hashlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import sys

A = Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
X = Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')
P = Path('/ssd/chenkeyi/qwen35-sft-772-20261101/assets/source17-r1')
S = P / 'package-payload'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(n, v):
    (A / n).write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')

def binding(f):
    chain=[]
    while f is not None:
        item={'module':getattr(f,'__module__',None),'name':getattr(f,'__qualname__',repr(f))}
        if isinstance(f,functools.partial):
            item['partial_keywords']=f.keywords; chain.append(item); f=f.func; continue
        code=getattr(f,'__code__',None)
        if code:
            item.update(file=code.co_filename,line=code.co_firstlineno,sha256=sha(code.co_filename))
        next_function=getattr(f,'__wrapped__',None)
        if next_function is None and getattr(f,'__closure__',None):
            captures={k:c.cell_contents for k,c in zip(f.__code__.co_freevars,f.__closure__)}
            functions={k:v for k,v in captures.items() if inspect.isfunction(v)}
            item['closure_functions']={k:{'file':v.__code__.co_filename,'line':v.__code__.co_firstlineno,'name':v.__qualname__}for k,v in functions.items()}
            # Official DeepSpeed NVTX decorator omits functools.wraps. Its captured `func` is the actual callee.
            if 'func' in functions:
                next_function=functions['func'];item['next_edge']='closure:func'
        chain.append(item)
        f=next_function
    return chain

def main():
    A.mkdir(exist_ok=True)
    provenance={'kind':'POST_RUN_RECONSTRUCTION_NOT_LIVE_PROCESS', 'python':sys.executable,
        'python_realpath':str(Path(sys.executable).resolve()),'python_sha256':sha(sys.executable),
        'sys_path':sys.path,'env':{k:os.environ.get(k) for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']},
        'original_venv_exists':Path('/ssd/chenkeyi/qwen35-sft-772-20261101/env/train-r1/bin/python').exists(),
        'pth':{str(p):p.read_text() for p in S.glob('*.pth')},'customize':{},'packages':{}}
    for name in ['sitecustomize','usercustomize']:
        spec=importlib.util.find_spec(name);provenance['customize'][name]=None if spec is None else spec.origin
    for name in ['torch','transformers','accelerate','deepspeed','llamafactory','liger-kernel','datasets','flash-attn','trl']:
        dist=md.distribution(name)
        provenance['packages'][name]={'version':dist.version,'path':str(dist.locate_file('')),
            'direct_url':dist.read_text('direct_url.json'),'metadata_sha256':sha(Path(dist._path)/'METADATA')}
    transfer=json.loads((P/'TRANSFER_MANIFEST.json').read_text())
    sealed={v['path']:v for v in transfer['files']}
    allfiles={}
    for name in ['llamafactory','deepspeed','transformers','accelerate','liger_kernel']:
        rows=[]
        for p in sorted((S/name).rglob('*.py')):
            rel=str(p.relative_to(P));h=sha(p);orig=sealed.get(rel)
            rows.append({'file':str(p.relative_to(S)),'sha256':h,'sealed_sha256':orig and orig['sha256'],
                'matches_20261101_transfer':bool(orig and orig['sha256']==h)})
        allfiles[name]=rows
    save('PACKAGE_FILE_SHA.json',allfiles)
    provenance['transfer_manifest_sha256']=sha(P/'TRANSFER_MANIFEST.json')
    provenance['sealed_match_counts']={k:dict(total=len(v),matches=sum(r['matches_20261101_transfer']for r in v))for k,v in allfiles.items()}
    diffs={}
    for name,up in [('llamafactory','lf'),('deepspeed','ds')]:
        installed={str(p.relative_to(S/name)):p for p in (S/name).rglob('*.py')}
        upstream={str(p.relative_to(A/up)):p for p in (A/up).rglob('*.py')}
        rows=[];patch=[]
        for rel in sorted(installed.keys()|upstream.keys()):
            left=upstream.get(rel);right=installed.get(rel)
            a=left.read_text()if left else '';b=right.read_text()if right else ''
            status='same'if a==b else 'modified'if left and right else 'installed_only'if right else 'upstream_only'
            rows.append(dict(file=rel,status=status,upstream_sha256=sha(left)if left else None,actual_payload_sha256=sha(right)if right else None))
            if status!='same':patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='upstream/'+rel,tofile='sealed-payload/'+rel))
        (A/(name+'-upstream.diff')).write_text(''.join(patch))
        save(name+'-file-comparison.json',rows)
        diffs[name]={k:sum(r['status']==k for r in rows)for k in ['same','modified','installed_only','upstream_only']}
    provenance['diff_counts']=diffs
    # Import order mirrors historical frozen train_ci: normalization(torch), LF trainer/tuner, project callback, observer.
    import torch
    torch.set_num_threads(1)
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from llamafactory.train.tuner import run_exp
    from sft.runtime.lf_train import EvidenceCallback, optimizer_evidence
    from sft.runtime.checkpoint_observer import CheckpointObserver
    from transformers import Trainer
    from transformers.trainer_callback import CallbackHandler
    from accelerate import Accelerator
    from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
    from normalization import install
    targets={
      'LF.compute_loss':(CustomSeq2SeqTrainer,'compute_loss'),
      'LF.get_batch_samples':(CustomSeq2SeqTrainer,'get_batch_samples'),
      'Trainer.training_step':(Trainer,'training_step'),'Trainer.train':(Trainer,'train'),
      'Trainer._save_checkpoint':(Trainer,'_save_checkpoint'),'Trainer.save_model':(Trainer,'save_model'),
      'Trainer._save_optimizer_and_scheduler':(Trainer,'_save_optimizer_and_scheduler'),
      'CallbackHandler.on_save':(CallbackHandler,'on_save'),
      'Accelerator.backward':(Accelerator,'backward'),
      'Accelerate.DS.backward':(DeepSpeedEngineWrapper,'backward'),
      'DS.backward':(DeepSpeedEngine,'backward'),'DS.step':(DeepSpeedEngine,'step'),
      'DS._backward_prologue':(DeepSpeedEngine,'_backward_prologue'),
      'DS.save_checkpoint':(DeepSpeedEngine,'save_checkpoint'),
      'DS._save_zero_checkpoint':(DeepSpeedEngine,'_save_zero_checkpoint'),
      'DS.consolidate':(DeepSpeedEngine,'_zero3_consolidated_16bit_state_dict'),
      'DS.torch_save':(TorchCheckpointEngine,'save'),
      'Zero3.backward':(DeepSpeedZeroOptimizer_Stage3,'backward'),
      'Zero3.step':(DeepSpeedZeroOptimizer_Stage3,'step'),
      'Zero3.reduce':(DeepSpeedZeroOptimizer_Stage3,'reduce_independent_p_g_buckets_and_remove_grads')}
    before={k:binding(getattr(o,n))for k,(o,n)in targets.items()}
    (A/'call-chain-source').mkdir(exist_ok=True)
    for k,(o,n)in targets.items():
        (A/'call-chain-source'/(k+'.py')).write_text(inspect.getsource(getattr(o,n)))
    install(CustomSeq2SeqTrainer,'token')
    cb=EvidenceCallback(A/'reconstructed-callback',loss_only=True)
    cb.checkpoint_observer=CheckpointObserver(cb,optimizer_evidence,archive_enabled=False).install()
    after={k:binding(getattr(o,n))for k,(o,n)in targets.items()}
    save('RUNTIME_BINDINGS_WITH_CLOSURES.json',{'before':before,'after_normalization_and_checkpoint_observer':after,
         'model_forward_partial':'Only applied on_train_begin: partial(official Liger forward, skip_logits=True); not invoked by this identity-only reconstruction'})
    provenance['imports']={n:m.__file__ for n,m in sys.modules.items()if n in ['torch','llamafactory','deepspeed','transformers','accelerate','liger_kernel','normalization','sft.runtime.lf_train']}
    save('RUNTIME_IDENTITY.json',provenance)
    # Actual-run windows, ranks, tail and actual forward hash coverage.
    run=X/'runs/ci-e2b-8197-e2-expansion-audited-r2'
    windows={};records={};evidence={}
    for rank in range(4):
        rows=[json.loads(l)for l in (run/f'normalization-rank{rank}.jsonl').open()]
        windows[rank]=[r for r in rows if r['event']=='window']
        records[rank]=[r for r in rows if r['event']=='microbatch']
        es=[json.loads(l)for l in (run/f'preflight-evidence-rank{rank}.jsonl').open()]
        evidence[rank]=dict(begin=next(r for r in es if r['event']=='train_begin'),
          last_optimizer=next(r for r in reversed(es)if r['event']=='optimizer_step'),
          saves=[r for r in es if r['event']=='checkpoint_save_complete'],
          forward_batches=[b for r in es if r['event']=='optimizer_step' for b in r['model_input_batches']])
    summary={'per_rank':{},'normal_step':{},'tails':{},'sha_files':{}}
    for rank in range(4):
        summary['per_rank'][rank]=dict(windows=len(windows[rank]),microbatches=len(records[rank]),
           supervised_exposure=sum(r['supervised_tokens']for r in records[rank]),
           actual_forward_batches=len(evidence[rank]['forward_batches']))
        summary['normal_step'][rank]=windows[rank][0]
        summary['tails'][rank]=[r for r in windows[rank]if r['ga']!=8]
    summary['global_count_agrees_every_step']=all(len({windows[r][i]['global_tokens']for r in range(4)})==1 for i in range(514))
    summary['logged_scaling_agrees']=all(abs(r['multiplier']-r['dp']*r['ga']*r['supervised_tokens']/r['global_tokens'])<1e-14 for rr in records.values()for r in rr)
    for p in [X/'runtime/formal-expanded.sh',X/'configs/train-r2.json',X/'configs/ds_z3.json']:
        summary['sha_files'][str(p)]=sha(p)
    save('ACTUAL_RUN_WINDOWS.json',summary);save('ACTUAL_FORWARD_AND_SAVE.json',evidence)
    print(json.dumps({'diff':diffs,'sealed':provenance['sealed_match_counts'],'windows':summary['per_rank']}),flush=True)

if __name__=='__main__':main()
