"""One finite native-entry acceptance, manually submitted; no polling/retry queue."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import re
import tempfile
import yaml
from .config import validate_native,executable_path
from .paths import runtime_config,workspace
from .launch import gpu_check,validate_resume
from .checkpoint_archive import sha256,atomic_json
from .new_entry_acceptance import new_entry_acceptance


def prepare_calibforge(name='verify-calibforge-data'):
    """CPU-only native Arrow input preparation; no GPU allocation or dispatch."""
    if not re.fullmatch('verify-[A-Za-z0-9_-]+',name):raise ValueError('Separate safe verify identity required')
    from .paths import data_manifest_path
    from .calibforge import validate_pair,validate_dataset,canary_input_contract
    from .launch import validate_data
    from ..qwen35 import digest
    root=workspace();cfg=validate_pair()['m3-a-128k']
    manifest_path=data_manifest_path('m3-a-128k');manifest=json.loads(manifest_path.read_text())
    validate_data(cfg['tokenized_path'],manifest);validate_dataset(cfg,manifest)
    longest=None;largest=None
    for split in ('train','validation'):
        with Path(manifest['row_indexes'][split]['path']).open()as stream:
            for index,line in enumerate(stream):
                row=json.loads(line)
                record=dict(source_split=split,index_in_split=index,
                    **{k:row[k]for k in ('input_sha256','labels_sha256','text_tokens','loss_tokens')})
                if longest is None or row['text_tokens']>longest['text_tokens']:longest=record
                if largest is None or row['loss_tokens']>largest['loss_tokens']:largest=record
    selected=list({(row['source_split'],row['index_in_split']):row for row in (longest,largest)}.values())
    cache=root/'data'/name;config_path=root/'configs'/f'{name}.yaml';sidecar=config_path.with_suffix('.inputs.json')
    output=Path(runtime_config()['hot_root'])/'runs'/name
    if cache.exists()or config_path.exists()or sidecar.exists()or output.exists():raise FileExistsError('Existing canary preparation preserved')
    from datasets import Dataset,DatasetDict,Value,load_from_disk
    source=load_from_disk(manifest['path']);rows=[]
    for selected_row in selected:
        row=source[selected_row['source_split']][selected_row['index_in_split']]
        if digest(row['input_ids'])!=selected_row['input_sha256']or digest(row['labels'])!=selected_row['labels_sha256']:
            raise ValueError('Actual Arrow row differs from frozen source index')
        rows.append(row)
    # 384 native exposures = DP4 * MB1 * GA32 * three steps. The diagnostic
    # model is discarded; formal A/B data order and splits are untouched.
    def repeated():
        for index in range(384):yield rows[(index//4)%len(rows)]|{'_calibforge_sample_index':index}
    features=source['train'].features.copy();features['_calibforge_sample_index']=Value('int64')
    with tempfile.TemporaryDirectory(prefix='calibforge-canary-',dir=root/'data')as temp:
        data=Dataset.from_generator(repeated,features=features,
            cache_dir=str(Path(temp)/'cache'),writer_batch_size=1,keep_in_memory=False)
        staged=Path(temp)/'tokenized';DatasetDict(train=data).save_to_disk(str(staged),max_shard_size='400MB',num_proc=1)
        os.rename(staged,cache)
    cfg.update(max_steps=3,save_steps=1,save_strategy='steps',do_eval=False,eval_strategy='no',
               eval_on_start=False,eval_dataset=None,output_dir=str(output),tokenized_path=str(cache),
               remove_unused_columns=False)
    validate_native(cfg,bounded=True)
    proof=dict(kind='calibforge_frozen_data_gpu_canary',source_dataset_manifest_sha256=sha256(manifest_path),
        selected_rows=selected,diagnostic_exposures=384,formal_training_data_modified=False,
        cache_path=str(cache),cache_artifacts=[dict(path=str(p.relative_to(cache)),bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(cache.rglob('*'))if p.is_file()])
    with config_path.open('x')as stream:yaml.safe_dump(cfg,stream,sort_keys=False)
    atomic_json(sidecar,proof);contract=canary_input_contract(sidecar)
    return dict(config=str(config_path),inputs=str(sidecar),gpu_started=False,
        maximum_real_input_tokens=contract['maximum_real_input_tokens'],
        maximum_real_supervised_tokens=contract['maximum_real_supervised_tokens'])


def run(config,ids,resume_only=False,timeout=10800):
    path=Path(config).resolve();cfg=validate_native(yaml.safe_load(path.read_text()),bounded=True)
    if cfg.get('runtime_recipe')=='qwen35_772_b32_e5':
        return run_capacity772(path,cfg,ids,resume_only,timeout)
    runtime=runtime_config();root=workspace();output=Path(cfg['output_dir']);stage='verify-current-entry-'+str(cfg['cutoff_len'])
    if not output.name.startswith('verify-'):raise ValueError('Separate verify output required')
    inputs=None
    if cfg.get('runtime_recipe')=='calibforge_appendix_e':
        from .calibforge import validate_predecessor,canary_input_contract,validate_initialization
        validate_predecessor()
        validate_initialization('m3-a-128k',cfg,verify_weights=True)
        sidecar=json.loads(path.with_suffix('.inputs.json').read_text())
        if Path(sidecar['cache_path']).resolve()!=Path(cfg['tokenized_path']).resolve():raise ValueError('Canary native cache path mismatch')
        cache=Path(sidecar['cache_path'])
        if {str(p.relative_to(cache))for p in cache.rglob('*')if p.is_file()}!={i['path']for i in sidecar['cache_artifacts']}:
            raise ValueError('Canary cache file inventory changed')
        for item in sidecar['cache_artifacts']:
            target=cache/item['path']
            if target.stat().st_size!=item['bytes']or sha256(target)!=item['sha256']:raise ValueError('Canary cache corrupted')
        inputs=canary_input_contract(path.with_suffix('.inputs.json'))
    with ExitStack()as stack:
        for lock_path in sorted({root/'locks/training.lock',Path(runtime.get('project_training_lock',root/'locks/training.lock'))}):
            handle=stack.enter_context(lock_path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not resume_only and output.exists()and any(output.iterdir()):raise FileExistsError('Canary output preserved')
        for resume in ([False,True]if not resume_only else[True]):
            checkpoint=output/'checkpoint-2'
            if resume:validate_resume(checkpoint,output,stage,sha256(path))
            devices=gpu_check(ids,runtime)  # Immediately before EACH fresh CUDA lifecycle.
            output.mkdir(parents=True,exist_ok=True)
            if inputs is not None:
                proof_path=output/'GPU_CANARY_INPUTS.json'
                if proof_path.exists():
                    if json.loads(proof_path.read_text())!=sidecar:raise ValueError('Resume canary input identity changed')
                else:atomic_json(proof_path,sidecar)
            job=stage+('-resume-'if resume else'-initial-')+str(time.time_ns())
            env=os.environ|dict(QWEN35_JOB_ID=job,QWEN35_STAGE=stage,QWEN35_CONFIG_SHA256=sha256(path),
                QWEN35_MODEL_PATH=runtime['model_path'],CUDA_VISIBLE_DEVICES=','.join(devices[i]['uuid']for i in ids),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
            cmd=[runtime['python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
                '--max-restarts=0','--module','sft.runtime.lf_train',str(path),'--archive-checkpoints','--bounded-acceptance']
            cmd+=['--resume',str(checkpoint)]if resume else['--stop-after-checkpoint-2']
            with (output/(job+'.log')).open('x')as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=child.wait(timeout=timeout)
                    if code:raise RuntimeError('Native torchrun failed: '+str(code))
                finally:
                    # Only this invocation's process group; never touch unrelated CUDA processes.
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result=new_entry_acceptance(output,cfg['cutoff_len'],configuration=cfg,input_contract=inputs);atomic_json(output/'new-entry-acceptance.json',result)
        if not result['passed']:raise ValueError('Current-entry acceptance failed: '+str(result))
        return result


def prepare_capacity772(corpus,context,dp):
    """Select real longest-input/largest-supervision rows for a discarded test."""
    from datasets import load_from_disk,Dataset,DatasetDict,Value
    from ..qwen35 import digest
    if context not in(131072,262144)or dp not in(4,8):raise ValueError('Invalid capacity candidate')
    root=workspace();runtime=runtime_config();corpus=Path(corpus)
    name=f'verify-capacity772-{context}-dp{dp}'
    cache=root/'data'/name;config=root/'configs'/(name+'.yaml')
    if cache.exists()or config.exists():raise FileExistsError('Existing capacity inputs preserved')
    selected=[];rows=[];features=None
    for pool in('calibforge','tmax'):
        data=load_from_disk(str(corpus/pool/f'cache-{context}'))
        candidates=[]
        for split in('train','validation'):
            for index,row in enumerate(data[split]):
                candidates.append((len(row['input_ids']),sum(x!=-100 for x in row['labels']),split,index))
        required={max(candidates,key=lambda x:x[0]),max(candidates,key=lambda x:x[1])}
        for length,loss_tokens,split,index in sorted(required):
            row=data[split][index]
            selected.append(dict(pool=pool,source_split=split,index_in_split=index,
                input_sha256=digest(row['input_ids']),labels_sha256=digest(row['labels']),
                text_tokens=length,loss_tokens=loss_tokens,
                source_data_manifest_sha256=sha256(corpus/pool/'DATA_MANIFEST.json')))
            rows.append(row)
        features=data['train'].features.copy()
    features['_calibforge_sample_index']=Value('int64')
    def diagnostic_rows():
        for index in range(96):
            yield rows[(index//dp)%len(rows)]|{'_calibforge_sample_index':index}
    with tempfile.TemporaryDirectory(prefix=name+'-',dir=root/'data')as temp:
        ds=Dataset.from_generator(diagnostic_rows,features=features,
            cache_dir=str(Path(temp)/'generator'),writer_batch_size=1)
        target=Path(temp)/'cache';DatasetDict(train=ds).save_to_disk(str(target),max_shard_size='400MB',num_proc=1)
        os.rename(target,cache)
    cfg=yaml.safe_load(Path(runtime['capacity_config_template']).read_text())
    cfg.update(tokenized_path=str(cache),output_dir=str(Path(runtime['hot_root'])/'runs'/name),
        cutoff_len=context,gradient_accumulation_steps=32//dp)
    validate_native(cfg,bounded=True)
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    sidecar=dict(kind='REAL_772_DISCARDED_CAPACITY_INPUTS',dp=dp,context=context,
        selected_rows=selected,diagnostic_exposures=96,formal_training=False,
        diagnostic_weights_must_not_initialize_formal_training=True,
        cache_artifacts=[dict(path=str(p.relative_to(cache)),bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(cache.rglob('*'))if p.is_file()])
    atomic_json(config.with_suffix('.inputs.json'),sidecar)
    return dict(config=str(config),inputs=str(config.with_suffix('.inputs.json')),
        dp=dp,context=context,selected_rows=selected)


def run_capacity772(path,cfg,ids,resume_only=False,timeout=10800):
    """One finite native LF capacity test; no new queue, retries or train loop."""
    from .new_entry_acceptance import capacity772
    runtime=runtime_config()|dict(max_concurrent_gpus=len(ids));root=workspace();output=Path(cfg['output_dir'])
    inputs=json.loads(path.with_suffix('.inputs.json').read_text())
    dp=len(ids)
    if inputs['dp']!=dp:raise ValueError('Actual capacity world size differs from frozen inputs')
    if dp not in(4,8)or cfg['gradient_accumulation_steps']!=32//dp:
        raise ValueError('Capacity config does not implement global batch32')
    if not output.name.startswith('verify-'):raise ValueError('Discarded diagnostic output required')
    if inputs['diagnostic_exposures']!=96:raise ValueError('Three global batches required')
    for item in inputs['cache_artifacts']:
        candidate=Path(cfg['tokenized_path'])/item['path']
        if candidate.stat().st_size!=item['bytes']or sha256(candidate)!=item['sha256']:
            raise ValueError('Immutable diagnostic Arrow cache changed')
    with ExitStack()as stack:
        for name in sorted({root/'locks/training.lock',Path(runtime['project_training_lock'])}):
            handle=stack.enter_context(name.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not resume_only and output.exists()and any(output.iterdir()):
            raise FileExistsError('Existing capacity output preserved; no replay')
        for resume in([False,True]if not resume_only else[True]):
            checkpoint=output/'checkpoint-2'
            stage='verify-current-772-'+output.name
            if resume:validate_resume(checkpoint,output,stage,sha256(path))
            devices=gpu_check(ids,runtime)
            output.mkdir(parents=True,exist_ok=True)
            env=os.environ|dict(QWEN35_JOB_ID=output.name+('-resume'if resume else'-initial'),
                QWEN35_STAGE=stage,QWEN35_CONFIG_SHA256=sha256(path),
                QWEN35_MODEL_PATH=runtime['model_path'],CUDA_VISIBLE_DEVICES=','.join(devices[i]['uuid']for i in ids),
                PATH=executable_path(runtime),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
            cmd=[runtime['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(dp),
                '--max-restarts=0','--module','sft.runtime.lf_train',str(path),'--archive-checkpoints','--bounded-acceptance']
            cmd+=['--resume',str(checkpoint)]if resume else['--stop-after-checkpoint-2']
            with(output/('resume.log'if resume else'initial.log')).open('x')as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=child.wait(timeout=timeout)
                    if code:raise RuntimeError('Native capacity torchrun failed: '+str(code))
                finally:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result=capacity772(output,cfg,inputs,dp)
        atomic_json(output/'CAPACITY_ACCEPTANCE.json',result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--gpus',required=True);p.add_argument('--resume-only',action='store_true');p.add_argument('--timeout',type=int,default=10800)
    a=p.parse_args()
    if not 60<=a.timeout<=10800:raise ValueError('Finite timeout must be 60..10800 seconds')
    print(json.dumps(run(a.config,a.gpus.split(','),a.resume_only,a.timeout),indent=2))
