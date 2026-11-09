"""Resource/data gates followed by one native torchrun, without a training loop."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from contextlib import ExitStack

import yaml

from .checkpoint_archive import atomic_json,sha256,validate_checkpoint,storage_settings
from .config import validate_native
from .paths import config_path,runtime_config,workspace,data_manifest_path,CALIBFORGE_STAGES


def gpu_check(ids,config):
    if socket.gethostname()!=config['host']:raise ValueError('Host differs from the authorized workspace')
    expected=config.get('max_concurrent_gpus',4)
    if expected not in(4,8) or len(ids)!=expected or len(set(ids))!=expected:
        raise ValueError('Exactly the accepted four or eight distinct authorized GPUs required')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
    cards={}
    for line in text.splitlines():
        index,uuid,used,total=map(str.strip,line.split(','));cards[index]=dict(uuid=uuid,used_mib=int(used),total_mib=int(total))
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=20)
    occupied={line.split(',')[0].strip()for line in text.splitlines()if line.strip()}
    for index in ids:
        card=cards[index]
        if card['uuid']not in config['authorized_gpu_uuids']:raise ValueError('GPU UUID not authorized; index alone is insufficient')
        if card['uuid']in occupied or card['used_mib']>1024 or (card['total_mib']-card['used_mib'])/card['total_mib']<.98:
            raise ValueError('Existing allocation on GPU '+index+'; utilization zero does not authorize use')
    return {i:cards[i]for i in ids}


def validate_data(path,manifest):
    path=Path(path)
    if not manifest.get('no_truncation')or not manifest.get('task_groups_disjoint'):raise ValueError('Dataset split/truncation contract missing')
    if not manifest.get('formal_training_eligible'):raise ValueError('Formal data not approved')
    expected={item['path']for item in manifest['artifacts']}
    actual={str(p.relative_to(path))for p in path.rglob('*')if p.is_file()}
    if actual!=expected:raise ValueError('Dataset cache file inventory changed')
    for item in manifest['artifacts']:
        target=path/item['path']
        if target.is_symlink()or target.stat().st_size!=item['bytes']or sha256(target)!=item['sha256']:
            raise ValueError('Dataset cache corruption: '+item['path'])
    if any(not manifest['counts'].get(split,{}).get('rows')or not manifest['counts'][split].get('supervised_tokens')for split in ('train','validation')):
        raise ValueError('Empty split or all labels masked')


def exported_m1(path):
    path=Path(path);state=json.loads((path/'trainer_state.json').read_text())
    if (state.get('epoch',0)<1 or state.get('max_steps',0)<1
        or state.get('global_step')!=state['max_steps']):
        raise ValueError('Final native training budget incomplete; an early stopped epoch is not the final model')
    index=path/'model.safetensors.index.json'
    names=set(json.loads(index.read_text())['weight_map'].values())if index.exists()else {'model.safetensors'}
    if not(path/'config.json').is_file()or any(not(path/name).is_file()for name in names):raise ValueError('M1 final HF export incomplete; DeepSpeed shards are not a final export')


def validate_resume(checkpoint,output,stage,config_sha256):
    checkpoint=Path(checkpoint).resolve()
    if checkpoint.parent!=Path(output).resolve():raise ValueError('Resume must belong to the same stage')
    validate_checkpoint(checkpoint,require_marker=True)
    identity=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_text())['identity']
    if identity.get('stage')!=stage or identity.get('config_sha256')!=config_sha256:
        raise ValueError('Checkpoint stage/configuration identity mismatch')


def preflight(stage,ids,resume=None,check_gpu=True):
    root=workspace();runtime=runtime_config();path=config_path(stage)
    cfg=validate_native(yaml.safe_load(path.read_text()))
    storage=json.loads(storage_settings().read_text())
    output=Path(cfg['output_dir']).resolve();hot=Path(storage['hot_root'])
    if hot.stat().st_dev!=storage['hot_device']:raise ValueError('Hot storage mount changed')
    if shutil.disk_usage(hot).free<storage['minimum_formal_start_free_bytes']:raise ValueError('Insufficient hot checkpoint storage')
    if not os.access(hot,os.W_OK):raise ValueError('Hot storage is not writable')
    manifest=json.loads(data_manifest_path(stage).read_text())
    validate_data(cfg['tokenized_path'],manifest)
    if manifest['model_revision']!=cfg['model_revision']:raise ValueError('Dataset/model revision mismatch')
    if manifest['cutoff']!=cfg['cutoff_len']:raise ValueError('Dataset length/config mismatch')
    if cfg.get('runtime_recipe')=='qwen35_772_b32_e5':
        from .public_stages import validate_stage,validate_initialization
        validate_stage(stage,cfg,manifest);validate_initialization(stage,cfg)
    if stage in CALIBFORGE_STAGES:
        from .calibforge import validate_pair,validate_dataset,validate_initialization,validate_predecessor,validate_stage_order
        validate_predecessor();validate_stage_order(stage)
        validate_pair();validate_dataset(cfg,manifest)
        validate_initialization(stage,cfg,verify_weights=True)
    if stage.startswith('m2-'):
        contract=root/'configs/m2-tool-contract.json'
        if manifest.get('tool_contract_sha256')!=sha256(contract):raise ValueError('M2 data/tool contract identity mismatch')
        exported_m1(cfg['model_name_or_path'])
        from .hf_export import final_verified
        if not final_verified(cfg['model_name_or_path']):raise ValueError('M1 final HF export needs sealed integrity and actual load/generation smoke')
        if Path(cfg['model_name_or_path']).resolve()!=Path(yaml.safe_load(config_path('m1-64k').read_text())['output_dir']).resolve():raise ValueError('M2 must continue from actual M1 final weights')
    if resume:
        validate_resume(resume,output,stage,sha256(path))
    elif output.exists()and any(output.iterdir()):raise ValueError('Existing output preserved; explicit same-stage resume required')
    ready=json.loads((root/'checks/readiness.json').read_text())
    if not ready['stages'].get(stage,{}).get('ready_to_train'):raise ValueError('Readiness has unverified acceptance items')
    from .readiness import model_verified,VERSIONS
    if not model_verified():raise ValueError('Fixed model identity is no longer verified')
    acceptance=ready['stages'][stage]
    from .identity import training_source_sha256
    if acceptance.get('training_source_sha256')!=training_source_sha256():
        raise ValueError('Training source changed after Readiness')
    if acceptance.get('deepspeed_sha256')!=sha256(cfg['deepspeed']):raise ValueError('DeepSpeed configuration changed after Readiness')
    if acceptance.get('config_sha256')!=sha256(path)or acceptance.get('versions_sha256')!=sha256(VERSIONS):
        raise ValueError('Configuration or dependency identity changed after Readiness')
    if stage in CALIBFORGE_STAGES:
        for field in ('initial_models_manifest','tool_contract','prior_campaign_report'):
            if acceptance.get(field+'_sha256')!=sha256(runtime[field]):
                raise ValueError('CalibForge frozen '+field+' changed after Readiness')
        if acceptance.get('data_manifest_sha256')!=sha256(data_manifest_path(stage)):
            raise ValueError('Shared CalibForge data changed after Readiness')
    devices=gpu_check(ids,runtime)if check_gpu else None
    return cfg,devices


def launch(stage,ids,job_id,resume=None):
    root=workspace();runtime=runtime_config()
    if not runtime['formal_training_authorized']or runtime['formal_authorized_stage']!=stage:raise ValueError('Explicit authorization for this formal stage is required')
    with ExitStack()as stack:
        paths={root/'locks'/f'{stage}.lock',root/'locks/training.lock',
            Path(runtime.get('project_training_lock',root/'locks/training.lock'))}
        for path in sorted(paths):
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,devices=preflight(stage,ids,resume)
        config=config_path(stage);output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
        env=os.environ|dict(QWEN35_JOB_ID=job_id,QWEN35_STAGE=stage,QWEN35_CONFIG_SHA256=sha256(config),
            QWEN35_MODEL_PATH=runtime['model_path'],CUDA_VISIBLE_DEVICES=','.join(devices[i]['uuid']for i in ids),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        cmd=[runtime['python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(len(ids)),
             '--max-restarts=0','--module','sft.runtime.lf_train',str(config),'--archive-checkpoints']
        if resume:cmd+=['--resume',str(resume)]
        log=root/'jobs'/job_id/'training.log'
        # No periodic exits, timeout wrapper, retry, or new optimizer during a stage.
        with log.open('x')as stream:result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        attempt=dict(job_id=job_id,config_sha256=sha256(config),devices=devices,returncode=result.returncode,
            log=str(log),resume=resume,formal_training=True)
        atomic_json(root/'jobs'/job_id/'attempt.json',attempt)
        return result.returncode
