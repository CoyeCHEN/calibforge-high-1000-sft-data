"""CPU checks and read-only evidence joins; never infer GPU updates from YAML."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from .acceptance import capacity,continuous_acceptance
from .checkpoint_archive import atomic_json,sha256,storage_settings
from .config import validate_native,native_arguments
from .paths import workspace,runtime_config,config_path
from .launch import validate_data,exported_m1,gpu_check
from .new_entry_acceptance import new_entry_acceptance
from .identity import training_source_sha256

VERSIONS=Path(__file__).resolve().parents[1]/'versions.json'


def verify_model():
    model=Path(runtime_config()['model_path']);items=json.loads((VERSIONS.parent/'model-sha256.json').read_text())
    files=[]
    for item in items:
        path=model/item['file']
        if path.stat().st_size!=item['bytes']or sha256(path)!=item['sha256']:raise ValueError('Fixed model weight SHA mismatch')
        stat=path.stat();files.append(dict(file=item['file'],sha256=item['sha256'],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    result=dict(model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',files=files)
    atomic_json(workspace()/'checks/model-identity.json',result)
    return result


def model_verified():
    try:
        proof=json.loads((workspace()/'checks/model-identity.json').read_text());model=Path(runtime_config()['model_path'])
        expected=json.loads((VERSIONS.parent/'model-sha256.json').read_text())
        if {i['file']:i['sha256']for i in proof['files']}!={i['file']:i['sha256']for i in expected}:return False
        return proof['model_revision']=='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'and all(
            (model/i['file']).stat().st_size==i['bytes']and(model/i['file']).stat().st_mtime_ns==i['mtime_ns']for i in proof['files'])
    except (OSError,ValueError,KeyError):return False


def import_evidence(source):
    source=Path(source).resolve();config=runtime_config();paths={}
    for name in ('short-stateproof','m1-real','128k','128k-target','continuous','96k','96k-target'):
        cfg=yaml.safe_load((source/'configs'/f'verify-{name}.yaml').read_text())
        if cfg['model_revision']!='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'or not cfg['bf16']:raise ValueError('Evidence model identity changed')
        paths[name]=cfg['output_dir']
    config['gpu_evidence']=paths;config['gpu_evidence_scope']='Existing model/framework evidence reused; migrated code GPU canaries remain pending until explicitly run. Existing queue retained.'
    review=source/'gpu-watch-20261023/checks/128k-fallback-review.json'
    if review.exists():atomic_json(workspace()/'checks/128k-fallback-review.json',json.loads(review.read_text())|dict(source_sha256=sha256(review)))
    atomic_json(workspace()/'configs/runtime.json',config)
    return paths


def dependencies_verified():
    versions=json.loads(VERSIONS.read_text())
    try:
        if not all(metadata.version(name).split('+')[0]==version for name,version in versions['versions'].items()):return False
        import llamafactory
        lf=Path(llamafactory.__file__).parent
        return all(sha256(lf/name)==digest for name,digest in versions['LF_runtime_files_match_frozen_archive'].items())
    except (metadata.PackageNotFoundError,ImportError,OSError):return False


def check(native=True):
    if runtime_config().get('training_plan')=='qwen35_772_cf_tmax':
        from .public_stages import check as check_public
        return check_public(native=native)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    root=workspace();runtime=runtime_config();versions=json.loads(VERSIONS.read_text());checks={};details={}
    if runtime.get('training_recipe')=='calibforge_appendix_e':
        from .calibforge import check as check_calibforge
        return check_calibforge(native=native)
    for stage in ('m1-64k','m2-128k','m2-96k'):
        cfg=None
        try:
            cfg=validate_native(yaml.safe_load(config_path(stage).read_text()))
            if native:
                from llamafactory.hparams import get_train_args
                get_train_args(native_arguments(cfg)|dict(use_cpu=True,deepspeed=None))
            checks[stage+'_config']=True
        except (ValueError,OSError,ImportError)as error:checks[stage+'_config']=False;details[stage+'_config']=type(error).__name__+': '+str(error)
        try:
            if cfg is None:raise ValueError('Cannot validate data against a failed config')
            manifest=json.loads((root/'manifests'/f'{stage}.json').read_text());validate_data(cfg['tokenized_path'],manifest)
            checks[stage+'_data']=True
        except (ValueError,OSError,KeyError)as error:checks[stage+'_data']=False;details[stage+'_data']=type(error).__name__+': '+str(error)
    checks['dependencies']=dependencies_verified()
    checks['model_identity']=model_verified()
    try:
        proof=json.loads((root/'checks/m1-full-loader.json').read_text())
        data=json.loads((root/'manifests/m1-64k.json').read_text())
        checks['m1_full_loader']=bool(proof['passed']and proof['task_groups_disjoint']and
            proof['manifest_sha256']==data['imported_manifest_sha256']and
            {s:proof['counts'][s]['rows']for s in ('train','validation')}=={'train':5320,'validation':191})
    except (OSError,ValueError,KeyError):checks['m1_full_loader']=False
    try:
        os.environ['QWEN35_MODEL_PATH']=runtime['model_path']
        from sft import qwen35
        qwen35.MODEL=runtime['model_path'];qwen35.load_tokenizer();checks['tokenizer']=True
    except (OSError,ImportError,ValueError)as error:checks['tokenizer']=False;details['tokenizer']=type(error).__name__+': '+str(error)
    storage=json.loads(storage_settings().read_text());hot=Path(storage['hot_root'])
    checks['storage']=hot.stat().st_dev==storage['hot_device']and shutil.disk_usage(hot).free>=storage['minimum_formal_start_free_bytes']
    try:
        gpu_check(runtime.get('planned_gpu_ids',['0','1','2','3']),runtime);checks['gpu_available']=True
    except (OSError,ValueError,KeyError,subprocess.SubprocessError)as error:
        checks['gpu_available']=False;details['gpu_available']=str(error)
    try:
        contract=json.loads((root/'configs/m2-tool-contract.json').read_text())
        checks['m2_tool_contract']=bool(contract['status']=='APPROVED'and contract['schema_sha256']and contract['tools'])
    except (OSError,ValueError,KeyError):checks['m2_tool_contract']=False
    evidence={}
    for name,directory in runtime.get('gpu_evidence',{}).items():
        evidence[name]=continuous_acceptance(directory)if name=='continuous'else capacity(directory,name)
    current={name:new_entry_acceptance(runtime.get('new_entry_gpu_evidence',{}).get(name),length,supervision)
        for name,length,supervision in [('m1',65536,1),('m2',131072,1),('m2-target',131072,3000),('m2-96k',98304,3000)]}
    m1gpu=all(evidence.get(name,{}).get('passed')for name in ('short-stateproof','m1-real','continuous'))and current['m1']['passed']
    m2gpu=all(evidence.get(name,{}).get('passed')for name in ('128k','128k-target'))and current['m2']['passed']and current['m2-target']['passed']
    review=root/'checks/128k-fallback-review.json'
    fallback=json.loads(review.read_text())if review.exists()else{}
    m296=all(evidence.get(name,{}).get('passed')for name in ('96k','96k-target'))and current['m2-96k']['passed']and fallback.get('fallback_permitted')and fallback.get('technical_reason')
    m1=checks['m1-64k_config']and checks['m1-64k_data']and checks['m1_full_loader']and checks['storage']and checks['tokenizer']and checks['dependencies']and checks['model_identity']and m1gpu and checks['gpu_available']
    started_stages=[]
    for stage in ('m1-64k','m2-128k','m2-96k'):
        cfg=yaml.safe_load(config_path(stage).read_text())
        record=Path(cfg['output_dir'])/'preflight-evidence-rank0.jsonl'
        if record.exists():
            with record.open()as events:
                for line in events:
                    try:event=json.loads(line)
                    except ValueError:continue  # An interrupted final log line is not checkpoint evidence.
                    if event.get('event')=='train_begin':started_stages.append(stage);break
    m1export=False
    try:
        from .hf_export import final_verified
        m1export=final_verified(yaml.safe_load(config_path('m1-64k').read_text())['output_dir'])
    except (OSError,ValueError,KeyError):pass
    result=dict(checks=checks,details=details,gpu_evidence=evidence,new_entry_gpu_evidence=current,
        stages={'m1-64k':dict(ready_to_train=bool(m1)),
                'm2-128k':dict(ready_to_train=bool(m1 and m1export and m2gpu and checks['m2_tool_contract']and checks['m2-128k_config']and checks['m2-128k_data'])),
                'm2-96k':dict(ready_to_train=bool(m1 and m1export and m296 and checks['m2_tool_contract']and checks['m2-96k_config']and checks['m2-96k_data']),reason='Requires a separately approved real 128K failure review and 96K GPU evidence; never auto-lower length')},
        formal_training_started=bool(started_stages),started_stages=started_stages,automatic_formal_training=False,
        evidence_scope='CPU configuration/data checks plus existing rank records; not production reliability proof')
    for stage,item in result['stages'].items():
        item['config_sha256']=sha256(config_path(stage));item['versions_sha256']=sha256(VERSIONS)
        item['training_source_sha256']=training_source_sha256()
        item['deepspeed_sha256']=sha256(yaml.safe_load(config_path(stage).read_text())['deepspeed'])
    atomic_json(root/'checks/readiness.json',result)
    return result
