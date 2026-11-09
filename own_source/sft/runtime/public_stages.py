"""The fixed CF416 -> TMax356 route on the existing native LF job lifecycle."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

import yaml

from .checkpoint_archive import atomic_json,sha256,storage_settings
from .config import validate_native,native_arguments
from .identity import training_source_sha256
from .paths import workspace,runtime_config,config_path,data_manifest_path,PUBLIC_STAGES,PUBLIC_SOURCES


def validate_stage(stage,cfg,manifest):
    dataset,revision,tasks=PUBLIC_SOURCES[stage]
    expected=dict(kind='QWEN35_772_NATIVE_STAGE',stage=stage,source_dataset=dataset,
        source_revision=revision,source_primary_task_count=tasks,supervision_scope='all_assistant_once',
        original_provider_prefixes_verified=True,finish_enabled=True,tool_result_budget_enabled=True)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen task/source/11-tool data identity differs')
    if cfg['runtime_recipe']!='qwen35_772_b32_e5':raise ValueError('Wrong formal recipe')
    if manifest['cutoff']!=cfg['cutoff_len']:raise ValueError('Cache/context identity differs')
    release=Path(manifest['release_manifest'])
    if sha256(release)!=manifest['release_manifest_sha256']:raise ValueError('Final data release changed')
    value=json.loads(release.read_text())
    if not(value['complete']and value['all_native_sglang_prefix_targets_verified']
        and value['task_family_split_verified']and value['tb21_independent_leakage_audit_passed']):
        raise ValueError('Complete 772 data admission is not proven')
    if manifest['tool_contract_sha256']!=sha256(runtime_config()['tool_contract']):
        raise ValueError('Final eleven-tool contract changed')


def validate_initialization(stage,cfg):
    runtime=runtime_config();model=Path(cfg['model_name_or_path']).resolve()
    if stage==PUBLIC_STAGES[0]:
        from .readiness import model_verified
        if model!=Path(runtime['model_path']).resolve()or not model_verified():
            raise ValueError('M1 must initialize from the verified original M0')
    else:
        from .hf_export import final_verified
        previous=yaml.safe_load(config_path(PUBLIC_STAGES[0]).read_text())
        if model!=Path(previous['output_dir']).resolve()or not final_verified(model):
            raise ValueError('M2 requires the completed new M1 final HF')
        state=json.loads((model/'trainer_state.json').read_text())
        if state['epoch']<4.99 or state['global_step']!=state['max_steps']:
            raise ValueError('New M1 has not completed five epochs')
        proof=json.loads((model/'TOOL_PROTOCOL_SMOKE.json').read_text())
        if not proof['passed']or proof['export_manifest_sha256']!=sha256(model/'HF_EXPORT_COMPLETE.json'):
            raise ValueError('M1 cold-load/tool/finish acceptance missing')
    if cfg.get('resume_from_checkpoint'):raise ValueError('A new stage starts new Adam and scheduler')


def check(native=True):
    from .readiness import dependencies_verified,model_verified,VERSIONS
    from .launch import validate_data,gpu_check
    root=workspace();runtime=runtime_config();checks={};details={};stages={}
    def probe(name,fn):
        try:checks[name]=fn()is not False
        except (OSError,ValueError,KeyError,TypeError,AssertionError,ImportError,subprocess.SubprocessError)as e:
            checks[name]=False;details[name]=type(e).__name__+': '+str(e)
    probe('dependencies',dependencies_verified);probe('original_model',model_verified)
    probe('gpu_available',lambda:gpu_check(runtime['planned_gpu_ids'],runtime))
    def storage():
        d=json.loads(storage_settings().read_text());p=Path(d['hot_root'])
        import shutil
        return p.stat().st_dev==d['hot_device']and shutil.disk_usage(p).free>=d['minimum_formal_start_free_bytes']
    probe('storage',storage)
    def capacity():
        p=Path(runtime['capacity_acceptance']);value=json.loads(p.read_text())
        if not value['passed']or value['source_sha256']!=training_source_sha256():raise ValueError('Current native capacity source is unverified')
        if value['global_batch']!=32 or value['dp']!=runtime['max_concurrent_gpus']:raise ValueError('Accepted actual DP/batch changed')
        return value
    probe('actual_update_save_resume',capacity)
    for stage in PUBLIC_STAGES:
        def config(stage=stage):validate_native(yaml.safe_load(config_path(stage).read_text()))
        probe(stage+'_config',config)
        if native and checks[stage+'_config']:
            def parse(stage=stage):
                from llamafactory.hparams import get_train_args
                cfg=yaml.safe_load(config_path(stage).read_text())
                get_train_args(native_arguments(cfg)|dict(use_cpu=True,do_train=False,do_eval=False,bf16=False,deepspeed=None))
            probe(stage+'_native_config',parse)
        else:checks[stage+'_native_config']=not native and checks[stage+'_config']
        def data(stage=stage):
            cfg=yaml.safe_load(config_path(stage).read_text());m=json.loads(data_manifest_path(stage).read_text())
            validate_data(cfg['tokenized_path'],m);validate_stage(stage,cfg,m)
            accepted=json.loads(Path(runtime['capacity_acceptance']).read_text())
            if accepted['context']!=cfg['cutoff_len']:raise ValueError('Context differs from actual capacity acceptance')
        probe(stage+'_data',data)
        probe(stage+'_initialization',lambda stage=stage:validate_initialization(stage,yaml.safe_load(config_path(stage).read_text())))
        names=['dependencies','original_model','gpu_available','storage','actual_update_save_resume']+[stage+'_'+x for x in('config','native_config','data','initialization')]
        cfg=yaml.safe_load(config_path(stage).read_text())if config_path(stage).exists()else{}
        stages[stage]=dict(ready_to_train=all(checks.get(n,False)for n in names),
            blockers=[dict(check=n,detail=details.get(n,'Not verified'))for n in names if not checks.get(n,False)],
            training_source_sha256=training_source_sha256(),versions_sha256=sha256(VERSIONS),
            config_sha256=sha256(config_path(stage))if cfg else None,
            deepspeed_sha256=sha256(cfg['deepspeed'])if cfg else None)
    result=dict(checked_at=time.time(),checks=checks,details=details,stages=stages)
    atomic_json(root/'checks/readiness.json',result);return result


def advance():
    from . import jobs
    from .hf_export import final_verified,smoke
    root=workspace();runtime=runtime_config()
    with(root/'locks/sequence-advance.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='EXISTING_ADVANCEMENT_RUNNING')
        for stage in PUBLIC_STAGES:
            job_id='new772-'+stage;folder=root/'jobs'/job_id
            if not(folder/'job.json').exists():
                ready=check()
                if not ready['stages'][stage]['ready_to_train']:
                    return dict(status='BLOCKED_BY_ACTUAL_ADMISSION',stage=stage,blockers=ready['stages'][stage]['blockers'])
                runtime.update(formal_training_authorized=True,formal_authorized_stage=stage)
                atomic_json(root/'configs/runtime.json',runtime)
                return dict(status='NATIVE_STAGE_SUBMITTED',job=jobs.submit(stage,runtime['planned_gpu_ids'],job_id=job_id))
            job=jobs.status(job_id)
            if job['status']!='completed':return dict(status='OBSERVE_OR_INSPECT_ORIGINAL_NATIVE_JOB',stage=stage,job=job)
            cfg=yaml.safe_load(config_path(stage).read_text());model=Path(cfg['output_dir'])
            if not final_verified(model):smoke(model)
            protocol=model/'TOOL_PROTOCOL_SMOKE.json'
            if not protocol.exists():
                cmd=runtime.get('hf_protocol_smoke_command')
                if not cmd:return dict(status='WAITING_NATIVE_COLD_TOOL_SMOKE',stage=stage)
                subprocess.run(cmd+[str(model)],check=True)
            p=json.loads(protocol.read_text())
            if not p['passed']or p['export_manifest_sha256']!=sha256(model/'HF_EXPORT_COMPLETE.json'):
                raise ValueError('Actual tool/finish smoke failed')
            atomic_json(folder/'HF_VERIFIED.json',dict(model=str(model),export_manifest_sha256=sha256(model/'HF_EXPORT_COMPLETE.json')))
        return dict(status='NEW_M1_AND_M2_HF_READY_FOR_TB21_N3',old_four_gpu_dispatch_allowed=False)
