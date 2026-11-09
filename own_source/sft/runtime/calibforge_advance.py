"""One bounded advancement on the existing monitor; no timer, queue or retry loop."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from .checkpoint_archive import atomic_json, sha256
from .paths import workspace, runtime_config, config_path, data_manifest_path
from . import jobs

STAGES = ('m3-a-128k', 'c1-b-128k')
CGROUP_ROOT = Path('/sys/fs/cgroup')


def next_stage(records):
    """An unknown/failed original job never causes an automatic second attempt."""
    for stage in STAGES:
        job = records.get(stage)
        if job is None:
            return stage, 'NOT_SUBMITTED'
        if job.get('status') != 'completed':
            return stage, 'WAIT_OR_INSPECT_ORIGINAL_JOB'
        if not job.get('hf_verified'):
            return stage, 'VERIFY_FINAL_HF'
    return None, 'TRAINING_HF_COMPLETE_EVAL_HANDOFF_REQUIRED'


def release_prior_serving(runtime):
    from .calibforge import validate_predecessor, checked_json
    validate_predecessor()
    receipt = checked_json(runtime['prior_campaign_release_receipt'],
                           runtime['prior_campaign_release_receipt_sha256'])
    if (receipt.get('primary_report_sha256') != runtime['prior_campaign_report_sha256']
        or receipt.get('all_controllers_ended') is not True
        or receipt.get('campaign_running_containers') != []
        or receipt.get('source_verification') != 'LIVE_SSH_AFTER_PRIMARY_COMPLETION'):
        raise ValueError('Actual prior campaign drain/release evidence is missing')
    unit = runtime['prior_serving_unit']
    if not unit.endswith('.service') or '/' in unit:
        raise ValueError('Invalid project serving unit')
    raw = subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID',
                                   '-p','ControlGroup','-p','Result'],
                                  text=True, timeout=15)
    actual = dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    expected = runtime['prior_serving_process']
    if actual.get('MainPID') == '0' and actual.get('ActiveState') in ('inactive','failed'):
        if jobs.process_matches(expected):
            raise ValueError('Recorded serving process still exists despite the stopped unit')
        control_group=actual.get('ControlGroup','')
        if control_group:
            group=CGROUP_ROOT/control_group.lstrip('/')
            if group.is_dir() and any(path.read_text().strip() for path in group.rglob('cgroup.procs')):
                raise ValueError('Stopped serving unit still has live cgroup processes')
        # A SIGKILL exit may leave the unit failed after successful stop/cleanup.
        # This proves only the old allocation ended; gpu_check remains independent.
        return dict(already_stopped=True,unit=unit,unit_state=actual['ActiveState'],
                    unit_result=actual.get('Result'),recorded_process_gone=True,cgroup_empty=True)
    if int(actual.get('MainPID','0')) != expected['pid'] or not jobs.process_matches(expected):
        raise ValueError('Serving process changed; do not signal an unknown allocation')
    # Only the explicitly recorded campaign-owned systemd control group is stopped.
    subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=150)
    return dict(stopped_unit=unit, verified_process=expected)


def advance():
    if runtime_config().get('training_plan')=='qwen35_772_cf_tmax':
        from .public_stages import advance as advance_public
        return advance_public()
    from .calibforge import validate_predecessor, check, validate_pair
    from .hf_export import final_verified, smoke
    root=workspace();runtime=runtime_config();path=root/'checks/sequence-observation.json'
    result=dict(checked_at=time.time(),observer=jobs.process_identity(),actions=[],automatic_restart=False)
    with (root/'locks/sequence-advance.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='EXISTING_ADVANCEMENT_RUNNING')
        try:
            validate_predecessor()
            result['prior_serving']=release_prior_serving(runtime)
            manifest_path=data_manifest_path(STAGES[0])
            if not manifest_path.is_file():
                result.update(status='BLOCKED_BY_FROZEN_CORPUS',
                    detail='Task release is present; approved successful multi-turn train/validation corpus is not frozen.')
                return result
            # Preparation rejects missing approval, bad masks/splits, and changed source/cache hashes.
            canary=root/'configs/verify-calibforge-data.yaml'
            if not canary.exists():
                from .canary import prepare_calibforge
                result['canary_inputs']=prepare_calibforge()
            cfg=yaml.safe_load(canary.read_text());output=Path(cfg['output_dir'])
            acceptance=output/'new-entry-acceptance.json'
            unit=runtime['service_prefix']+'-verify-common-recipe.service'
            if not acceptance.exists():
                state=subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState'],text=True,timeout=15)
                fields=dict(line.split('=',1)for line in state.splitlines()if'='in line)
                if fields.get('LoadState')!='not-found':
                    result.update(status='WAIT_OR_INSPECT_ORIGINAL_CANARY',unit=unit)
                    return result
                if output.exists()and any(output.iterdir()):
                    raise ValueError('Unaccepted prior canary output exists; explicit investigation required')
                from .launch import gpu_check
                gpu_check(runtime['planned_gpu_ids'],runtime)
                command=['systemd-run','--user','--unit='+unit,'--property=Type=exec',
                    '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
                    '--property=RuntimeMaxSec=21720','--property=MemoryMax=192G',
                    '--setenv=OPENHARNESS_SFT_WORKSPACE='+str(root),
                    '--setenv=PYTHONPATH='+str(Path(__file__).resolve().parents[2]),
                    '--property=StandardOutput=append:'+str(root/'logs/common-canary.log'),
                    '--property=StandardError=append:'+str(root/'logs/common-canary.log'),
                    runtime['python'],'-m','sft.runtime.canary',str(canary),'--gpus',','.join(runtime['planned_gpu_ids'])]
                subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
                result.update(status='COMMON_GA32_CANARY_SUBMITTED',unit=unit)
                result['actions'].append('submit_one_bounded_native_canary')
                return result
            if not json.loads(acceptance.read_text()).get('passed'):
                raise ValueError('Common recipe canary failed; no automatic retry')
            runtime['calibforge_gpu_evidence']=str(output)
            atomic_json(root/'configs/runtime.json',runtime)
            records={}
            for stage in STAGES:
                job_id='calibforge-'+stage+'-20261027'
                record=root/'jobs'/job_id/'job.json'
                if record.exists():
                    observed=jobs.status(job_id)
                    proof=root/'jobs'/job_id/'HF_VERIFIED.json'
                    observed['hf_verified']=False
                    if proof.is_file():
                        model=Path(validate_pair()[stage]['output_dir'])
                        verified=json.loads(proof.read_text())
                        observed['hf_verified']=(verified.get('model')==str(model)
                            and verified.get('export_manifest_sha256')==sha256(model/'HF_EXPORT_COMPLETE.json')
                            and final_verified(model))
                    records[stage]=observed
            stage,action=next_stage(records)
            result.update(stage=stage,status=action)
            if action=='VERIFY_FINAL_HF':
                cfg=validate_pair()[stage];model=Path(cfg['output_dir'])
                if not final_verified(model):smoke(model)
                if not final_verified(model):raise ValueError('Final HF integrity/load/generation failed')
                atomic_json(root/'jobs'/records[stage]['job_id']/'HF_VERIFIED.json',dict(
                    checked_at=time.time(),model=str(model),export_manifest_sha256=sha256(model/'HF_EXPORT_COMPLETE.json')))
                result['actions'].append('verify_native_hf_export_and_independent_cpu_generation')
            elif action=='NOT_SUBMITTED':
                ready=check()
                if not ready['stages'][stage]['ready_to_train']:
                    result.update(status='BLOCKED_BY_READINESS',blockers=ready['stages'][stage]['blockers'])
                    return result
                runtime.update(formal_training_authorized=True,formal_authorized_stage=stage)
                atomic_json(root/'configs/runtime.json',runtime)
                submitted=jobs.submit(stage,runtime['planned_gpu_ids'],job_id='calibforge-'+stage+'-20261027')
                result.update(status='NATIVE_FORMAL_JOB_SUBMITTED',job_id=submitted['job_id'])
                result['actions'].append('submit_native_stage_after_real_gates')
            elif stage is None:
                finished=root/'checks/calibforge-evaluation-final.json'
                if finished.is_file():
                    evaluated=json.loads(finished.read_text())
                    if evaluated.get('scope')!='CALIBFORGE_AB_PREDECLARED_FIRST_89'or evaluated.get('completed_primary_trials')!=178:
                        raise ValueError('Final evaluation evidence scope differs')
                    intent=json.loads((root/'checks/evaluation-serving-submitted.json').read_text())
                    unit=runtime['evaluation_serving_unit']
                    current=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,timeout=10))
                    if current:
                        if (intent.get('unit')!=unit or intent.get('status')!='STARTED'
                            or current!=intent['process']['pid']or not jobs.process_matches(intent['process'])):
                            raise ValueError('Evaluation serving process identity changed; never signal an unknown allocation')
                        subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=150)
                        result['actions'].append('stop_only_verified_campaign_serving_after_all_trials_drained')
                    result.update(status='SEQUENCE_COMPLETED_EVIDENCE_READY',evaluation_report_sha256=sha256(finished))
                    return result
                models={}
                for alias,target in zip(('m3a','c1b'),STAGES):
                    model=Path(validate_pair()[target]['output_dir'])
                    exported=json.loads((model/'HF_EXPORT_COMPLETE.json').read_text())
                    weights=[{k:f[k]for k in ('path','bytes','sha256')}for f in exported['files']if f['path'].endswith('.safetensors')]
                    if not weights:raise ValueError('No final HF weights for evaluation')
                    version=(weights[0]['sha256']if len(weights)==1 else
                        hashlib.sha256(json.dumps(weights,sort_keys=True,separators=(',',':')).encode()).hexdigest())
                    models[alias]=dict(path=str(model),weight_sha256=version,base_revision=exported['base_revision'],
                        hf_export_manifest_sha256=sha256(model/'HF_EXPORT_COMPLETE.json'),
                        independent_load_generate_passed=True,files=exported['files'],
                        model_stage='M3_A_FROM_TB21_ADAPTED_M2'if alias=='m3a'else'C1_B_FROM_OFFICIAL_M0')
                atomic_json(root/'checks/evaluation-handoff.json',dict(status='TRAINING_HF_READY',models=models,
                    benchmark_trials_per_model=89,repeat_automatically=False,tool_contract_sha256=sha256(runtime['tool_contract'])))
                unit=runtime['evaluation_serving_unit'];intent=root/'checks/evaluation-serving-submitted.json'
                if not intent.exists():
                    from .launch import gpu_check
                    gpu_check(runtime['planned_gpu_ids'],runtime)
                    atomic_json(intent,dict(unit=unit,status='SUBMISSION_INTENT',created_at=time.time()))
                    subprocess.run(['systemctl','--user','start',unit],check=True,timeout=30)
                    pid=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,timeout=10))
                    if not pid:raise ValueError('Serving unit has no actual process; inspect, never blindly restart')
                    atomic_json(intent,dict(unit=unit,status='STARTED',process=jobs.process_identity(pid),created_at=time.time()))
                    result['actions'].append('start_project_serving_after_both_final_hf_verified')
            return result
        except (OSError,ValueError,KeyError,subprocess.SubprocessError)as error:
            result.update(status='BLOCKED_OR_FAILED_EVIDENCE',error=type(error).__name__+': '+str(error))
            return result
        finally:
            # This observation never signals or cleans up a healthy native training job.
            atomic_json(path,result)
            with (root/'logs/sequence-monitor.jsonl').open('a')as log:log.write(json.dumps(result)+'\n')


if __name__=='__main__':print(json.dumps(advance(),indent=2))
