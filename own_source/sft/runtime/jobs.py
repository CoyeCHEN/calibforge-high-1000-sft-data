"""Detached user services; manifests observe jobs, never act as a scheduler."""
import datetime
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

from .checkpoint_archive import atomic_json,sha256,storage_settings
from .paths import workspace,runtime_config,config_path,data_manifest_path
from .identity import training_source_sha256


def process_identity(pid=None):
    pid=pid or os.getpid()
    return dict(pid=pid,startticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def process_matches(identity):
    try:return process_identity(identity['pid'])==identity
    except (OSError,KeyError,ProcessLookupError):return False


def require_linger():
    value=subprocess.check_output(['loginctl','show-user',str(os.getuid()),'--property=Linger','--value'],text=True,timeout=15).strip()
    if value!='yes':raise ValueError('User linger must be enabled before detached jobs; ask the host administrator, do not rely on SSH')


def submit(stage,ids,resume=None,job_id=None,probe_seconds=0):
    root=workspace();runtime=runtime_config()
    require_linger()
    if probe_seconds and not 1<=probe_seconds<=60:raise ValueError('CPU probe duration must be 1..60 seconds')
    if not probe_seconds and(not runtime['formal_training_authorized']or runtime['formal_authorized_stage']!=stage):
        raise ValueError('Explicit formal stage authorization is required before submission')
    job_id=job_id or 'sft-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    if not re.fullmatch('[A-Za-z0-9_-]+',job_id):raise ValueError('Unsafe job identity')
    folder=root/'jobs'/job_id;folder.mkdir(mode=0o750)  # Atomic duplicate rejection, including after completion.
    unit=runtime['service_prefix']+'-'+job_id+'.service'
    config=config_path(stage)
    job=dict(job_id=job_id,unit=unit,stage=stage,gpu_ids=ids,resume=resume,probe_seconds=probe_seconds,
        status='submitted',submitted_at=time.time(),host=socket.gethostname(),config_sha256=sha256(config),
        training_source_sha256=training_source_sha256(),source_commit=runtime.get('source_commit'),
        data_manifest_sha256=None if probe_seconds else sha256(data_manifest_path(stage)),
        no_local_heartbeat=True,automatic_restart=False)
    atomic_json(folder/'job.json',job)
    args=['systemd-run','--user','--unit='+unit,'--property=Type=exec','--property=Restart=no',
        '--property=KillMode=control-group','--property=KillSignal=SIGTERM','--property=SendSIGKILL=yes',
        '--property=TimeoutStopSec=60','--property=OOMPolicy=stop','--property=UMask=0027',
        '--property=MemoryMax=192G','--setenv=OPENHARNESS_SFT_WORKSPACE='+str(root),
        '--setenv=PYTHONPATH='+str(Path(__file__).resolve().parents[2]),
        '--property=StandardOutput=append:'+str(folder/'service.log'),
        '--property=StandardError=append:'+str(folder/'service.log')]
    if probe_seconds:args+=['--property=RuntimeMaxSec='+str(probe_seconds+30)]
    args+=[runtime['python'],'-m','sft','--workspace',str(root),'worker',job_id]
    try:subprocess.run(args,check=True,timeout=30,capture_output=True,text=True)
    except Exception as error:
        # A lost systemd-run response does not imply that the remote service
        # failed. Never overwrite a worker's running/completed manifest or retry.
        atomic_json(folder/'submission-error.json',dict(status='unknown',error=type(error).__name__,unit=unit))
        raise
    return job


def work(job_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',job_id):raise ValueError('Unsafe job identity')
    root=workspace();folder=root/'jobs'/job_id;path=folder/'job.json';job=json.loads(path.read_text())
    with(folder/'worker.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if job['status']!='submitted':raise ValueError('Job was already accepted; no replay')
        if job['config_sha256']!=sha256(config_path(job['stage'])):raise ValueError('Submitted config changed')
        if job['training_source_sha256']!=training_source_sha256():raise ValueError('Submitted training source changed')
        if not job['probe_seconds']and job['data_manifest_sha256']!=sha256(data_manifest_path(job['stage'])):
            raise ValueError('Submitted data manifest changed')
        job.update(status='running',process=process_identity(),started_at=time.time());atomic_json(path,job)
        result=1
        try:
            if job['probe_seconds']:
                for second in range(job['probe_seconds']):
                    atomic_json(folder/'probe.json',dict(process=process_identity(),second=second))
                    time.sleep(1)
                result=0
            else:
                # Only the child launch imports the GPU training framework. This observer
                # samples native allocations; it never creates a CUDA context or heartbeat lease.
                cmd=[runtime_config()['python'],'-m','sft','--workspace',str(root),'launch-internal',job_id]
                with(folder/'launcher.log').open('x')as log,(folder/'gpu-samples.jsonl').open('x')as samples:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    job['launcher_process']=process_identity(child.pid)
                    try:atomic_json(path,job)
                    except OSError:logging.exception('Non-critical launcher observation write failed; child continues')
                    sample_writes=True
                    while True:
                        sample=dict(at=time.time())
                        for name,query in [('gpu','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'),
                                           ('processes','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')]:
                            try:sample[name]=subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader'],text=True,timeout=15).splitlines()
                            except (OSError,subprocess.SubprocessError)as error:sample[name]=type(error).__name__
                        if sample_writes:
                            try:samples.write(json.dumps(sample)+'\n');samples.flush()
                            except OSError:
                                logging.exception('Non-critical GPU sample write failed; child continues')
                                sample_writes=False
                        try:result=child.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
        except Exception as error:
            job['error']=type(error).__name__+': '+str(error)
            raise
        finally:
            job.update(status='completed'if result==0 else'failed',returncode=result,ended_at=time.time())
            try:atomic_json(path,job)
            except OSError:logging.exception('Job observation write failed; native child return code is %s',result)
        return result


def status(job_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',job_id):raise ValueError('Unsafe job identity')
    path=workspace()/'jobs'/job_id/'job.json';job=json.loads(path.read_text())
    if job['unit']!=runtime_config()['service_prefix']+'-'+job_id+'.service':raise ValueError('Job unit does not belong to this workspace')
    result=subprocess.run(['systemctl','--user','show',job['unit'],'-p','ActiveState','-p','SubState','-p','Result','-p','MainPID'],
                          capture_output=True,text=True,timeout=20)
    job['systemd']={k:v for line in result.stdout.splitlines()if'='in line for k,v in [line.split('=',1)]}
    job['process_identity_matches']=process_matches(job.get('process',{}))
    if job['status']in ('running','submitted')and job['systemd'].get('ActiveState')not in ('active','activating'):
        job['observed_status']='interrupted; inspect last complete checkpoint; never auto-restart'
    return job


def install_archive():
    root=workspace();config=runtime_config();storage=json.loads(storage_settings().read_text())
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    service=storage['archive_service'];prefix=service.removesuffix('.service')
    (units/service).write_text(f'''[Unit]
Description=Immutable OpenHarness SFT checkpoint archive
StartLimitIntervalSec=60
StartLimitBurst=3
[Service]
Type=oneshot
Environment=OPENHARNESS_SFT_WORKSPACE={root}
Environment=PYTHONPATH={Path(__file__).resolve().parents[2]}
ExecStart={config['python']} -m sft --workspace {root} archive-worker
TimeoutStartSec=2h
MemoryMax=512M
CPUQuota=100%
Nice=10
IOSchedulingClass=idle
UMask=0027
KillMode=control-group
Restart=no
''')
    (units/(prefix+'.path')).write_text(f'''[Unit]
Description=Local immutable SFT archive queue
[Path]
DirectoryNotEmpty={storage['queue_root']}
Unit={service}
[Install]
WantedBy=default.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',prefix+'.path'],check=True)
    return dict(archive_service=service,path_unit=prefix+'.path')
