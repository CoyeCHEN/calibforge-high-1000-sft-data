"""One CPU-only observation per user-timer activation. Never dispatch training."""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time

from .checkpoint_archive import atomic_json
from .paths import runtime_config,workspace


def _observe():
    root=workspace();config=runtime_config();now=time.time()
    if socket.gethostname()!=config['host']:raise ValueError('Monitor host identity changed')
    state_path=root/'checks/gpu-status.json';old=json.loads(state_path.read_text())if state_path.exists()else{}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True,timeout=20)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=20)
    processes=[]
    for line in apps.splitlines():
        uuid,pid,name=map(str.strip,line.split(',',2));user='unknown'
        try:user=pwd.getpwuid(Path('/proc/'+pid).stat().st_uid).pw_name
        except (OSError,KeyError):pass
        processes.append(dict(gpu_uuid=uuid,pid=int(pid),name=name,user=user,is_sglang='sglang'in name.lower()))
    cards=[];since={};stable=[]
    recent=0<=now-old.get('checked_epoch',0)<=1800
    for line in rows.splitlines():
        index,uuid,total,used,free,util=map(str.strip,line.split(','));authorized=uuid in config['authorized_gpu_uuids']
        owners=[p for p in processes if p['gpu_uuid']==uuid]
        idle=authorized and not owners and int(used)<=1024 and int(free)/int(total)>=.98 and int(util)==0
        if idle:
            since[uuid]=old.get('idle_observed_since',{}).get(uuid,now)if recent else now
            if recent and now-since[uuid]>=1195:stable.append(index)
        cards.append(dict(index=index,uuid=uuid,total_mib=int(total),used_mib=int(used),free_mib=int(free),utilization=int(util),
                          authorized=authorized,idle_candidate=idle,processes=owners))
    result=dict(last_checked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),checked_epoch=now,
        monitor_pid=os.getpid(),gpu_status=cards,existing_processes=processes,authorized=True,
        available_gpu_ids=[c['index']for c in cards if c['idle_candidate']],stable_gpu_ids=stable,
        stable_idle=len(stable)>=4,gpu_test_ready=len(stable)>=4,idle_observed_since=since,
        sglang_running=any(p['is_sglang']for p in processes),automatic_training=False,notification_sent=False,
        blockers=[]if len(stable)>=4 else['Four authorized unallocated GPUs observed idle twice over 20 minutes required'])
    atomic_json(state_path,result)
    logger=logging.getLogger('sft-gpu-monitor');logger.setLevel(logging.INFO)
    handler=RotatingFileHandler(root/'logs/gpu-monitor.log',maxBytes=2*1024**2,backupCount=5)
    logger.addHandler(handler);logger.info(json.dumps(result));handler.close();logger.removeHandler(handler)
    return result


def observe():
    try:return _observe()
    except (OSError,ValueError,subprocess.SubprocessError)as error:
        result=dict(last_checked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checked_epoch=time.time(),
            gpu_status=[],available_gpu_ids=[],existing_processes=[],authorized=False,stable_idle=False,gpu_test_ready=False,
            automatic_training=False,blockers=[type(error).__name__+': '+str(error)])
        atomic_json(workspace()/'checks/gpu-status.json',result)
        return result


def install():
    config=runtime_config();root=workspace();prefix=config['service_prefix']+'-gpu-monitor'
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    (units/(prefix+'.service')).write_text(f'''[Unit]
Description=CPU-only OpenHarness SFT GPU observation
[Service]
Type=oneshot
Environment=OPENHARNESS_SFT_WORKSPACE={root}
Environment=PYTHONPATH={Path(__file__).resolve().parents[2]}
ExecStart={config['python']} -m sft --workspace {root} monitor
TimeoutStartSec=90
MemoryMax=128M
CPUQuota=10%
Nice=10
''')
    (units/(prefix+'.timer')).write_text(f'''[Unit]
Description=Observe authorized SFT GPUs every 20 minutes
[Timer]
OnStartupSec=5
OnUnitActiveSec=20min
AccuracySec=1s
[Install]
WantedBy=timers.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',prefix+'.timer'],check=True)
    return dict(timer=prefix+'.timer',automatic_training=False)
