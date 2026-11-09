"""Read-only CI observation, called by the existing twenty-minute timer."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def query(argv):
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr[-1500:])
    except subprocess.TimeoutExpired:
        return dict(status='UNKNOWN',reason='query timed out; no training control action taken')


def main(root):
    root=Path(root)
    state=dict(checked_at=datetime.now().astimezone().isoformat(),observer='read-only / existing timer',
        gpu=query(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader']),
        processes=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']),
        jobs=query(['systemctl','--user','list-units','--all','--plain','--no-pager','qwen35-ci*']),
        disk=query(['df','-h',str(root),'/data','/']),
        training=[],evaluation=[])
    runs=list((root/'runs').glob('*'))+list(root.glob('expansion-*/runs/*'))
    for run in sorted(runs):
        if not run.is_dir():continue
        item=dict(run=str(run),checkpoints=[x.name for x in sorted(run.glob('checkpoint-*'))])
        trainer=run/'trainer_state.json'
        if trainer.is_file():
            data=json.loads(trainer.read_text())
            item.update(global_step=data.get('global_step'),epoch=data.get('epoch'))
        evidence=run/'preflight-evidence-rank0.jsonl'
        if evidence.is_file():
            item['last_rank0_progress_mtime']=evidence.stat().st_mtime
            # Read only a bounded tail while the training process appends.
            with evidence.open('rb') as handle:
                handle.seek(max(0,evidence.stat().st_size-128*1024))
                lines=handle.read().splitlines()
            for line in reversed(lines):
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if record.get('event')=='optimizer_step':
                    item.update(global_step=record['global_step'],
                        last_optimizer_event_at=record['time'],
                        parameter_update_observed=record['parameter_update_observed'],
                        fp32_master=record['master_partition_dtypes'],
                        adam_moments=record['optimizer_moment_dtypes'],
                        peak_reserved_bytes_rank0=record['peak_reserved_bytes'])
                    break
        state['training'].append(item)
    for run in sorted((root/'evaluation').glob('*-isolated-r*')):
        files={str(p.with_suffix('')):p for p in run.glob('*/predictions/*/*/*.json')
               if not p.name.startswith('tmp_')}
        for path in run.glob('*/predictions/*/*/tmp_*.jsonl'):
            key=str(path.with_name(path.stem[4:]))
            if key not in files:files[key]=path
        notebooks=questions=0
        for path in files.values():
            try:
                data=([json.loads(line)for line in path.open()if line.strip()] if path.suffix=='.jsonl'
                      else list(json.loads(path.read_text()).values()))
                notebooks+=len(data);questions+=sum(len(row['prediction'])for row in data)
            except (OSError,json.JSONDecodeError):pass  # A live writer is not a failed evaluation.
        edition='isolated-'+run.name.rsplit('-isolated-',1)[1]
        protocol=root/f'configs/evaluation-protocol-{edition}.json'
        metadata=json.loads(protocol.read_text())if protocol.is_file()else {}
        state['evaluation'].append(dict(run=str(run),edition=edition,
            native_error_mapping=metadata.get('native_error_mapping','UNVERIFIED; earlier status mapping'),
            prediction_scopes=len(files),
            persisted_notebooks=notebooks,persisted_questions=questions,
            result_files=len(list(run.glob('*/results/*/*/*.json')))))
    path=root/'monitor.jsonl'
    if path.exists() and path.stat().st_size>8*1024*1024:
        for n in range(3,0,-1):
            old=Path(str(path)+'.'+str(n))
            if old.exists():old.replace(str(path)+'.'+str(n+1))
        path.replace(str(path)+'.1')
    with path.open('a') as handle:handle.write(json.dumps(state,ensure_ascii=False)+'\n')
    (root/'checks/latest-observation.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main(sys.argv[1])
