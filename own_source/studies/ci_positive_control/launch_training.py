"""One bounded submission entry; reuse resource checks, locks and native torchrun."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from sft.runtime.launch import gpu_check,validate_resume
from sft.runtime.jobs import process_identity,require_linger


def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--gpus',required=True)
    p.add_argument('--python',required=True,help='Existing validated training venv Python on this host')
    p.add_argument('--resume');a=p.parse_args();root=Path(a.root).resolve()
    config=root/'configs/train-r2.json';spec=json.loads(config.read_text())
    contract=json.loads((root/'configs/resource-authorization.json').read_text())
    ids=a.gpus.split(',');output=Path(spec['LF_arguments']['output_dir'])
    require_linger()
    baseline=json.loads((root/'checks/baseline-policy.json').read_text())
    assert baseline['mode']=='parallel_frozen_m0'
    assert baseline['model_revision']==spec['LF_arguments']['model_revision']
    assert baseline['model_manifest_sha256']==hashlib.sha256((root/'checks/m0-model-files.json').read_bytes()).hexdigest()
    for manifest in ['training-input-files.json','training-source-files.json']:
        for relative,wanted in json.loads((root/'checks'/manifest).read_text()).items():
            with (root/relative).open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=wanted:raise ValueError('Frozen training input/source changed: '+relative)
    if shutil.disk_usage(root).free<spec['minimum_free_disk_bytes']:
        raise OSError('Insufficient disk headroom for the frozen native checkpoint/export budget')
    with ExitStack() as stack:
        paths=[root/'runtime/ci-training.lock']+[Path('/data/chenkeyi/gpu-execution-locks')/(uuid+'.lock')
            for uuid in contract['authorized_gpu_uuids'] if uuid in
            [contract['index_to_uuid'][i]for i in ids]]
        for path in sorted(paths):
            path.parent.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        devices=gpu_check(ids,contract)
        cfgsha=hashlib.sha256(config.read_bytes()).hexdigest()
        if list(output.glob('PROCESS_COMPLETE-rank*.json')):
            raise ValueError('Training already completed; no duplicate epoch replay')
        if a.resume:validate_resume(Path(a.resume),output,'ci-public-control',cfgsha)
        elif output.exists() and any(output.iterdir()):raise FileExistsError('Existing run preserved; explicit complete checkpoint required')
        identity=process_identity()
        evidence=dict(host=socket.gethostname(),launcher=identity,devices=devices,
                      config_sha256=cfgsha,resume=a.resume,source='native LF + global token normalization')
        with (root/f"checks/actual-training-launch-{identity['pid']}-{identity['startticks']}.json").open('x') as handle:
            handle.write(json.dumps(evidence,indent=2)+'\n')
        python=str(Path(a.python).absolute())
        if not os.access(python,os.X_OK):raise ValueError('Training Python is not executable')
        trainbin=str(Path(python).parent)
        source_root=Path(__file__).resolve().parents[2]
        env=os.environ|dict(CUDA_VISIBLE_DEVICES=','.join(devices[i]['uuid']for i in ids),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            PYTHONPATH=str(source_root),
            PATH=trainbin+':/usr/local/cuda/bin:'+os.environ['PATH'])
        cmd=[python,'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc_per_node=4','--max-restarts=0',
             str(source_root/'studies/ci_positive_control/train_ci.py'),str(config)]
        if a.resume:cmd+=['--resume',a.resume]
        raise SystemExit(subprocess.call(cmd,env=env))


if __name__=='__main__':main()
