"""The single public entry point for preparation, remote jobs, and recovery."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',required=True)
    commands=parser.add_subparsers(dest='command',required=True)
    init=commands.add_parser('init');init.add_argument('--model',required=True);init.add_argument('--hot',required=True);init.add_argument('--archive',required=True)
    init.add_argument('--gpu-uuids',required=True);init.add_argument('--transfer-workers',type=int,choices=[1,2],default=1)
    m1=commands.add_parser('prepare-m1');m1.add_argument('--train',required=True);m1.add_argument('--validation',required=True)
    adopt=commands.add_parser('adopt-m1');adopt.add_argument('manifest')
    m2=commands.add_parser('prepare-m2');m2.add_argument('approval');m2.add_argument('--length',type=int,choices=[131072,98304],default=131072)
    calib=commands.add_parser('prepare-calibforge');calib.add_argument('approval');calib.add_argument('--tool-contract',required=True)
    calib_canary=commands.add_parser('prepare-calibforge-canary');calib_canary.add_argument('--name',default='verify-calibforge-data')
    release=commands.add_parser('adopt-m2-candidates');release.add_argument('approval')
    commands.add_parser('check')
    commands.add_parser('advance')
    commands.add_parser('verify-model')
    evidence=commands.add_parser('evidence');evidence.add_argument('--source',required=True)
    from .runtime.paths import STAGES
    run=commands.add_parser('run');run.add_argument('--stage',choices=STAGES,default='m1-64k')
    run.add_argument('--gpus',default='');run.add_argument('--resume');run.add_argument('--job-id');run.add_argument('--probe-seconds',type=int,default=0)
    for name in ('worker','launch-internal','status','stop'):
        sub=commands.add_parser(name);sub.add_argument('job_id')
    commands.add_parser('install-archive');commands.add_parser('archive-worker')
    commands.add_parser('monitor');commands.add_parser('install-monitor')
    freeze=commands.add_parser('freeze-collection');freeze.add_argument('--scope',required=True);freeze.add_argument('--base-job',required=True)
    freeze.add_argument('--wheel',required=True);freeze.add_argument('--teacher',required=True);freeze.add_argument('--version-basis',required=True)
    approve=commands.add_parser('approve-m2');approve.add_argument('candidates')
    approve_calib=commands.add_parser('approve-calibforge');approve_calib.add_argument('candidates')
    restore=commands.add_parser('restore');restore.add_argument('--archive',required=True);restore.add_argument('--destination',required=True)
    args=parser.parse_args();os.environ['OPENHARNESS_SFT_WORKSPACE']=str(Path(args.workspace).resolve())
    if args.command=='advance':
        from .runtime.calibforge_advance import advance
        result=advance()
    elif args.command=='init':
        from .runtime.config import initialize
        result=initialize(args.workspace,args.model,args.hot,args.archive,args.gpu_uuids.split(','),args.transfer_workers)
    elif args.command=='prepare-m1':
        from .prepare import prepare_m1
        from .runtime.paths import runtime_config
        from . import qwen35
        qwen35.MODEL=runtime_config()['model_path'];result=prepare_m1(args.train,args.validation)
    elif args.command=='adopt-m1':
        from .prepare import adopt_m1
        result=adopt_m1(args.manifest)
    elif args.command=='prepare-m2':
        from .prepare import prepare_m2
        from .runtime.paths import runtime_config
        from . import qwen35
        qwen35.MODEL=runtime_config()['model_path'];result=prepare_m2(args.approval,args.length)
    elif args.command=='adopt-m2-candidates':
        from .prepare import adopt_m2_candidates
        result=adopt_m2_candidates(args.approval)
    elif args.command=='prepare-calibforge':
        from .prepare import prepare_calibforge
        from .runtime.paths import runtime_config
        from . import qwen35
        qwen35.MODEL=runtime_config()['model_path'];result=prepare_calibforge(args.approval,args.tool_contract)
    elif args.command=='prepare-calibforge-canary':
        from .runtime.canary import prepare_calibforge
        result=prepare_calibforge(args.name)
    elif args.command=='check':
        from .runtime.readiness import check
        result=check()
    elif args.command=='verify-model':
        from .runtime.readiness import verify_model
        result=verify_model()
    elif args.command=='evidence':
        from .runtime.readiness import import_evidence
        result=import_evidence(args.source)
    elif args.command=='run':
        from .runtime.jobs import submit
        result=submit(args.stage,args.gpus.split(',')if args.gpus else[],args.resume,args.job_id,args.probe_seconds)
    elif args.command=='worker':
        from .runtime.jobs import work
        return work(args.job_id)
    elif args.command=='launch-internal':
        from .runtime.launch import launch
        from .runtime.paths import workspace
        job=json.loads((workspace()/'jobs'/args.job_id/'job.json').read_text())
        return launch(job['stage'],job['gpu_ids'],args.job_id,job['resume'])
    elif args.command=='status':
        from .runtime.jobs import status
        result=status(args.job_id)
    elif args.command=='stop':
        import subprocess
        from .runtime.jobs import status
        job=status(args.job_id)
        subprocess.run(['systemctl','--user','stop',job['unit']],check=True)
        result=dict(stopped_unit=job['unit'],resume='Explicit last complete same-stage checkpoint only; no automatic restart')
    elif args.command=='install-archive':
        from .runtime.jobs import install_archive
        result=install_archive()
    elif args.command=='archive-worker':
        from .runtime.checkpoint_archive import worker,storage_settings
        return 1 if worker(json.loads(storage_settings().read_text()))else 0
    elif args.command in ('monitor','install-monitor'):
        from .runtime.monitor import observe,install
        result=observe()if args.command=='monitor'else install()
    elif args.command=='freeze-collection':
        from .collection import freeze
        result=freeze(args.scope,args.base_job,args.wheel,args.teacher,args.version_basis)
    elif args.command=='approve-m2':
        from .collection import approve
        result=approve(args.candidates)
    elif args.command=='approve-calibforge':
        from .collection import approve
        result=approve(args.candidates,purpose='formal_calibforge')
    else:
        from .runtime.checkpoint_archive import restore,storage_settings
        settings=json.loads(storage_settings().read_text())
        result=restore(args.archive,args.destination,settings['transfer_workers'])
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
