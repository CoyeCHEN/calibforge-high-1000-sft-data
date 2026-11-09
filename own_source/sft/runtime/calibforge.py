"""CalibForge A/B contracts around the existing native Trainer and remote jobs.

No dispatch, retries, serving control, or training loop lives here.
"""
import json
from pathlib import Path
import os
import shutil
import subprocess

import yaml

from .checkpoint_archive import sha256
from .config import validate_native
from .paths import CALIBFORGE_STAGES,config_path,runtime_config,workspace,data_manifest_path
from .. import qwen35 as codec

ROUTES={'m3-a-128k':'M3-A','c1-b-128k':'C1-B'}


def checked_json(path,expected):
    if sha256(path)!=expected:raise ValueError('Frozen CalibForge evidence SHA mismatch')
    return json.loads(Path(path).read_text())


def validate_pair():
    configs={stage:validate_native(yaml.safe_load(config_path(stage).read_text()))
             for stage in CALIBFORGE_STAGES}
    if any(cfg.get('runtime_recipe')!='calibforge_appendix_e'for cfg in configs.values()):
        raise ValueError('CalibForge stages require the Appendix E recipe')
    compared=[{k:v for k,v in cfg.items()if k not in ('model_name_or_path','output_dir')}
              for cfg in configs.values()]
    if compared[0]!=compared[1]:raise ValueError('A/B data order or training configuration differs')
    outputs=[Path(cfg['output_dir']).resolve()for cfg in configs.values()]
    if outputs[0]==outputs[1]:raise ValueError('A/B output directories must be distinct')
    hot=Path(runtime_config()['hot_root']).resolve()
    if any(not path.is_relative_to(hot)or path==hot for path in outputs):
        raise ValueError('CalibForge output must remain in its own hot storage')
    if any(cfg.get('resume_from_checkpoint')for cfg in configs.values()):
        raise ValueError('Stage initialization uses new optimizer; same-stage resume is explicit CLI only')
    return configs


def validate_dataset(cfg,manifest):
    runtime=runtime_config()
    expected=dict(kind='CalibForge_public_teacher_full_multiturn_sft',cutoff=131072,
        text_budget=131006,native_collator_reserved_tokens=66,supervision_scope='all_assistant_once',
        original_provider_prefixes_verified=True,model_revision=codec.REVISION,template_sha256=codec.TEMPLATE_SHA)
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('Shared CalibForge encoding/supervision contract differs')
    if Path(manifest['path']).resolve()!=Path(cfg['tokenized_path']).resolve():
        raise ValueError('CalibForge native cache path differs')
    if manifest.get('codec_sha256')!=sha256(Path(codec.__file__)):
        raise ValueError('CalibForge codec changed after data encoding')
    from ..prepare import read_approval
    checked_json(manifest['source_manifest_path'],manifest['source_manifest_sha256'])
    read_approval(manifest['source_manifest_path'],purpose='formal_calibforge')
    contract=json.loads(Path(runtime['tool_contract']).read_text())
    if (manifest['tool_contract_sha256']!=sha256(runtime['tool_contract'])
        or contract.get('status')!='APPROVED'or contract.get('finish')is not False):
        raise ValueError('Shared CalibForge tool contract changed')
    for split in ('train','validation'):
        index=manifest['row_indexes'][split]
        if Path(index['path']).stat().st_size!=index['bytes']or sha256(index['path'])!=index['sha256']:
            raise ValueError('CalibForge row/source mapping index changed')


def validate_initialization(stage,cfg,*,verify_weights=False):
    runtime=runtime_config()
    identity=checked_json(runtime['initial_models_manifest'],runtime['initial_models_manifest_sha256'])
    if identity.get('base_model')!='Qwen/Qwen3.5-4B'or identity.get('revision')!=codec.REVISION:
        raise ValueError('Wrong CalibForge base model identity')
    route=identity['routes'][ROUTES[stage]];model=Path(route['initial_model_path']).resolve()
    if Path(cfg['model_name_or_path']).resolve()!=model:
        raise ValueError('CalibForge stage initialized from the wrong model')
    if route.get('initial_optimizer')!='NEW'or route.get('initial_scheduler')!='NEW':
        raise ValueError('Cross-stage optimizer/scheduler reuse is forbidden')
    proof=checked_json(route['source_verification'],route['source_verification_sha256'])
    if stage=='m3-a-128k':
        if (proof.get('status')!='M2_COMPLETED_HF_VERIFIED'
            or not proof.get('hf_export',{}).get('complete_final_hf')
            or not proof.get('hf_load_smoke',{}).get('model_loaded')
            or proof['hf_load_smoke'].get('generated_tokens',0)<1):
            raise ValueError('M3-A requires the verified final M2 HF model')
        weights=[item for item in proof['hf_export']['files']if item['path'].endswith('.safetensors')]
        if len(weights)!=1 or weights[0]['sha256']!=route['weight_sha256']:
            raise ValueError('M3-A M2 weight identity mismatch')
    else:
        if (proof.get('stage')!='M0_UNTUNED_OFFICIAL'or not proof.get('original_official_shards_match')
            or proof.get('base_revision')!=codec.REVISION or proof.get('weight_sha256')!=route['weight_sha256']):
            raise ValueError('C1-B requires the original official M0 model')
        official=json.loads((Path(__file__).parents[1]/'model-sha256.json').read_text())
        expected={item['file']:(item['sha256'],item['bytes'])for item in official}
        actual={item['path']:(item['sha256'],item['bytes'])for item in route['expected_files']if item['path']in expected}
        if actual!=expected:raise ValueError('C1-B official shard identity mismatch')
    # Prior immutable full-file hashes are reusable while all recorded sizes and
    # mtimes match. The actual launch rehashes once at the model handoff.
    if not route['expected_files']:raise ValueError('No frozen initialization files')
    for item in route['expected_files']:
        path=model/item['path'];stat=path.stat()
        if path.is_symlink()or stat.st_size!=item['bytes']or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Initialization file changed: '+item['path'])
        if verify_weights and sha256(path)!=item['sha256']:
            raise ValueError('Initialization SHA mismatch: '+item['path'])
    return route


def validate_predecessor():
    runtime=runtime_config()
    report=checked_json(runtime['prior_campaign_report'],runtime['prior_campaign_report_sha256'])
    result=validate_predecessor_report(report)
    return result|dict(report_sha256=sha256(runtime['prior_campaign_report']))


def validate_predecessor_report(report):
    """Validate the same frozen report before a remote evidence handoff is admitted."""
    models=report['models'];seen=set();task_sets=[]
    task_ids=set(json.loads((Path(__file__).parents[1]/'policies/evaluation-exclusion.json').read_text())['instruction_normalized_sha256'])
    if (set(models)!={'m0','m1','m2'}or report.get('scope')!='PREDECLARED_FIRST_89_PER_MODEL'
        or report.get('planned_primary_trials')!=267 or report.get('completed_primary_trials')!=267
        or report.get('primary_complete')is not True):
        raise ValueError('M0/M1/M2 primary first-round campaign is not sealed')
    for name in ('m0','m1','m2'):
        model=models[name];rows=model['trials'];tasks=[r['task_id']for r in rows];task_sets.append(set(tasks))
        if (model.get('primary')is not True or set(tasks)!=task_ids or len(tasks)!=89
            or model['terminal']!=89 or '-'+name+'-r1-'not in model['job_name']):
            raise ValueError('Prior campaign task quota incomplete')
        if len(set(tasks))!=89:raise ValueError('Duplicate prior task identity')
        for row in rows:
            ident=row['trial_id']
            if not ident or ident in seen:
                raise ValueError('Prior campaign Trial identity/quota mismatch')
            if (len(row.get('result_sha256',''))!=64 or not row.get('result_path')
                or not row.get('trial_name')):raise ValueError('Prior raw result identity is missing')
            seen.add(ident)
    if not task_sets[0]==task_sets[1]==task_sets[2]or len(seen)!=267:
        raise ValueError('Prior campaign task scope differs')
    return dict(terminal_trials=len(seen))


def validate_stage_order(stage):
    if stage=='c1-b-128k':
        path=Path(yaml.safe_load(config_path('m3-a-128k').read_text())['output_dir'])
        state=json.loads((path/'trainer_state.json').read_text())
        from .hf_export import final_verified
        if (state.get('epoch',0)<10-1e-6 or state.get('max_steps',0)<1
            or state.get('global_step')!=state['max_steps']or not final_verified(path)):
            raise ValueError('C1-B waits for final epoch-10 M3-A HF validation')


def canary_input_contract(path):
    """Bind observed GPU tensors to the actual corpus extremes, not padding."""
    proof=json.loads(Path(path).read_text());dataset_path=data_manifest_path('m3-a-128k')
    manifest=json.loads(dataset_path.read_text())
    if (proof.get('kind')!='calibforge_frozen_data_gpu_canary'
        or proof.get('source_dataset_manifest_sha256')!=sha256(dataset_path)
        or proof.get('diagnostic_exposures')!=384 or not manifest.get('formal_training_eligible')):
        raise ValueError('GPU canary is not bound to the frozen approved CalibForge corpus')
    selected={(row['source_split'],row['index_in_split']):row for row in proof['selected_rows']}
    if not 1<=len(selected)<=2 or len(selected)!=len(proof['selected_rows']):
        raise ValueError('Select the actual longest-input and largest-supervision corpus rows')
    found={};max_input=0;max_loss=0
    for split in ('train','validation'):
        source=manifest['row_indexes'][split]
        if sha256(source['path'])!=source['sha256']:raise ValueError('Frozen corpus row index changed')
        count=0
        with Path(source['path']).open()as stream:
            for index,line in enumerate(stream):
                row=json.loads(line);count+=1
                max_input=max(max_input,row['text_tokens']);max_loss=max(max_loss,row['loss_tokens'])
                declared=selected.get((split,index))
                if declared is not None:
                    fields=('input_sha256','labels_sha256','text_tokens','loss_tokens')
                    if any(declared[key]!=row[key]for key in fields):raise ValueError('Canary selected row differs from source index')
                    found[(row['input_sha256'],row['labels_sha256'])]={key:row[key]for key in fields}
        if count!=manifest['counts'][split]['rows']:raise ValueError('Frozen corpus index count differs')
    if (len(found)!=len(selected)or max(row['text_tokens']for row in found.values())!=max_input
        or max(row['loss_tokens']for row in found.values())!=max_loss):
        raise ValueError('Canary omits real longest-input or largest-supervision source data')
    return dict(allowed=found,required=set(found),maximum_real_input_tokens=max_input,
                maximum_real_supervised_tokens=max_loss,diagnostic_exposures=384,
                source_dataset_manifest_sha256=sha256(dataset_path))


def check(*,native=True):
    from .readiness import dependencies_verified,model_verified,VERSIONS
    from .launch import validate_data,gpu_check
    from .checkpoint_archive import atomic_json,storage_settings
    from .identity import training_source_sha256
    from .new_entry_acceptance import new_entry_acceptance
    from .config import native_arguments
    from .paths import data_manifest_path
    root=workspace();runtime=runtime_config();checks={};details={}
    def probe(name,operation):
        try:
            value=operation();checks[name]=value is not False
            return value
        except (OSError,ValueError,KeyError,ImportError,TypeError,subprocess.SubprocessError)as error:
            checks[name]=False;details[name]=type(error).__name__+': '+str(error)
    configs=probe('shared_recipe',validate_pair)or{}
    for stage,cfg in configs.items():
        if native:
            def parse(cfg=cfg):
                from llamafactory.hparams import get_train_args
                get_train_args(native_arguments(cfg)|dict(use_cpu=True,deepspeed=None))
            probe(stage+'_native_config',parse)
        else:checks[stage+'_native_config']=False;details[stage+'_native_config']='Native argument parsing deliberately not executed'
        probe(stage+'_initialization',lambda stage=stage,cfg=cfg:validate_initialization(stage,cfg))
        probe(stage+'_order',lambda stage=stage:validate_stage_order(stage))
    def dataset():
        manifest=json.loads(data_manifest_path(CALIBFORGE_STAGES[0]).read_text())
        cfg=configs[CALIBFORGE_STAGES[0]]
        validate_data(cfg['tokenized_path'],manifest);validate_dataset(cfg,manifest)
    probe('shared_data',dataset);probe('prior_campaign_complete',validate_predecessor)
    probe('dependencies',dependencies_verified);probe('model_identity',model_verified)
    def tokenizer():
        codec.MODEL=runtime['model_path'];codec.load_tokenizer()
    probe('tokenizer',tokenizer)
    def storage():
        settings=json.loads(storage_settings().read_text());hot=Path(settings['hot_root'])
        return hot.stat().st_dev==settings['hot_device']and os.access(hot,os.W_OK)and shutil.disk_usage(hot).free>=settings['minimum_formal_start_free_bytes']
    probe('storage',storage)
    probe('gpu_available',lambda:gpu_check(runtime.get('planned_gpu_ids',['0','1','2','3']),runtime))
    cfg=configs.get(CALIBFORGE_STAGES[0])
    gpu=dict(passed=False,reason='Missing current-corpus GPU evidence')
    if cfg and runtime.get('calibforge_gpu_evidence'):
        try:
            directory=Path(runtime['calibforge_gpu_evidence'])
            inputs=canary_input_contract(directory/'GPU_CANARY_INPUTS.json')
            gpu=new_entry_acceptance(directory,131072,configuration=cfg,input_contract=inputs)
            gpu.update(maximum_real_input_tokens=inputs['maximum_real_input_tokens'],
                maximum_real_supervised_tokens=inputs['maximum_real_supervised_tokens'])
        except (OSError,ValueError,KeyError,TypeError)as error:gpu=dict(passed=False,reason=str(error))
    checks['current_recipe_gpu_update_resume']=gpu['passed']
    common=('shared_recipe','shared_data','prior_campaign_complete','dependencies','model_identity','tokenizer','storage','gpu_available','current_recipe_gpu_update_resume')
    stages={}
    for stage in CALIBFORGE_STAGES:
        cfg=configs.get(stage);path=config_path(stage)
        names=common+(stage+'_native_config',stage+'_initialization',stage+'_order')
        item=dict(ready_to_train=all(checks.get(name,False)for name in names),
            blockers=[name for name in names if not checks.get(name,False)],
            config_sha256=sha256(path)if path.exists()else None,
            versions_sha256=sha256(VERSIONS),training_source_sha256=training_source_sha256(),
            deepspeed_sha256=sha256(cfg['deepspeed'])if cfg else None,
            data_manifest_sha256=sha256(data_manifest_path(stage))if data_manifest_path(stage).exists()else None)
        for field in ('initial_models_manifest','tool_contract','prior_campaign_report'):
            target=runtime.get(field)
            item[field+'_sha256']=sha256(target)if target and Path(target).is_file()else None
        stages[stage]=item
    started=[]
    for stage,cfg in configs.items():
        record=Path(cfg['output_dir'])/'preflight-evidence-rank0.jsonl'
        if record.is_file():
            with record.open()as stream:
                for line in stream:
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get('event')=='train_begin'and event.get('runtime_recipe')=='calibforge_appendix_e':
                        started.append(stage);break
    result=dict(runtime_recipe='calibforge_appendix_e',checks=checks,details=details,stages=stages,
        new_entry_gpu_evidence=gpu,formal_training_started=bool(started),started_stages=started,
        evidence_scope='Actual CPU/source/config checks plus current recipe GPU evidence; missing artifacts remain false')
    atomic_json(root/'checks/readiness.json',result)
    return result
