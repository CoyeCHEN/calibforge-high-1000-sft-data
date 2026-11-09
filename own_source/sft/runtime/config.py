"""Render the frozen native LF configurations; no hidden length or optimizer overrides."""
import hashlib
import json
import os
from pathlib import Path
import socket
from string import Template
import sys

import yaml

from .checkpoint_archive import atomic_json

SOURCE=Path(__file__).resolve().parents[1]/'templates'


def executable_path(runtime):
    parts=[str(Path(runtime['python']).parent)]
    if runtime.get('cuda_bin'):parts.append(runtime['cuda_bin'])
    return os.pathsep.join(parts+[os.environ.get('PATH',os.defpath)])


def initialize(root,model,hot,archive,gpu_uuids,transfer_workers=1):
    root,hot,archive=map(lambda p:Path(p).resolve(),(root,hot,archive))
    if (root/'configs/runtime.json').exists():raise FileExistsError('Existing workspace preserved')
    if len(gpu_uuids)<4 or len(set(gpu_uuids))!=len(gpu_uuids):raise ValueError('At least four unique authorized GPU UUIDs required')
    for name in ('configs','checks','manifests','logs','data','jobs','locks'):(root/name).mkdir(parents=True,exist_ok=True,mode=0o750)
    hot.mkdir(parents=True,exist_ok=True,mode=0o750);archive.mkdir(parents=True,exist_ok=True,mode=0o750)
    identity=hashlib.sha256(str(root).encode()).hexdigest()[:10]
    values=dict(WORKSPACE=str(root),MODEL_PATH=str(Path(model).resolve()),HOT_ROOT=str(hot))
    for name in ('m1-64k.yaml','m2-128k.yaml','m2-96k.yaml','ds_z3.json'):
        text=Template((SOURCE/name).read_text()).substitute(values)
        (root/'configs'/name).write_text(text)
    settings=dict(hot_root=str(hot),archive_root=str(archive/'checkpoints'),pin_root=str(hot/'archive-pins'),
        queue_root=str(hot/'archive-queue'),failed_root=str(hot/'archive-failed'),
        transfer_workers=transfer_workers,hot_device=hot.stat().st_dev,archive_device=archive.stat().st_dev,
        minimum_checkpoint_free_bytes=96*1024**3,minimum_formal_start_free_bytes=256*1024**3,
        archive_service=f'openharness-sft-{identity}-archive.service')
    for key in ('archive_root','pin_root','queue_root','failed_root'):
        Path(settings[key]).mkdir(parents=True,exist_ok=True,mode=0o750)
    atomic_json(root/'configs/checkpoint-storage.json',settings)
    atomic_json(root/'configs/runtime.json',dict(host=socket.gethostname(),python=sys.executable,model_path=values['MODEL_PATH'],
        workspace=str(root),service_prefix=f'openharness-sft-{identity}',authorized_gpu_uuids=gpu_uuids,max_concurrent_gpus=4,
        formal_training_authorized=False,formal_authorized_stage=None))
    atomic_json(root/'manifests/m2-approved-sources.json',dict(purpose='formal_m2',entries=[]))
    atomic_json(root/'configs/m2-tool-contract.json',dict(status='UNDECIDED',tools=[],schema_sha256=[],finish=False))
    return dict(workspace=str(root),formal_training_authorized=False)


def validate_native(cfg, *, bounded=False):
    recipe=cfg.get('runtime_recipe','agent_sft_v1')
    if recipe=='qwen35_772_b32_e5':
        required=dict(model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
            stage='sft',finetuning_type='full',bf16=True,pure_bf16=False,fp16=False,
            freeze_vision_tower=True,freeze_multi_modal_projector=True,freeze_language_model=False,
            template='qwen3_5',enable_thinking=True,preserve_thinking=True,train_on_prompt=False,
            mask_history=False,packing=False,neat_packing=False,save_only_model=False,
            optim='adamw_torch',learning_rate=1e-5,adam_beta1=0.9,adam_beta2=0.999,
            adam_epsilon=1e-8,weight_decay=0.0,max_grad_norm=1.0,
            lr_scheduler_type='cosine_with_min_lr',lr_scheduler_kwargs=dict(min_lr_rate=0.1),
            warmup_ratio=0.05,num_train_epochs=5.0,per_device_train_batch_size=1,
            gradient_checkpointing=True,use_reentrant_gc=False,flash_attn='fa2',
            enable_liger_kernel=True,seed=42,data_seed=42,dataloader_num_workers=0)
        if any(cfg.get(k)!=v for k,v in required.items()):
            raise ValueError('Frozen 772 / batch32 / epoch5 recipe changed')
        if cfg['gradient_accumulation_steps']not in(4,8)or cfg['cutoff_len']not in(131072,262144):
            raise ValueError('Only accepted pure DP4/8 and 128K/256K candidates are allowed')
        if bounded:
            if any(cfg.get(k)!=v for k,v in dict(max_steps=3,save_steps=1,do_eval=False,
                eval_strategy='no',save_strategy='steps',eval_on_start=False).items()):
                raise ValueError('Capacity acceptance must update/save/resume the three-step diagnostic only')
        elif cfg.get('max_steps',-1)>0 or cfg.get('save_strategy')!='epoch' or cfg.get('eval_strategy')!='epoch':
            raise ValueError('Formal training must retain its complete five-epoch schedule and checkpoints')
        if json.loads(Path(cfg['deepspeed']).read_text())!=json.loads((SOURCE/'ds_z3.json').read_text()):
            raise ValueError('Frozen native ZeRO3 recipe changed')
        return cfg
    if recipe not in ('agent_sft_v1','calibforge_appendix_e'):
        raise ValueError('Unknown frozen training recipe')
    appendix_e=recipe=='calibforge_appendix_e'
    if cfg['model_revision']!='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a':raise ValueError('Model revision changed')
    required=dict(stage='sft',finetuning_type='full',bf16=True,pure_bf16=False,freeze_vision_tower=True,
        freeze_multi_modal_projector=True,freeze_language_model=False,template='qwen3_5',enable_thinking=True,
        preserve_thinking=True,train_on_prompt=False,packing=False,neat_packing=False,save_only_model=False,
        dataloader_num_workers=0,optim='adamw_torch',fp16=False,
        adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,weight_decay=0.0 if appendix_e else 0.01,
        max_grad_norm=1.0,lr_scheduler_type='cosine',warmup_ratio=0.05 if appendix_e else 0.03,
        per_device_train_batch_size=1,gradient_accumulation_steps=32 if appendix_e else 4,
        gradient_checkpointing=True,use_reentrant_gc=False,flash_attn='fa2',enable_liger_kernel=True,seed=42,data_seed=42)
    if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError('Frozen native training contract changed')
    if cfg['num_train_epochs']!=(10 if appendix_e else 1):
        raise ValueError('Formal stage requires the frozen epoch budget, no bounded stop')
    if bounded:
        if cfg.get('max_steps')!=3 or cfg.get('save_steps')!=1 or cfg.get('do_eval')is not False:
            raise ValueError('Bounded acceptance requires exactly three scheduled steps and save every step')
    elif cfg.get('max_steps',-1)>0:raise ValueError('Formal stage requires the frozen epoch budget, no bounded stop')
    length=cfg['cutoff_len']
    if length not in ((131072,)if appendix_e else(65536,98304,131072)):raise ValueError('Frozen length changed')
    expected_lr=1e-5 if appendix_e else(2e-6 if length==65536 else 5e-7)
    if cfg['learning_rate']!=expected_lr:raise ValueError('Frozen learning rate changed')
    if cfg['mask_history']!=(False if appendix_e else length!=65536):raise ValueError('Frozen assistant supervision changed')
    if appendix_e:
        if cfg.get('lr_scheduler_kwargs',{})!={}:raise ValueError('Unverified minimum-LR override')
        if not bounded and any(cfg.get(k)!=v for k,v in dict(do_train=True,do_eval=True,
            save_strategy='epoch',eval_strategy='epoch',save_total_limit=None,load_best_model_at_end=False,
            eval_on_start=True).items()):
            raise ValueError('Every epoch checkpoint and validation must be preserved; select final epoch')
        if bounded and(cfg.get('save_strategy')!='steps'or cfg.get('eval_strategy')!='no'or cfg.get('eval_on_start',False)):
            raise ValueError('Bounded acceptance must use step saves and no validation')
    if json.loads(Path(cfg['deepspeed']).read_text())!=json.loads((SOURCE/'ds_z3.json').read_text()):
        raise ValueError('Frozen ZeRO-3/BF16/offload configuration changed')
    return cfg


def native_arguments(cfg):
    """Runtime recipe identity is wrapper metadata, never an ignored LF argument."""
    return {k:v for k,v in cfg.items()if k!='runtime_recipe'}
