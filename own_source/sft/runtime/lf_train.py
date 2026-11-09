"""LF's native run_exp with evidence callbacks and its official Liger loss-only API."""
import argparse
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import random
from pathlib import Path
import time
import yaml
import torch
import numpy as np
from transformers import Trainer,TrainerCallback
from llamafactory.train.tuner import run_exp
from sft.qwen35 import digest
from .checkpoint_observer import CheckpointObserver
from .paths import workspace
from .identity import training_source_sha256
from .batch_contract import measure_data_parallel


def optimizer_evidence(optimizer):
    raw=optimizer;seen=set()
    masters={}
    probe=optimizer
    while probe is not None and id(probe)not in masters:
        masters[id(probe)]=probe
        probe=getattr(probe,'optimizer',None)
    partitions={id(t):t for obj in masters.values()
        for t in getattr(obj,'fp32_partitioned_groups_flat',[]) if torch.is_tensor(t)}
    while getattr(raw,'optimizer',None)is not None and id(raw)not in seen:seen.add(id(raw));raw=raw.optimizer
    sha=hashlib.sha256();dtypes=set();moments=set();states=0;state_bytes=moment_bytes=0
    # ZeRO-3 clears param_groups[*]['params'] between updates; its actual
    # Adam state remains in optimizer.state, keyed by FP32 partitions.
    for parameter_state in getattr(raw,'state',{}).values():
        for key,value in sorted(parameter_state.items()):
            sha.update(key.encode())
            if torch.is_tensor(value):
                states+=1;dtypes.add(str(value.dtype));flat=value.detach().reshape(-1)
                state_bytes+=value.numel()*value.element_size()
                if key in ('exp_avg','exp_avg_sq'):moment_bytes+=value.numel()*value.element_size()
                if key in ('exp_avg','exp_avg_sq'):moments.add(str(value.dtype))
                sample=torch.cat([flat[:16],flat[-16:]]).float().cpu()
                sha.update(str((list(value.shape),str(value.dtype))).encode());sha.update(sample.numpy().tobytes())
            else:sha.update(json.dumps(value,sort_keys=True).encode())
    return dict(optimizer_state_sample_sha256=sha.hexdigest()if states else None,optimizer_state_dtypes=sorted(dtypes),optimizer_moment_dtypes=sorted(moments),optimizer_state_tensors=states,
        optimizer_state_bytes=state_bytes,optimizer_moment_bytes=moment_bytes,
        fp32_master_partition_bytes=sum(t.numel()*t.element_size()for t in partitions.values()),
        master_partition_dtypes=sorted({str(t.dtype)for t in partitions.values()}))


class EvidenceCallback(TrainerCallback):
    def __init__(self, output, loss_only):
        self.output=Path(output);self.loss_only=loss_only;self.started=None;self.selected=[];self.initial={};self.gradients={}
        self.supervised_tokens=0;self.input_tokens=0
        self.forward_batches=[]
        self.checkpoint_observer=None
        self.step_learning_rates=[];self.parameter_updates=0
        self.runtime_recipe='agent_sft_v1'
        self.bounded_acceptance=False

    def observe_batch(self,module,inputs,keywords):
        if keywords.get('labels')is None:return
        labels=keywords['labels'];loss_tokens=int((labels!=-100).sum());mask=keywords.get('attention_mask')
        if loss_tokens==0:raise RuntimeError('All assistant labels are masked')
        if labels.shape[-1]>self.cutoff:raise RuntimeError('Model input exceeds the declared cutoff; no truncation allowed')
        if not module.training:return
        input_tokens=int(mask.sum())if mask is not None else int(labels.numel())
        self.supervised_tokens+=loss_tokens;self.input_tokens+=input_tokens
        observed=dict(sequence_tokens=labels.shape[-1],nonpadding_tokens=input_tokens,supervised_tokens=loss_tokens)
        if self.runtime_recipe=='qwen35_772_b32_e5' or(self.bounded_acceptance and self.runtime_recipe=='calibforge_appendix_e'):
            if labels.shape[0]!=1 or mask is None or mask.ndim!=2:
                raise RuntimeError('CalibForge canary signature requires native MB1 / unpacked 2D mask')
            keep=mask[0].bool()
            observed.update(input_sha256=digest(keywords['input_ids'][0][keep].detach().cpu().tolist()),
                            labels_sha256=digest(labels[0][keep].detach().cpu().tolist()))
            # Diagnostic identity is removed before model.forward. It never
            # changes a token or becomes teacher/student conditioning.
            if self.bounded_acceptance:
                observed['canary_sample_index']=int(keywords.pop('_calibforge_sample_index').item())
        self.forward_batches.append(observed)

    def record(self,event,state,**data):
        rank=int(os.environ.get('RANK',0))
        entry=dict(event=event,global_step=state.global_step,rank=rank,time=time.time(),
                   pid=os.getpid(),job_id=os.environ.get('QWEN35_JOB_ID'),**data)
        if self.checkpoint_observer and 'identity'not in entry:
            entry['identity']=self.checkpoint_observer.identity()
        try:
            self.output.mkdir(parents=True,exist_ok=True)
            with (self.output/f'preflight-evidence-rank{rank}.jsonl').open('a')as f:
                f.write(json.dumps(entry)+'\n')
                if event in ('optimizer_step','checkpoint_save_complete','train_end'):
                    f.flush();os.fsync(f.fileno())
        except OSError:
            logging.exception('Non-critical evidence write failed; native training continues')

    def weight(self,param):
        if hasattr(param,'ds_id'):
            from deepspeed.utils import safe_get_full_fp32_param
            value=safe_get_full_fp32_param(param)
        else:value=param
        if value is None:raise RuntimeError('Cannot inspect FP32 master parameter')
        if value.dtype!=torch.float32:raise RuntimeError('Expected real FP32 master parameter')
        return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()

    def on_train_begin(self,args,state,control,model=None,optimizer=None,lr_scheduler=None,**kwargs):
        self.started=time.monotonic()
        parallelism=measure_data_parallel(self.checkpoint_observer.trainer.model_wrapped,torch.distributed)
        if self.runtime_recipe=='qwen35_772_b32_e5' and (
            parallelism['data_parallel_group_size']not in(4,8)
            or parallelism['world_size']!=parallelism['data_parallel_group_size']
            or parallelism['effective_global_batch']!=32
            or parallelism['microbatch_per_gpu']!=1
            or parallelism['gradient_accumulation_steps']!=32//parallelism['data_parallel_group_size']):
            raise RuntimeError('Actual pure DP / accumulation does not implement global batch32')
        if self.runtime_recipe=='calibforge_appendix_e' and (
            parallelism['data_parallel_group_size']!=4 or parallelism['effective_global_batch']!=128
            or parallelism['microbatch_per_gpu']!=1 or parallelism['gradient_accumulation_steps']!=32):
            raise RuntimeError('CalibForge candidate requires actual DP4 / MB1 / GA32 / global128')
        model.register_forward_pre_hook(self.observe_batch,with_kwargs=True)
        if self.loss_only:
            if 'skip_logits' not in inspect.signature(model.forward).parameters:
                raise RuntimeError('Pinned official Liger loss-only forward API unavailable')
            # Official public argument; preserve LF/Transformers kernels and loss shifting.
            # Validation also needs this: Liger defaults to full logits in model.eval().
            model.forward=functools.partial(model.forward,skip_logits=True)
        n=lambda p:getattr(p,'ds_numel',p.numel())
        trainable=sum(n(p)for name,p in model.named_parameters()if p.requires_grad)
        visual_trainable=sum(n(p)for name,p in model.named_parameters()if 'visual' in name and p.requires_grad)
        language_frozen=sum(n(p)for name,p in model.named_parameters()if 'visual'not in name and not p.requires_grad)
        if trainable!=4205751296 or visual_trainable or language_frozen:
            raise RuntimeError(f'Unexpected trainable range: {trainable}, visual={visual_trainable}, frozen_language={language_frozen}')
        norm=next((x for x in model.named_parameters()if x[0].endswith('input_layernorm.weight')and x[1].requires_grad),None)
        delta=next((x for x in model.named_parameters()if 'linear_attn' in x[0]and x[1].requires_grad and n(x[1])<=2560),None)
        self.selected=[x for x in (norm,delta)if x is not None]
        if len(self.selected)!=2:raise RuntimeError('Missing language/Gated DeltaNet evidence parameters')
        self.initial={name:self.weight(p)for name,p in self.selected}
        for name,p in self.selected:
            def capture(gradient,name=name):
                self.gradients[name]=dict(norm=float(gradient.float().norm()),finite=bool(torch.isfinite(gradient).all()),dtype=str(gradient.dtype))
                return gradient
            p.register_hook(capture)
        self.record('train_begin',state,trainable_language_parameters=trainable,visual_trainable=visual_trainable,
                    language_frozen=language_frozen,initial_weight_hashes=self.initial,
                    parameter_dtypes=sorted({str(p.dtype)for p in model.parameters()if p.requires_grad}),
                    bf16=args.bf16,pure_bf16=False,loss_only_official_liger_api=self.loss_only,
                    runtime_recipe=self.runtime_recipe,parallelism=parallelism,
                    bounded_acceptance=self.bounded_acceptance,
                    ignore_data_skip=args.ignore_data_skip,
                    training_source_sha256=training_source_sha256(),
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.learning_rate,warmup_ratio=args.warmup_ratio,cutoff_len=self.cutoff,
                    cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),
                    scheduler_state=lr_scheduler.state_dict()if lr_scheduler else None,**optimizer_evidence(optimizer))

    def on_step_begin(self,args,state,control,optimizer=None,**kwargs):
        # Capture BEFORE backward: Accelerate's DeepSpeed wrapper can execute
        # engine.step() inside backward, before Trainer's pre-optimizer event.
        self.step_learning_rates=[float(group['lr'])for group in optimizer.param_groups]

    def on_step_end(self,args,state,control,optimizer=None,lr_scheduler=None,**kwargs):
        if len(self.gradients)!=2 or any(not g['finite']for g in self.gradients.values()):raise RuntimeError('Missing/nonfinite language gradient')
        now={name:self.weight(p)for name,p in self.selected}
        changed={name:now[name]!=self.initial[name]for name in now}
        if not self.step_learning_rates:raise RuntimeError('Missing native optimizer learning-rate evidence')
        updated=any(changed.values())
        # These are two sampled tensors, not the whole optimizer. A small
        # nonzero cosine-tail update can round away in FP32. Keep this as
        # evidence; bounded acceptance separately requires observed updates.
        self.parameter_updates+=int(updated)
        self.record('optimizer_step',state,weight_hashes=now,changed=changed,gradients=self.gradients,
                    optimizer_step_learning_rates=self.step_learning_rates,parameter_update_observed=updated,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    elapsed_seconds=time.monotonic()-self.started,
                    supervised_tokens_since_begin=self.supervised_tokens,input_tokens_since_begin=self.input_tokens,
                    supervised_tokens_per_second=self.supervised_tokens/(time.monotonic()-self.started),
                    model_input_batches=self.forward_batches,batch_contract_verified=bool(self.forward_batches),
                    scheduler_state=lr_scheduler.state_dict()if lr_scheduler else None,**optimizer_evidence(optimizer))
        self.initial=now
        self.gradients={}
        self.forward_batches=[]

    def on_log(self,args,state,control,logs=None,**kwargs):
        self.record('metrics',state,metrics=logs or{})
        for k in ('loss','eval_loss','grad_norm'):
            if k in (logs or{})and not math.isfinite(float(logs[k])):raise RuntimeError('Nonfinite '+k)
        if self.runtime_recipe=='qwen35_772_b32_e5':
            for key,value in(logs or{}).items():
                if key.endswith('_loss')and not math.isfinite(float(value)):raise RuntimeError('Nonfinite '+key)
        if self.runtime_recipe=='calibforge_appendix_e'and'eval_loss'in(logs or{}):
            from .training_health import calibforge_validation_health
            health=calibforge_validation_health(state.log_history)
            self.record('validation_health',state,**health)
            if health['severe_validation_regression']:
                # Native torchrun tears down the affected ranks together. Keep
                # the last complete checkpoint; never export this as epoch 10.
                raise RuntimeError('Predeclared severe validation regression; inspect the last complete epoch checkpoint')

    def on_save(self,args,state,control,**kwargs):
        folder=self.output/f'checkpoint-{state.global_step}'
        self.record('checkpoint_saved',state,path=str(folder),files=[str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()])

    def on_train_end(self,args,state,control,**kwargs):
        self.record('train_end',state,elapsed_seconds=time.monotonic()-self.started,
                    sampled_parameter_updates_observed=self.parameter_updates,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    num_input_tokens_seen=getattr(state,'num_input_tokens_seen',None))

    def on_evaluate(self,args,state,control,**kwargs):
        if self.runtime_recipe!='qwen35_772_b32_e5' or self.bounded_acceptance:return
        from .paths import runtime_config
        runtime=runtime_config();path=runtime.get('cf_retention_cache')
        if not path or os.environ.get('QWEN35_STAGE')!='new-cf416-tmax356-m2' or getattr(self,'retention_running',False):return
        from datasets import load_from_disk
        self.retention_running=True
        try:
            self.checkpoint_observer.trainer.evaluate(eval_dataset=load_from_disk(path)['validation'],
                metric_key_prefix='cf_validation')
        finally:self.retention_running=False


def audit_native_rng_restore(callback):
    """Observe the pinned Trainer's own restore, then compare actual RNG values."""
    native=Trainer._load_rng_state
    def checked(trainer,checkpoint):
        native(trainer,checkpoint)
        if checkpoint is None:return
        rank=trainer.args.process_index
        path=Path(checkpoint)/f'rng_state_{rank}.pth'
        saved=torch.load(path,map_location='cpu',weights_only=False)  # Our own checkpoint, never an external pickle.
        live=np.random.get_state();expected=saved['numpy']
        cuda_live=torch.cuda.random.get_rng_state_all()
        matches=dict(python=random.getstate()==saved['python'],
            numpy=live[0]==expected[0]and np.array_equal(live[1],expected[1])and live[2:]==expected[2:],
            torch_cpu=torch.equal(torch.random.get_rng_state(),saved['cpu']),
            torch_cuda=len(cuda_live)==len(saved['cuda'])and all(torch.equal(a,b)for a,b in zip(cuda_live,saved['cuda'])))
        if not all(matches.values()):raise RuntimeError('Native RNG checkpoint restore mismatch: '+str(matches))
        callback.record('rng_restored',trainer.state,matches=matches,checkpoint_rng_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    Trainer._load_rng_state=checked


class StopAfterCheckpoint(TrainerCallback):
    """Bounded acceptance only: stop after step 2 without changing the step-3 scheduler."""
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==2:
            control.should_save=True
            control.should_training_stop=True
        return control


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Native LF training; explicit bounded acceptance uses the same entry.')
    parser.add_argument('config');parser.add_argument('--resume');parser.add_argument('--archive-checkpoints',action='store_true',required=True)
    parser.add_argument('--bounded-acceptance',action='store_true')
    parser.add_argument('--stop-after-checkpoint-2',action='store_true')
    cli=parser.parse_args();config=yaml.safe_load(Path(cli.config).read_text())
    from .config import validate_native,native_arguments
    validate_native(config,bounded=cli.bounded_acceptance)
    if cli.bounded_acceptance and not os.environ.get('QWEN35_STAGE','').startswith('verify-'):
        raise ValueError('Bounded acceptance must use a separate verify stage/output')
    if cli.stop_after_checkpoint_2 and (not cli.bounded_acceptance or cli.resume):
        raise ValueError('Checkpoint-2 stop is only for the first bounded invocation')
    if not cli.bounded_acceptance:
        from .paths import runtime_config,config_path
        from .checkpoint_archive import sha256
        runtime=runtime_config();stage=os.environ.get('QWEN35_STAGE')
        if not runtime['formal_training_authorized']or runtime['formal_authorized_stage']!=stage:
            raise ValueError('Explicit formal-stage authorization required; use the remote job entry')
        if Path(cli.config).resolve()!=config_path(stage).resolve()or sha256(cli.config)!=os.environ.get('QWEN35_CONFIG_SHA256'):
            raise ValueError('Formal resolved configuration differs from submitted job')
    if cli.resume:config['resume_from_checkpoint']=cli.resume
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',0));job_id=os.environ['QWEN35_JOB_ID']
    (output/f'resolved-config-{job_id}-rank{rank}.json').write_text(json.dumps(config,indent=2))
    callback=EvidenceCallback(output,config.get('enable_liger_kernel',False));callback.cutoff=config['cutoff_len']
    callback.runtime_recipe=config.get('runtime_recipe','agent_sft_v1')
    callback.bounded_acceptance=cli.bounded_acceptance
    if callback.runtime_recipe=='qwen35_772_b32_e5':
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        from .eval_gather import preserve_loss_only_gather
        CustomSeq2SeqTrainer.evaluation_loop=preserve_loss_only_gather(CustomSeq2SeqTrainer.evaluation_loop)
    callback.checkpoint_observer=CheckpointObserver(callback,optimizer_evidence,True).install()
    if cli.resume:audit_native_rng_restore(callback)
    callbacks=[callback]
    if cli.stop_after_checkpoint_2:callbacks.append(StopAfterCheckpoint())
    run_exp(args=native_arguments(config),callbacks=callbacks)
