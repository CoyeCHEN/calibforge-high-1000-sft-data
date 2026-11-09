"""The accepted native LF/Trainer/DeepSpeed loop, strict global token mean."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from studies.native_lf.token_mean import enable_native_token_mean


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume")
    args=parser.parse_args()
    spec=json.loads(Path(args.config).read_text())
    cfg=spec["LF_arguments"]
    assert spec["objective"]=="global_supervised_token_mean"
    assert int(os.environ['WORLD_SIZE'])==4
    assert cfg['learning_rate']==1e-5 and cfg['num_train_epochs']==2
    assert cfg['bf16'] and not cfg['pure_bf16'] and not cfg['enable_thinking']
    assert cfg['cutoff_len']==262144 and not cfg['packing'] and not cfg['neat_packing']
    output=Path(cfg['output_dir']);output.mkdir(exist_ok=True)
    if not args.resume and ((output/'trainer_state.json').exists() or list(output.glob('checkpoint-*'))):
        raise FileExistsError("Existing native training state; explicit validated --resume required")
    if args.resume:
        from sft.runtime.launch import validate_resume
        validate_resume(Path(args.resume),output,'ci-public-control',
                        hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
        cfg['resume_from_checkpoint']=args.resume
    os.environ['QWEN35_STAGE']='ci-public-control'
    os.environ['QWEN35_JOB_ID']=spec['job_id']
    os.environ['QWEN35_CONFIG_SHA256']=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    rank=int(os.environ['RANK'])
    with (output/f'resolved-config-rank{rank}-pid{os.getpid()}.json').open('x') as handle:
        handle.write(json.dumps(spec,indent=2)+"\n")
    from llamafactory.train.tuner import run_exp
    from sft.runtime.lf_train import EvidenceCallback, optimizer_evidence, audit_native_rng_restore
    from sft.runtime.checkpoint_observer import CheckpointObserver
    from sft.qwen35 import digest
    class CIEvidence(EvidenceCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            enable_native_token_mean(self.checkpoint_observer.trainer)
            return super().on_train_begin(args, state, control, **kwargs)

        def observe_batch(self,module,inputs,keywords):
            super().observe_batch(module,inputs,keywords)
            if not module.training or keywords.get('labels') is None:return
            keep=keywords['attention_mask'][0].bool()
            self.forward_batches[-1].update(
                input_sha256=digest(keywords['input_ids'][0][keep].detach().cpu().tolist()),
                labels_sha256=digest(keywords['labels'][0][keep].detach().cpu().tolist()))
    # CI has no eval; official Liger already skips logits for training labels.
    callback=CIEvidence(output,loss_only=False)
    callback.cutoff=cfg['cutoff_len']
    callback.runtime_recipe='ci_public_positive_control'
    callback.checkpoint_observer=CheckpointObserver(callback,optimizer_evidence,archive_enabled=False).install()
    if args.resume:audit_native_rng_restore(callback)
    run_exp(args=cfg,callbacks=[callback])
    actual=callback.checkpoint_observer.trainer.state.global_step
    assert actual==spec['expected_total_optimizer_steps']
    (output/f'PROCESS_COMPLETE-rank{rank}.json').write_text(json.dumps(dict(optimizer_steps=actual,
        objective=spec['objective'],cfg_sha256=os.environ['QWEN35_CONFIG_SHA256']))+"\n")


if __name__=='__main__':main()
