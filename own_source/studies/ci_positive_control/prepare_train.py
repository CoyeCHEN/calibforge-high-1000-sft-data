"""Concrete two-epoch native LF recipe; no terminal training is submitted."""
import hashlib
import json
import math
from pathlib import Path
import sys


def main(root):
    root = Path(root)
    encoding=json.loads((root/'checks/encoding-r2.json').read_text())
    rows=encoding['train']['training_rows']
    local_rows=math.ceil(rows/4)
    steps=math.ceil(local_rows/8)
    padding=4*local_rows-rows
    job_id=f'ci-e2b-{rows}-e2-{root.name}'
    (root/"runs").mkdir(exist_ok=True)
    ds=root/"configs/ds_z3.json"
    original=Path(__file__).resolve().parents[2]/"sft/templates/ds_z3.json"
    ds.write_bytes(original.read_bytes())
    cfg=dict(model_name_or_path="/data/models/Qwen/Qwen3.5-4B",
             model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             stage="sft",do_train=True,do_eval=False,finetuning_type="full",
             freeze_vision_tower=True,freeze_multi_modal_projector=True,freeze_language_model=False,
             pure_bf16=False,bf16=True,fp16=False,optim="adamw_torch",learning_rate=1e-5,
             adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,weight_decay=0.,max_grad_norm=1.,
             lr_scheduler_type="cosine",warmup_ratio=.05,num_train_epochs=2.,
             per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
             gradient_checkpointing=True,use_reentrant_gc=False,deepspeed=str(ds),
             flash_attn="fa2",enable_liger_kernel=True,template="qwen3_5",enable_thinking=False,
             preserve_thinking=False,train_on_prompt=False,mask_history=False,packing=False,neat_packing=False,
             preprocessing_num_workers=1,dataloader_num_workers=0,cutoff_len=262144,
             seed=42,data_seed=42,report_to="none",logging_steps=1,save_strategy="epoch",save_total_limit=None,
             save_only_model=False,eval_strategy="no",prediction_loss_only=True,
             include_num_input_tokens_seen=True,ddp_timeout=1800,ddp_find_unused_parameters=False,
             overwrite_output_dir=False,plot_loss=False,logging_nan_inf_filter=False,
             dataset="ci-e2b-public-control",tokenized_path=str(root/"data/cache-262144-r2"),
             output_dir=str(root/'runs'/job_id),max_steps=-1,eval_on_start=False,
             disable_tqdm=True,remove_unused_columns=True,load_best_model_at_end=False)
    result=dict(experiment="CI_PUBLIC_POSITIVE_CONTROL", objective="global_supervised_token_mean",
                LF_arguments=cfg, job_id=job_id, training_authorized=True, primary_epochs=2,
                minimum_free_disk_bytes=160*1024**3,
                disk_budget_basis='Two observed native checkpoints at 60.24GB each, final weights, hardlinked exports, and >=30GiB reserve',
                expected_rows=rows, expected_optimizer_steps_per_epoch=steps, expected_total_optimizer_steps=2*steps,
                expected_ga_tail=(local_rows%8 or 8), expected_sampler_padding_per_epoch=padding,
                expected_unique_supervision_per_epoch=encoding['train']['supervised_tokens'],
                expected_total_supervised_exposures=(None if padding else 2*encoding['train']['supervised_tokens']),
                expected_total_trajectory_exposures=2*4*local_rows,
                padding_token_exposure_policy='Count actual per-rank labels; sampler padding is reported separately from unique corpus tokens',
                validation_policy="Offline unique-row global token NLL at M0, E1, E2; never distributed sample-mean eval_loss",
                data_manifest_sha256=hashlib.sha256((root/"checks/selection.json").read_bytes()).hexdigest(),
                encoding_sha256=hashlib.sha256((root/"checks/encoding-r2.json").read_bytes()).hexdigest(),
                normalization_origin="Transformers 5.8 native num_items_in_batch; loss kwargs enabled after LF processor guard; CPU and bounded Qwen4B DP4 ZeRO3 GA8/GA2 controls, numerical differences calibrated with same-mode repeats")
    path=root/"configs/train-r2.json"
    if path.exists():raise FileExistsError("Frozen recipe already exists")
    path.write_text(json.dumps(result,indent=2)+"\n")
    (root/"configs/m0-identity-r2.json").write_text(json.dumps(dict(model=cfg['model_name_or_path'],
        revision=cfg['model_revision'],policy="Original post-trained Qwen3.5-4B M0; no old M1/M2 weights"),indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main(sys.argv[1])
