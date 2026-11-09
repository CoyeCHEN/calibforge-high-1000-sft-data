"""Bounded single-device full-language gradient probe; never claims DP/ZeRO.

Two copies of a real, untruncated LF batch retain 1 and 64 original labels.
The deliberate mask intervention is identical across arms, not production data.
"""
import argparse
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["native", "native-repeat", "liger", "liger-loss-only"])
    p.add_argument("output", type=Path)
    p.add_argument("--bf16-model", action='store_true')
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_per_process_memory_fraction(0.65)
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from llamafactory.hparams import ModelArguments, FinetuningArguments, TrainingArguments
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from normalization import install

    root = Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
    model_path = '/data/models/Qwen/Qwen3.5-4B'
    source = root / 'ACTUAL_MODEL_AWARE_FORWARD_BATCH.pt'
    from transformers.tokenization_utils_base import BatchEncoding
    with torch.serialization.safe_globals([BatchEncoding]):
        cpu_batch = torch.load(source, weights_only=True)
    positions = torch.nonzero(cpu_batch['labels'][0, 1:] != -100).flatten() + 1
    assert len(positions) >= 65
    batches = []
    for selected in (positions[:1], positions[1:65]):
        b = copy.deepcopy(cpu_batch)
        b['labels'].fill_(-100)
        b['labels'][0, selected] = cpu_batch['labels'][0, selected]
        batches.append({k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()})
    if a.mode == 'liger-loss-only':
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5(rms_norm=False, swiglu=False)
    elif a.mode == 'liger':
        from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
        apply_liger_kernel(AutoConfig.from_pretrained(model_path, local_files_only=True),
            ModelArguments(model_name_or_path=model_path, enable_liger_kernel=True), True, False)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16 if a.bf16_model else torch.float32,
        attn_implementation='flash_attention_2').cuda()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('visual' not in name)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4205751296
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    if a.mode.startswith('liger'):
        model.forward = functools.partial(model.forward, skip_logits=True)
        install(CustomSeq2SeqTrainer, 'token')
        trainer = CustomSeq2SeqTrainer(model=model,
            tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
            finetuning_args=FinetuningArguments(finetuning_type='full'), processor=None,
            args=TrainingArguments(output_dir=str(a.output / 'lf'), report_to=[],
                bf16=True, gradient_accumulation_steps=2))
        trainer.model_accepts_loss_kwargs = False
        trainer.is_in_train = True
        trainer.current_gradient_accumulation_steps = 2
        trainer._ab_window = dict(step=1, counts=[1, 64], global_tokens=65, ga=2, dp=1, cursor=0)
    rows = []
    cpu_accumulation = {}
    start = time.time()
    for i, batch in enumerate(batches):
        t = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if a.mode.startswith('liger'):
                loss = trainer.compute_loss(model, copy.copy(batch)) / 2
            else:
                out = model(**{k:v for k,v in batch.items() if k != 'labels'}, use_cache=False)
                shifted = batch['labels'][:, 1:]
                valid = shifted != -100
                # Masked positions contribute zero: avoid materializing their FP32 CE workspace.
                loss = F.cross_entropy(out.logits[:, :-1][valid].float(),
                    shifted[valid], reduction='sum') / 65
        loss.backward()
        torch.cuda.synchronize()
        rows.append(dict(micro=i, supervised=[1, 64][i], loss=float(loss.detach()), seconds=time.time()-t))
        print(json.dumps(rows[-1]), flush=True)
        if not a.mode.startswith('liger'):
            del out
        if a.bf16_model:
            # Explicit diagnostic emulation, not a claimed DeepSpeed accumulation buffer.
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert param.grad is not None and param.grad.dtype == torch.bfloat16, name
                gradient = param.grad.detach().cpu().float()
                if name not in cpu_accumulation:
                    cpu_accumulation[name] = gradient
                else:
                    cpu_accumulation[name].add_(gradient)
                param.grad = None
    # Stream one parameter at a time to CPU files: no second GPU model or optimizer.
    from safetensors.torch import save_file
    manifest = []
    for number, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue
        if a.bf16_model:
            grad = cpu_accumulation.pop(name).contiguous()
        else:
            assert param.grad is not None, name
            grad = param.grad.detach().cpu().contiguous()
        assert torch.isfinite(grad).all(), name
        file = a.output / f'gradient-{number:04d}.safetensors'
        save_file({'gradient': grad}, str(file))
        manifest.append(dict(name=name, file=file.name, shape=list(grad.shape), dtype=str(grad.dtype)))
    report = dict(mode=a.mode, scope='SINGLE_GPU_FULL_LANGUAGE_GRADIENT_NO_ZER0_NO_OPTIMIZER',
        source_batch_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mask_intervention='first original target token and next 64; context unmodified',
        rows=rows, global_tokens=65, loss=sum(x['loss'] for x in rows),
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        elapsed=time.time()-start, parameters=manifest,
        model_source=inspect.getfile(type(model)),
        precision='BF16 parameters/backward, explicit CPU FP32 microbatch accumulation, FA2' if a.bf16_model else 'FP32 parameters, BF16 autocast, FA2',
        not_tested=['actual GA8 or 9304/7064 tail', 'DP collectives', 'ZeRO3', 'Adam update'])
    (a.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'parameters'}), flush=True)


if __name__ == '__main__':
    main()
