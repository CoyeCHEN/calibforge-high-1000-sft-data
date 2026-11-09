"""One process-local loss intervention; native LF/Trainer/DeepSpeed keep the loop."""
import functools
import hashlib
import json
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def install(trainer_class, objective, expected=None):
    """B multiplies native mean CE by D*G*n/N; native backward divides by G,D.

    A uses multiplier 1. Both count exactly the same shifted labels, call the
    same collective, and record the same observations. No flag or kernel change.
    """
    assert objective in ('segment', 'token')
    native_batches = trainer_class.get_batch_samples
    native_loss = trainer_class.compute_loss

    @functools.wraps(native_batches)
    def batches(self, iterator, num_batches, device):
        samples, num_items = native_batches(self, iterator, num_batches, device)
        assert self.model_accepts_loss_kwargs is False
        assert self.compute_loss_func is None and num_items is None
        counts = []
        identities = []
        for batch in samples:
            labels = batch['labels']
            assert labels.shape[0] == 1 and (labels[:, 0] == -100).all()
            counts.append(int(labels[:, 1:].ne(-100).sum()))
            assert counts[-1] > 0
            if expected is not None:
                keep = batch['attention_mask'][0].bool()
                identities.append(dict(
                    input_sha256=digest(batch['input_ids'][0][keep].tolist()),
                    labels_sha256=digest(labels[0][keep].tolist()),
                    supervised_tokens=counts[-1]))
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        count = torch.tensor(sum(counts), dtype=torch.int64, device=device)
        if world > 1:
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        step = self.state.global_step + 1
        if expected is not None:
            old = expected[step]
            assert len(identities) == len(old), (step, len(identities), len(old))
            for live, frozen in zip(identities, old):
                assert all(live[k] == frozen[k] for k in live), (step, live, frozen)
        self._ab_window = dict(step=step, counts=counts, global_tokens=int(count),
                               ga=len(samples), dp=world, cursor=0)
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        append(Path(self.args.output_dir) / f'normalization-rank{rank}.jsonl',
               dict(event='window', objective=objective, **self._ab_window,
                    identities=identities, actual_order_matches_original=expected is not None))
        return samples, num_items

    @functools.wraps(native_loss)
    def loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        training = model.training and self.is_in_train
        n = int(inputs['labels'][:, 1:].ne(-100).sum()) if training else None
        result = native_loss(self, model, inputs, return_outputs=return_outputs,
                             num_items_in_batch=num_items_in_batch)
        if not training:
            return result
        window = self._ab_window
        assert self.current_gradient_accumulation_steps == window['ga']
        assert n == window['counts'][window['cursor']]
        raw = result[0] if return_outputs else result
        factor = (window['dp'] * window['ga'] * n / window['global_tokens']
                  if objective == 'token' else 1.0)
        scaled = raw * factor
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        append(Path(self.args.output_dir) / f'normalization-rank{rank}.jsonl',
               dict(event='microbatch', objective=objective, step=window['step'],
                    micro=window['cursor'], supervised_tokens=n,
                    global_tokens=window['global_tokens'], ga=window['ga'], dp=window['dp'],
                    native_segment_mean=float(raw.detach()), multiplier=factor,
                    loss_before_native_ga_division=float(scaled.detach())))
        window['cursor'] += 1
        return (scaled, result[1]) if return_outputs else scaled

    trainer_class.get_batch_samples = batches
    trainer_class.compute_loss = loss
