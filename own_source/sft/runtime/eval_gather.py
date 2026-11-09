"""Keep Trainer's metric-aware gather on the native loss-only evaluation path.

Pinned Transformers 5.8 Seq2SeqTrainer.evaluate replaces gather_for_metrics with
gather even on loss-only evaluation. That counts distributed padding as real rows.
No generated-sequence evaluation or training computation is changed here.
"""
from functools import partial, wraps


def preserve_loss_only_gather(evaluation_loop):
    @wraps(evaluation_loop)
    def wrapped(self, dataloader, description, prediction_loss_only=None,
                ignore_keys=None, metric_key_prefix='eval'):
        requested = self.args.prediction_loss_only if prediction_loss_only is None else prediction_loss_only
        loss_only = requested or not self.args.predict_with_generate
        if loss_only:
            self.gather_function = partial(self.accelerator.gather_for_metrics,
                use_gather_object=self.args.eval_use_gather_object)
        return evaluation_loop(self, dataloader, description,
            prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)
    return wrapped
