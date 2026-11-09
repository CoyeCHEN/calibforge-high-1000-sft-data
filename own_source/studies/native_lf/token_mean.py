"""Use Transformers' native global-token denominator with pinned LF/TF.

LF disables loss kwargs whenever a processor exists. Qwen3.5's native and
Liger causal losses both accept num_items_in_batch. CPU/DDP and bounded
Qwen4B DP4/ZeRO3 GA8/GA2 controls verified; GPU differences are reported
against same-mode repeat error. Teacher offload/long contexts remain untested.
"""

def enable_native_token_mean(trainer):
    if trainer.compute_loss_func is not None:
        raise ValueError('A custom compute_loss_func would change the objective')
    if not trainer.args.average_tokens_across_devices:
        raise ValueError('Global token mean requires average_tokens_across_devices')
    trainer.model_accepts_loss_kwargs = True
