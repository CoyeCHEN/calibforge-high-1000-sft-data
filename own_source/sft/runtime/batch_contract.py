"""Measure the live DeepSpeed DP group; GPU count is not batch-size evidence."""


def measure_data_parallel(engine, distributed):
    if not distributed.is_initialized():
        raise RuntimeError('Native distributed process group is not initialized')
    group = engine.data_parallel_group
    dp_size = distributed.get_world_size(group)
    world_size = distributed.get_world_size()
    if dp_size != engine.dp_world_size or dp_size != world_size:
        raise RuntimeError('Only the verified pure data-parallel ZeRO-3 route is supported')
    ranks = (list(range(world_size)) if group is None
             else distributed.get_process_group_ranks(group))
    if sorted(ranks) != list(range(world_size)):
        raise RuntimeError('Unexpected native data-parallel group membership')
    microbatch = engine.train_micro_batch_size_per_gpu()
    accumulation = engine.gradient_accumulation_steps()
    global_batch = engine.train_batch_size()
    if global_batch != dp_size * microbatch * accumulation:
        raise RuntimeError('DeepSpeed global batch disagrees with its actual DP group')
    return dict(world_size=world_size, data_parallel_group_size=dp_size,
                data_parallel_group_ranks=ranks, microbatch_per_gpu=microbatch,
                gradient_accumulation_steps=accumulation,
                effective_global_batch=global_batch)
