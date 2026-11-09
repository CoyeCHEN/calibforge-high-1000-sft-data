"""Read-only, CPU comparison of native ZeRO-3 masters and final HF tensors.

Uses the official DeepSpeed lazy consolidation helper, not project export code.
No model loading, CUDA context, checkpoint rewriting or optimizer update.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def main():
    torch.set_num_threads(1)
    root = Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')
    checkpoint = root / 'runs/ci-e2b-8197-e2-expansion-audited-r2/checkpoint-514'
    target = root / 'exports/epoch2/model.safetensors'
    output = Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
    started = time.time()
    print('Reconstructing explicit global_step514 on CPU using official lazy helper', flush=True)
    state = get_fp32_state_dict_from_zero_checkpoint(
        str(checkpoint), tag='global_step514', exclude_frozen_parameters=False, lazy_mode=True)
    rows = []
    with safe_open(target, framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        for index, name in enumerate(sorted(keys)):
            actual = handle.get_tensor(name)
            if name not in state:
                rows.append(dict(name=name, status='MISSING_FROM_ZERO_RECONSTRUCTION'))
                continue
            master = state[name].contiguous()
            expected = master.to(actual.dtype)
            same = master.shape == actual.shape and torch.equal(expected, actual)
            finite = bool(torch.isfinite(master).all())
            row = dict(name=name, shape=list(actual.shape), elements=actual.numel(),
                       reconstructed_dtype=str(master.dtype), exported_dtype=str(actual.dtype),
                       same_after_export_dtype_cast=same, finite=finite)
            if not same and expected.shape == actual.shape:
                row['max_abs_after_cast'] = float((expected.float()-actual.float()).abs().max())
                row['unequal_elements'] = int(torch.count_nonzero(expected != actual))
            rows.append(row)
            del actual, master, expected
            if index % 100 == 0:
                print('checked', index + 1, '/', len(keys), name, flush=True)
    result = dict(scope='native global_step514 FP32 masters/frozen parameters vs final HF export; not live pre-save logits',
                  checkpoint=str(checkpoint), tag='global_step514', export=str(target),
                  seconds=time.time()-started, tensors=len(rows),
                  all_equal_after_dtype_cast=all(r.get('same_after_export_dtype_cast') and r.get('finite') for r in rows),
                  zero_only_keys=sorted(set(state)-keys), rows=rows,
                  cuda_initialized=torch.cuda.is_initialized(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'ZERO_MASTER_VS_HF_EXPORT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
    if not result['all_equal_after_dtype_cast']:
        raise RuntimeError('Native master and export differ; see per-tensor evidence')


if __name__ == '__main__':
    main()
