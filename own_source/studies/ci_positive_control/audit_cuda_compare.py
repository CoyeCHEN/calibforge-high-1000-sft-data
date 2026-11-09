"""Read-only full gradient comparison; actual separable CPU AdamW first update.

The optimizer runs one parameter at a time to bound memory. This is NOT ZeRO3.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open


def sums(x, y):
    total = [0., 0., 0., 0.]
    maximum = 0.
    # Largest single tensor fits comfortably under the declared GPU memory cap.
    chunk = x.numel() if x.is_cuda else 500000
    for a, b in zip(x.reshape(-1).split(chunk), y.reshape(-1).split(chunk)):
        a, b = a.double(), b.double()
        for i, v in enumerate([(a*a).sum(), (b*b).sum(), (a*b).sum(), ((a-b)**2).sum()]):
            total[i] += float(v)
        maximum = max(maximum, float((a-b).abs().max()))
    return total + [maximum]


def metrics(v):
    aa, bb, ab, error, maximum = v
    return dict(reference_norm=math.sqrt(aa), actual_norm=math.sqrt(bb),
        norm_ratio=math.sqrt(bb/aa) if aa else None,
        cosine=ab/math.sqrt(aa*bb) if aa*bb else None,
        relative_l2=math.sqrt(error/aa) if aa else None, max_abs=maximum)


def add(dst, src):
    for i in range(4):
        dst[i] += src[i]
    dst[4] = max(dst[4], src[4])


def load(path):
    with safe_open(path, framework='pt', device='cpu') as f:
        return f.get_tensor('gradient')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('right', choices=['native-repeat', 'liger', 'liger-loss-only'])
    p.add_argument('--adam', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.65)
    left = json.loads((a.root/'native/result.json').read_text())
    right = json.loads((a.root/a.right/'result.json').read_text())
    assert left['source_batch_sha256'] == right['source_batch_sha256']
    assert left['parameters'] == right['parameters']
    rows = []
    total = [0.]*5
    for item in left['parameters']:
        x = load(a.root/'native'/item['file']).to(a.device)
        y = load(a.root/a.right/item['file']).to(a.device)
        values = sums(x, y)
        rows.append(dict(name=item['name'], gradient=metrics(values)))
        add(total, values)
    report = dict(scope='FULL_LANGUAGE_GRADIENT_SINGLE_GPU', right=a.right, comparison_device=a.device,
        comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_loss=left['loss'], actual_loss=right['loss'],
        gradient=metrics(total), parameters=rows)
    if a.adam:
        model = Path('/data/models/Qwen/Qwen3.5-4B')
        mapping = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        scales = [min(1., 1./(math.sqrt(n)+1e-6)) for n in total[:2]]
        totals = {k: [0.]*5 for k in ['first_moment', 'second_moment', 'fp32_update']}
        for item, row in zip(left['parameters'], rows):
            with safe_open(model/mapping[item['name']], framework='pt', device='cpu') as f:
                initial = f.get_tensor(item['name']).float().to(a.device)
            outputs = []
            for mode, scale in zip(['native', a.right], scales):
                parameter = torch.nn.Parameter(initial.clone())
                parameter.grad = load(a.root/mode/item['file']).to(a.device).float().clone().mul_(scale)
                optimizer = torch.optim.AdamW([parameter], lr=1e-5, betas=(.9,.999),
                    eps=1e-8, weight_decay=0, foreach=False)
                optimizer.step()
                state = optimizer.state[parameter]
                assert int(state['step']) == 1
                outputs.append([state['exp_avg'].clone(), state['exp_avg_sq'].clone(),
                                (parameter.detach()-initial).clone()])
                del optimizer, parameter
            for key, x, y in zip(totals, *outputs):
                values = sums(x, y)
                row[key] = metrics(values)
                add(totals[key], values)
            del outputs, initial
        report['adam'] = dict(scope=f'actual {a.device} torch.optim.AdamW per parameter, globally clipped; no distributed optimizer',
            lr=1e-5, clip_scales=scales, **{k: metrics(v) for k, v in totals.items()})
    output = a.output or a.root/f'compare-{a.right}.json'
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'parameters'}), flush=True)


if __name__ == '__main__':
    main()
