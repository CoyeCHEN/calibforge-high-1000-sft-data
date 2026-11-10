"""Verify the complete downloaded debug package before training; no GPU calls."""

import hashlib
import json
import math
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            raise RuntimeError('Missing packaged file; use the complete release archive: ' + entry['path'])
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise RuntimeError('Package file identity mismatch: ' + entry['path'])
    own = json.loads((root / 'metadata/OWN_SOURCE_MANIFEST.json').read_text())
    expected_commit = '03074a83743c70394d59b4e17b07a0b122495e54'
    if own['source_snapshot_commit'] != expected_commit:
        raise RuntimeError('Unexpected own-source baseline')
    for name, expected in own['files'].items():
        if sha(root / 'own_source' / name) != expected:
            raise RuntimeError('Own-source identity mismatch: ' + name)
    cfg = yaml.safe_load((root / 'configs/train-native.yaml').read_text())
    expected = {
        'num_train_epochs': 5.0,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1.0e-5,
        'warmup_ratio': 0.05,
        'lr_scheduler_type': 'cosine_with_min_lr',
        'lr_scheduler_kwargs': {'min_lr_rate': 0.1},
        'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1.0e-8,
        'weight_decay': 0.0, 'max_grad_norm': 1.0,
        'bf16': True, 'cutoff_len': 262144,
        'mask_history': False, 'packing': False,
        'enable_thinking': True, 'preserve_thinking': True,
        'gradient_checkpointing': True, 'use_reentrant_gc': False,
        'seed': 42, 'data_seed': 42, 'dataloader_drop_last': False,
        'save_strategy': 'epoch', 'save_only_model': False,
        'load_best_model_at_end': False,
        'output_dir': 'outputs/calibforge-high-native-e5',
    }
    for name, value in expected.items():
        if cfg.get(name) != value:
            raise RuntimeError('Unexpected training recipe field: ' + name)
    baseline = json.loads((root / 'metadata/BASELINE_03074a8.json').read_text())
    world = baseline['expected_dp_world_size']
    steps_per_epoch = math.ceil(math.ceil(baseline['train_rows'] / world)
                                / cfg['per_device_train_batch_size']
                                / cfg['gradient_accumulation_steps'])
    if steps_per_epoch != 20 or steps_per_epoch * cfg['num_train_epochs'] != 100:
        raise RuntimeError('Unexpected planned training-step count')
    split = json.loads((root / 'metadata/SPLIT.json').read_text())
    if len(split['train']) != 638 or len(split['validation']) != 35:
        raise RuntimeError('Unexpected public task membership')
    print(json.dumps({'package_files_verified': len(manifest['files']),
                      'source_files_verified': len(own['files']),
                      'source_snapshot_commit': expected_commit, 'epochs': 5,
                      'expected_optimizer_steps': 100,
                      'train_rows': 638, 'validation_rows': 35,
                      'scope': 'CPU integrity/configuration check; no training executed'}))


if __name__ == '__main__':
    main()
