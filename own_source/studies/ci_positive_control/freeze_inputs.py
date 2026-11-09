"""Stream hashes of the actual M0, selected raw data and cache, once."""
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main(root):
    root=Path(root);model=Path('/data/models/Qwen/Qwen3.5-4B')
    files=[p for p in model.iterdir()if p.is_file() and p.suffix in ('.json','.jinja','.safetensors','.txt')]
    identity=dict(model='Qwen/Qwen3.5-4B',revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',path=str(model),
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(files)])
    (root/'checks/m0-model-files.json').write_text(json.dumps(identity,indent=2)+'\n')
    paths=[root/name for name in ['data/train-raw.jsonl','data/validation-raw.jsonl',
        'data/train-encoding-index-r2.jsonl','data/validation-encoding-index-r2.jsonl',
        'configs/train-r2.json','configs/ds_z3.json','configs/tools.json','configs/sglang-template-tools.json',
        'checks/selection.json','checks/encoding-r2.json']]
    paths+=[p for p in (root/'data/cache-262144-r2').rglob('*')if p.is_file()]
    manifest={str(p.relative_to(root)):sha(p)for p in sorted(paths)}
    (root/'checks/training-input-files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FROZEN_MODEL_AND_TRAINING_INPUT_FILES',len(files),len(manifest),flush=True)


if __name__=='__main__':main(sys.argv[1])
