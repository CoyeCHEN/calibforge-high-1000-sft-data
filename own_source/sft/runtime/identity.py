"""Content identity shared by training logs and acceptance gates."""
import hashlib
from pathlib import Path


def training_source_sha256():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    paths=sorted((root/'runtime').glob('*.py'))+[root/'qwen35.py']
    for path in paths:
        digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()
