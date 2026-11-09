"""Cache official CIBench NLTK assets; direct source access, security checks on."""
import hashlib
import json
import os
from pathlib import Path
import sys
import nltk


def main(root):
    root=Path(root);folder=root/'environment-nltk/nltk_data'
    os.environ['NO_PROXY']=os.environ['no_proxy']='127.0.0.1,localhost,raw.githubusercontent.com'
    for asset in ['punkt','punkt_tab','stopwords']:
        assert nltk.download(asset,download_dir=folder,raise_on_error=True)
    files=[]
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
            files.append(dict(path=str(path.relative_to(folder)),bytes=path.stat().st_size,sha256=sha))
    (root/'checks/nltk-public-assets.json').write_text(json.dumps(dict(status='COMPLETE',
        resources=['punkt','punkt_tab','stopwords'],files=files,nltk_version=nltk.__version__,
        proxy_policy='Official raw.githubusercontent.com direct; NLTK pathsec enforcement retained'),indent=2)+'\n')
    (root/'environment-nltk/Dockerfile').write_text('FROM qwen35-ci-kernel:20261108\n'
        'COPY --chown=1002:1002 nltk_data/ /usr/local/share/nltk_data/\n')


if __name__=='__main__':main(sys.argv[1])
