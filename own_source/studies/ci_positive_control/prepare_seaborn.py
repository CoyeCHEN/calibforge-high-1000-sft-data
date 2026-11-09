"""Cache required public built-in datasets, without reference notebook access."""
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


def main(root):
    root=Path(root);names=set()
    for file in (root/'data/cibench_dataset').rglob('*.ipynb'):
        if '/datasources/' in str(file):continue
        for cell in json.loads(file.read_text())['cells']:
            if cell['cell_type']=='code':
                names.update(re.findall(r'(?:sns|seaborn)\.load_dataset\([\'\"]([\w-]+)', ''.join(cell['source'])))
    proxy=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
    def get(url):
        with proxy.open(url,timeout=30) as response:return response.read()
    revision=json.loads(get('https://api.github.com/repos/mwaskom/seaborn-data/commits/master'))['sha']
    folder=root/'environment-seaborn/seaborn_data';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for name in sorted(names):
        data=get(f'https://raw.githubusercontent.com/mwaskom/seaborn-data/{revision}/{name}.csv')
        (folder/(name+'.csv')).write_bytes(data)
        entries.append(dict(name=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        print(name,len(data),flush=True)
    (root/'checks/seaborn-public-assets.json').write_text(json.dumps(dict(revision=revision,files=entries,
        purpose='Public task inputs only; no code or reference output enters the kernel image'),indent=2)+'\n')
    (folder.parent/'Dockerfile').write_text('FROM qwen35-ci-kernel:20261108-nltk-r1\n'
        'COPY --chown=1002:1002 seaborn_data/ /usr/local/share/seaborn-data/\n'
        'ENV SEABORN_DATA=/usr/local/share/seaborn-data\n')


if __name__=='__main__':main(sys.argv[1])
