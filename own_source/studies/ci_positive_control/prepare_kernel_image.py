"""Build only official public input assets into a cached Python base image."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile


def main(root):
    root=Path(root);context=root/'environment'
    context.mkdir(exist_ok=True)
    target=context/'datasources/data';target.mkdir(parents=True,exist_ok=False)
    written={}
    with zipfile.ZipFile(root/'data/cibench_dataset.zip') as archive:
        for kind in ['cibench_template','cibench_generation']:
            members=[]
            for item in archive.infolist():
                parts=Path(item.filename).parts
                if not item.is_dir() and len(parts)>=5 and parts[:2]==('cibench_dataset',kind) and parts[3]=='data':
                    members.append((parts[2],item.filename,'/'.join(parts[4:])))
            for library,name,relative in sorted(members):
                assert '..' not in Path(relative).parts
                dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source,dest.open('wb') as out:
                    shutil.copyfileobj(source,out,1024*1024)
                written[relative]=dict(source=name,bytes=dest.stat().st_size,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    assert not list(target.rglob('*.ipynb'))
    cwd=str(root/'data/cibench_dataset/datasources')
    dockerfile=('FROM python:3.11-slim\n'
        'RUN sed -i s/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g /etc/apt/sources.list.d/debian.sources '
        '&& apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 '
        '&& rm -rf /var/lib/apt/lists/*\n'
        'COPY --chown=1002:1002 datasources/ '+cwd+'/\n'
        'WORKDIR '+cwd+'\n')
    (context/'Dockerfile').write_text(dockerfile)
    (root/'checks/kernel-image-inputs.json').write_text(json.dumps(dict(
        inputs=written,total_bytes=sum(v['bytes'] for v in written.values()),
        no_reference_notebooks=True,archive_sha256='adb1bb098bc9661f3e8183e5b07572b546572a982afc7d69e6868bedec97efc2',
        dockerfile_sha256=hashlib.sha256(dockerfile.encode()).hexdigest()),indent=2)+'\n')
    print('PUBLIC_INPUT_IMAGE_CONTEXT_READY',len(written),flush=True)


if __name__=='__main__':main(sys.argv[1])
