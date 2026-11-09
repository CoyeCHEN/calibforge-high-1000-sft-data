"""Download the seven public calibration assets at the frozen DABStep revision."""
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi,hf_hub_download


REVISION='73c12f6e2b14890830ff10a1a23963653b2cb2a5'


def main(root):
    root=Path(root);repo='adyen/DABstep'
    info=HfApi().dataset_info(repo,revision=REVISION,files_metadata=True)
    entries=[]
    for file in info.siblings:
        if not file.rfilename.startswith('data/context/'):continue
        path=Path(hf_hub_download(repo,file.rfilename,repo_type='dataset',revision=REVISION,
                                 local_dir=root/'data/dabstep/context-release'))
        sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if file.lfs:assert sha==file.lfs.sha256
        entries.append(dict(path=file.rfilename,bytes=path.stat().st_size,sha256=sha))
        print(file.rfilename,path.stat().st_size,sha,flush=True)
    assert len(entries)==7
    (root/'checks/dabstep-context.json').write_text(json.dumps(dict(dataset=repo,revision=REVISION,
        files=entries),indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
