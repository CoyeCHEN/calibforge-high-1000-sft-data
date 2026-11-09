"""Bounded, idempotent cleanup of one phase's explicitly labelled kernels."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main(root,phase):
    listed=subprocess.run(['docker','ps','-aq','--filter','label=chenkeyi.run=qwen35-ci-positive-control-20261108',
                           '--filter','label=chenkeyi.phase='+phase],capture_output=True,text=True,check=True,timeout=15)
    removed=[]
    for cid in listed.stdout.split():
        value=json.loads(subprocess.check_output(['docker','inspect',cid],text=True,timeout=15))[0]
        labels=value['Config']['Labels']
        if labels.get('chenkeyi.run')!='qwen35-ci-positive-control-20261108' or labels.get('chenkeyi.phase')!=phase:
            raise RuntimeError('Container ownership does not match this phase')
        subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,check=True,timeout=25)
        removed.append(cid)
    with (Path(root)/'logs/kernel-cleanup.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=time.time(),phase=phase,removed=removed))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('phase');a=p.parse_args();main(a.root,a.phase)
