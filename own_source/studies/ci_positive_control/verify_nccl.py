"""Short real four-card communication check; no model or optimizer training."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist


def main(root):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==torch.cuda.device_count()==4
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90))
    try:
        a=torch.tensor([rank+1.],device='cuda',dtype=torch.float32)
        b=a.to(torch.bfloat16)
        dist.all_reduce(a);dist.all_reduce(b)
        assert a.item()==b.item()==10
        dist.barrier(device_ids=[local])
        result=dict(status='PASSED',rank=rank,local_rank=local,pid=os.getpid(),
            device=torch.cuda.get_device_name(local),cuda=torch.version.cuda,
            nccl=str(torch.cuda.nccl.version()),bf16_and_fp32_all_reduce=True,
            allocated_bytes=torch.cuda.max_memory_allocated(),scope='Communication only; no optimizer update')
        (Path(root)/f'checks/host18-nccl-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main(sys.argv[1])
