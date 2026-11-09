"""Separate full HF snapshots from immutable native epoch checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil

from sft.qwen35 import TEMPLATE_SHA,digest
from sft.runtime.checkpoint_archive import sha256,validate_checkpoint


def verify_snapshot(output, epoch, device):
    output=Path(output)
    manifest=json.loads((output/'HF_EXPORT_COMPLETE.json').read_text())
    assert manifest['epoch']==epoch
    for item in manifest['files']:
        file=output/item['path']
        assert file.stat().st_size==item['bytes'] and sha256(file)==item['sha256'],str(file)
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    import torch
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(output,local_files_only=True)
    assert digest(tokenizer.chat_template)==TEMPLATE_SHA
    model=AutoModelForImageTextToText.from_pretrained(output,local_files_only=True,
        dtype=torch.bfloat16,device_map=device).eval()
    batch=tokenizer.apply_chat_template([dict(role='user',content='Compute 1+1.')],
        add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
    batch=batch.to(device)
    with torch.inference_mode():result=model.generate(**batch,max_new_tokens=24,do_sample=False)
    suffix=tokenizer.decode(result[0,batch['input_ids'].shape[-1]:],skip_special_tokens=False)
    assert result.shape[-1]>batch['input_ids'].shape[-1]
    with (output/'HF_LOAD_SMOKE.json').open('x') as handle:
        json.dump(dict(status='PASSED',independent_process_from_training=True,
            device=device,thinking=False,text=suffix,
            export_manifest_sha256=sha256(output/'HF_EXPORT_COMPLETE.json')),handle,indent=2)
    print('FULL_HF_EXPORT_AND_INDEPENDENT_GENERATION_PASSED',str(output),suffix,flush=True)


def main(root,epoch,device,verify_existing=False):
    root=Path(root)
    spec=json.loads((root/'configs/train-r2.json').read_text())
    step=spec['expected_optimizer_steps_per_epoch']*epoch
    source=Path(spec['LF_arguments']['output_dir'])/f'checkpoint-{step}'
    state=validate_checkpoint(source,require_marker=True)
    assert state['global_step']==step and abs(state['epoch']-epoch)<1e-5
    output=root/f'exports/epoch{epoch}'
    if verify_existing:
        manifest=json.loads((output/'HF_EXPORT_COMPLETE.json').read_text())
        assert manifest['source_checkpoint']==str(source) and manifest['optimizer_step']==step
        assert manifest['complete_native_checkpoint_sha256']==sha256(source/'CHECKPOINT_COMPLETE.json')
        verify_snapshot(output,epoch,device)
        return
    output.mkdir(parents=True,exist_ok=False)
    index=source/'model.safetensors.index.json'
    weights=set(json.loads(index.read_text())['weight_map'].values())if index.exists()else {'model.safetensors'}
    for name in sorted(weights):os.link(source/name,output/name)
    for name in ['config.json','trainer_state.json']+(['model.safetensors.index.json']if index.exists()else[]):
        shutil.copy2(source/name,output/name)
    # Preserve the tokenizer saved by the native training checkpoint.
    for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json',
                 'added_tokens.json','special_tokens_map.json','generation_config.json',
                 'preprocessor_config.json','processor_config.json']:
        if (source/name).exists():shutil.copy2(source/name,output/name)
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p))for p in sorted(output.iterdir())]
    manifest=dict(epoch=epoch,optimizer_step=step,source_checkpoint=str(source),
        complete_native_checkpoint_sha256=sha256(source/'CHECKPOINT_COMPLETE.json'),
        model='Qwen/Qwen3.5-4B',base_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        full_hf_snapshot=True,stage_training_complete=epoch==2,
        tokenizer_policy='Native checkpoint tokenizer and processor files, without M0 replacement',files=files)
    (output/'HF_EXPORT_COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify_snapshot(output,epoch,device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('epoch',type=int,choices=[1,2])
    p.add_argument('--device',required=True,help='Explicit validated execution device, e.g. cuda:0')
    p.add_argument('--verify-existing',action='store_true',help='Verify an already published HF snapshot without rebuilding it')
    a=p.parse_args();main(a.root,a.epoch,a.device,a.verify_existing)
