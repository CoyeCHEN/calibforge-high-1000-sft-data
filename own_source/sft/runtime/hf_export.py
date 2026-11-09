"""Seal LF's final native HF export; never mistake ZeRO shards for a model."""
import argparse
import json
from pathlib import Path
from .launch import exported_m1
from .checkpoint_archive import sha256,atomic_json
from ..qwen35 import REVISION,TEMPLATE_SHA,digest


def seal(path):
    path=Path(path).resolve();exported_m1(path)
    config=json.loads((path/'config.json').read_text())
    if config['model_type']!='qwen3_5':raise ValueError('Wrong full Qwen3.5 model class')
    index=path/'model.safetensors.index.json'
    weights=set(json.loads(index.read_text())['weight_map'].values())if index.exists()else{'model.safetensors'}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    if digest(tokenizer.chat_template)!=TEMPLATE_SHA:raise ValueError('Official tokenizer template changed')
    required={'config.json','tokenizer_config.json','tokenizer.json','trainer_state.json'}|weights
    for optional in ('chat_template.jinja','generation_config.json','preprocessor_config.json','processor_config.json'):
        if(path/optional).exists():required.add(optional)
    if index.exists():required.add(index.name)
    files=[]
    for name in sorted(required):
        file=path/name
        if file.is_symlink()or not file.is_file():raise ValueError('Incomplete HF export: '+name)
        files.append(dict(path=name,bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns,sha256=sha256(file)))
    result=dict(model='Qwen/Qwen3.5-4B',base_revision=REVISION,path=str(path),complete_final_hf=True,
        optimizer_resume=False,resume_note='Same-stage resume uses native complete checkpoint-N; M1-to-M2 loads these final weights with NEW Adam/scheduler.',files=files)
    destination=path/'HF_EXPORT_COMPLETE.json'
    if destination.exists():
        if json.loads(destination.read_text())!=result:raise ValueError('Sealed export changed')
    else:atomic_json(destination,result)
    return result


def smoke(path):
    import torch
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    torch.set_num_threads(1);seal(path)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map='cpu').eval()
    batch=tokenizer.apply_chat_template([dict(role='user',content='Compute 1+1.')],add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True)
    with torch.inference_mode():output=model.generate(**batch,max_new_tokens=16,do_sample=False)
    if output.shape[-1]<=batch['input_ids'].shape[-1]:raise ValueError('No generated token')
    result=dict(model_loaded=True,generated_tokens=output.shape[-1]-batch['input_ids'].shape[-1],cuda_initialized=torch.cuda.is_initialized(),
        export_manifest_sha256=sha256(Path(path)/'HF_EXPORT_COMPLETE.json'),scope='CPU HF model/Tokenizer generation smoke, not a serving-parser or Agent evaluation proof')
    atomic_json(Path(path)/'HF_LOAD_SMOKE.json',result)
    return result


def final_verified(path):
    """Require a sealed export and actual load/generation; invalidate changed files."""
    try:
        path=Path(path);exported_m1(path)
        seal_path=path/'HF_EXPORT_COMPLETE.json';manifest=json.loads(seal_path.read_text())
        proof=json.loads((path/'HF_LOAD_SMOKE.json').read_text())
        return bool(manifest['complete_final_hf']and manifest['base_revision']==REVISION and proof['model_loaded']and proof['generated_tokens']>0
            and proof['export_manifest_sha256']==sha256(seal_path)and all(
                (path/f['path']).stat().st_size==f['bytes']and(path/f['path']).stat().st_mtime_ns==f['mtime_ns']and
                sha256(path/f['path'])==f['sha256']for f in manifest['files']))
    except (OSError,ValueError,KeyError):return False


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    print(json.dumps(smoke(a.model)if a.smoke else seal(a.model),indent=2))
