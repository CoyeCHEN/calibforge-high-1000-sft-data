"""Native LF data reproduction: no project tokenizer, label builder or Trainer patch."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--package-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.package_root.resolve()
a.output.mkdir(parents=True, exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DISABLE_VERSION_CHECK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Native LF environment interface: ordinary source-code markup is text, not media.
for kind in ['IMAGE', 'VIDEO', 'AUDIO']:
    os.environ[kind + '_PLACEHOLDER'] = '<|lf_' + kind.lower() + '_input|>'
sys.path.insert(0, str(root / 'framework/LLaMAFactory/src'))
import torch
from datasets import load_from_disk
from llamafactory.hparams import ModelArguments, DataArguments, TrainingArguments
from llamafactory.model import load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
torch.set_num_threads(1)

ma = ModelArguments(model_name_or_path=str(root / 'model_assets'), cache_dir=str(a.output / 'hf-cache'))
bundle = load_tokenizer(ma)
tok = bundle['tokenizer']
da = DataArguments(dataset='calibforge_high_train', eval_dataset='calibforge_high_validation',
    dataset_dir=str(root / 'data'), template='qwen3_5', enable_thinking=True,
    preserve_thinking=True, mask_history=False, cutoff_len=262144, packing=False,
    preprocessing_num_workers=4, overwrite_cache=True, tokenized_path=str(a.output / 'cache'))
ta = TrainingArguments(output_dir=str(a.output / 'unused-training'), use_cpu=True,
    do_train=True, report_to='none', dataloader_num_workers=0)
template = get_template_and_fix_tokenizer(tok, da)
get_dataset(template, ma, da, ta, 'sft', tok, bundle['processor'])
cache = load_from_disk(a.output / 'cache')
full_args = copy.deepcopy(da)
full_args.cutoff_len = 1000000
full_processor = SupervisedDatasetProcessor(template=template, tokenizer=tok,
    processor=bundle['processor'], data_args=full_args)

def digest(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

summary = {}
for split, dataset, file in [('train', 'calibforge_high_train', 'train.sharegpt.jsonl'),
                             ('validation', 'calibforge_high_validation', 'validation.sharegpt.jsonl')]:
    attr = get_dataset_list([dataset], str(root / 'data'))[0]
    converter = get_dataset_converter('sharegpt', attr, da)
    sources = [json.loads(l) for l in (root / 'data' / file).open()]
    assert not any(os.environ[k + '_PLACEHOLDER'] in json.dumps(r, ensure_ascii=False)
                   for r in sources for k in ['IMAGE', 'VIDEO', 'AUDIO'])
    assert len(cache[split]) == len(sources)
    stats = {'rows': len(sources), 'input_tokens': 0, 'supervised_tokens': 0, 'targets': 0, 'max_length': 0}
    with (a.output / (split + '-row-hashes.jsonl')).open('x') as hashes:
        for index, (source, row) in enumerate(zip(sources, cache[split])):
            aligned = converter(source)
            ids, labels = full_processor._encode_data_example(aligned['_prompt'], aligned['_response'],
                aligned['_system'], aligned['_tools'], [], [], [])
            assert ids == row['input_ids'] and labels == row['labels'], ('truncation_or_row_change', split, index)
            assert len(ids) <= 262144 and labels[0] == -100
            assert all(v == -100 or v == token for v, token in zip(labels, ids))
            starts = [i for i, v in enumerate(labels) if v != -100 and (i == 0 or labels[i-1] == -100)]
            expected = sum(m['from'] in ('gpt', 'function_call') for m in source['conversations'])
            assert len(starts) == expected, ('target_count', split, index)
            n = sum(v != -100 for v in labels[1:])
            assert n > 0
            stats['input_tokens'] += len(ids)
            stats['supervised_tokens'] += n
            stats['targets'] += expected
            stats['max_length'] = max(stats['max_length'], len(ids))
            hashes.write(json.dumps({'row': index, 'task_id': source['task_id'], 'input_sha256': digest(ids),
                'labels_sha256': digest(labels), 'tokens': len(ids), 'supervised': n, 'targets': expected}) + '\n')
    summary[split] = stats
tok.save_pretrained(a.output / 'runtime-tokenizer')
from importlib.metadata import version
versions = {k: version(k) for k in ['torch', 'transformers', 'tokenizers', 'datasets', 'accelerate', 'peft', 'trl']}
result = {'pipeline': 'own ShareGPT thin conversion -> unmodified pinned LF native data processor',
    'lf_commit': 'c683e59edd8f893abc5e1403184a1d0644817d2a', 'template': 'qwen3_5',
    'enable_thinking': True, 'preserve_thinking': True, 'mask_history': False,
    'row_unit': 'one whole trajectory; all assistant targets supervised once',
    'summary': summary, 'untruncated_native_reference_all_rows_equal': True,
    'SWE_Master_preprocessing_used': False, 'teacher_segments_used': False,
    'versions': versions, 'tokenizer_class': type(tok).__module__ + '.' + type(tok).__name__,
    'cuda_initialized': torch.cuda.is_initialized(), 'scope': 'CPU data/cache only; no backbone forward, training or evaluation'}
result['native_media_placeholder_environment'] = {k + '_PLACEHOLDER': os.environ[k + '_PLACEHOLDER'] for k in ['IMAGE', 'VIDEO', 'AUDIO']}
assert not result['cuda_initialized']
(a.output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
