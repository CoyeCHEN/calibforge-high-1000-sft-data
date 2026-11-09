"""Decode actual cached targets, then use the serving parser; CPU/read-only inputs."""
import collections
import hashlib
import inspect
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from transformers import AutoTokenizer


def main():
    root = Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')
    run = Path('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/expansion-audited-r2')
    output = root / 'parser-roundtrip-all-r1'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained('/data/models/Qwen/Qwen3.5-4B', local_files_only=True)
    tools = [Tool.model_validate(t) for t in json.loads((run / 'configs/sglang-template-tools.json').read_text())]
    cache = load_from_disk(str(run / 'data/cache-262144-r2'))
    totals = {}
    with (output / 'differences.jsonl').open('w') as differences, (output / 'rows.jsonl').open('w') as rows:
        for split in ['train', 'validation']:
            index = {json.loads(line)['trace_id']: i for i, line in enumerate((run / f'data/{split}-encoding-index-r2.jsonl').open())}
            counts = collections.Counter()
            for line in (run / f'data/{split}-raw.jsonl').open():
                source = json.loads(line)['source']
                row = cache[split][index[source['id']]]
                spans = []
                start = None
                for pos, label in enumerate(row['labels'] + [-100]):
                    if label != -100 and start is None:
                        start = pos
                    elif label == -100 and start is not None:
                        spans.append((start, pos))
                        start = None
                assistants = [(i, m) for i, m in enumerate(source['messages']) if m['role'] == 'assistant']
                if len(spans) != len(assistants):
                    raise ValueError(('assistant span count', source['id'], len(spans), len(assistants)))
                row_bad = False
                for (left, right), (mi, message) in zip(spans, assistants):
                    text = tokenizer.decode(row['input_ids'][left:right], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    expected = []
                    omitted = []
                    for call in message.get('tool_calls') or []:
                        function = call['function']
                        args = function['arguments']
                        args = json.loads(args) if isinstance(args, str) else args
                        # HF union-column null padding is recorded explicitly, not silently treated as text equality.
                        omitted.append([k for k, value in args.items() if value is None])
                        expected.append({'name': function['name'], 'arguments': {k: v for k, v in args.items() if v is not None}})
                    parser = FunctionCallParser(tools, 'qwen3_coder')
                    error = None
                    try:
                        _, calls = parser.parse_non_stream(text)
                        actual = [{'name': c.name, 'arguments': json.loads(c.parameters)} for c in calls]
                    except Exception as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                        actual = []
                    equal = error is None and actual == expected
                    counts.update(targets=1, calls=len(expected), exact_targets=equal,
                                  null_padding_keys_omitted=sum(map(len, omitted)))
                    counts.update({'tool_' + c['name']: 1 for c in expected})
                    record = {'split': split, 'trace_id': source['id'], 'message_index': mi,
                              'token_start': left, 'token_end_exclusive': right, 'arguments_exact_after_null_padding_removal': equal,
                              'null_padding_keys_removed': omitted,
                              'rendered_target_sha256': hashlib.sha256(text.encode()).hexdigest()}
                    rows.write(json.dumps(record) + '\n')
                    if not equal:
                        row_bad = True
                        counts['different_targets'] += 1
                        differences.write(json.dumps({**record, 'expected': expected, 'actual': actual,
                                                     'rendered_target': text, 'error': error}, ensure_ascii=False) + '\n')
                counts.update(rows=1, different_rows=row_bad)
                if counts['rows'] % 1000 == 0:
                    print(json.dumps({'split': split, **counts}), flush=True)
            totals[split] = dict(counts)
    file = Path(inspect.getfile(type(parser.detector)))
    report = {'scope': 'all cached assistant target text -> actual qwen3_coder non-stream parser; not sampled generation',
              'stats': totals, 'parser_file': str(file), 'parser_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cuda_initialized': torch.cuda.is_initialized()}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
