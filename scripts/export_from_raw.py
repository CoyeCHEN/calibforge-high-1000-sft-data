"""Adapt the real Recorder aggregate to the unchanged own ShareGPT exporter."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--package-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--check-against', type=Path)
a = p.parse_args()
root = a.package_root.resolve()
a.output.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('own_export', root / 'own_source/studies/native_lf/export_teacher_lf.py')
own = importlib.util.module_from_spec(spec)
spec.loader.exec_module(own)
raw = {r['task_id']: r for r in map(json.loads, (root / 'raw_data/qualified.jsonl').open())}
split = json.loads((root / 'metadata/SPLIT.json').read_text())
report = {'rule': 'Stop after the last real assistant decision; trailing tool feedback has no subsequent target.',
          'assistant_messages_removed': 0, 'rows': {}}
for target in ['train', 'validation']:
    trimmed = 0
    removed = 0
    targets = 0
    path = a.output / (target + '.sharegpt.jsonl')
    with path.open('x') as out:
        for task in split[target]:
            source = raw[task]
            messages = source['input']
            last = max(i for i, m in enumerate(messages) if m['role'] == 'assistant')
            tail = messages[last + 1:]
            assert all(m['role'] == 'tool' for m in tail)
            row = own.convert({'messages': messages[:last + 1], 'tools': source['tools']})
            n = sum(m['role'] == 'assistant' for m in messages)
            assert sum(m['from'] in ('gpt', 'function_call') for m in row['conversations']) == n
            row.update(task_id=task, trial_uuid=source['trial_uuid'])
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            targets += n
            trimmed += bool(tail)
            removed += len(tail)
    if a.check_against:
        assert path.read_bytes() == (a.check_against / path.name).read_bytes(), ('export_changed', target)
    report['rows'][target] = {'rows': len(split[target]), 'trajectories_with_terminal_tool_feedback_removed': trimmed,
        'terminal_tool_messages_removed': removed, 'assistant_targets_preserved': targets,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
info = {'calibforge_high_' + target: {'file_name': target + '.sharegpt.jsonl', 'formatting': 'sharegpt',
    'columns': {'messages': 'conversations', 'tools': 'tools'}} for target in ['train', 'validation']}
(a.output / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
(a.output / 'EXPORT_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report), flush=True)
