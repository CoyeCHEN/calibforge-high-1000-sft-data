"""Frozen public Harbor task provenance; this module never approves a review."""
import json
import re
from pathlib import Path

from . import qwen35 as codec
from .runtime.checkpoint_archive import sha256


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError('Published source evidence SHA mismatch')
    return json.loads(Path(path).read_text())


def source_files_sha256(files):
    # Training CPU may read a byte-identical snapshot on another authorized host.
    # Absolute storage paths are locations, not the public task's identity.
    return codec.digest(sorted((dict(relative_path=item['relative_path'],sha256=item['sha256'])
                                for item in files),key=lambda item:item['relative_path']))


def approved_scope(scope):
    if scope.get('kind') != 'approved_published_harbor_tasks' or scope.get('status') != 'APPROVED':
        raise ValueError('Published task scope is not approved')
    if not scope.get('dataset') or not re.fullmatch('[0-9a-f]{40}', scope.get('revision', '')):
        raise ValueError('Fixed public dataset identity required')
    if not scope.get('license') or not scope.get('tasks'):
        raise ValueError('Public license and explicit task scope required')
    review = checked_json(scope['isolation_review'], scope['isolation_review_sha256'])
    if (review.get('kind') != 'evaluation_isolation_review' or review.get('status') != 'APPROVED'
        or any(review.get(k) != scope[k] for k in ('dataset', 'revision'))
        or 'terminal-bench-2.1' not in review.get('retained_evaluation_sets', [])):
        raise ValueError('Approved evaluation isolation review required; lexical screening alone is insufficient')
    reviewed = review.get('tasks', {})
    groups = {}; tasks = set()
    for task in scope['tasks']:
        task_id = task['task_id']; group = task['task_group_id']; split = task['split']
        if task_id in tasks or split not in ('train', 'validation') or group in groups and groups[group] != split:
            raise ValueError('Duplicate task or task-group split leakage')
        tasks.add(task_id); groups[group] = split
        if (reviewed.get(task_id, {}).get('decision') != 'APPROVED'
            or reviewed[task_id].get('source_files_sha256') != task['source_files_sha256']):
            raise ValueError('Task/source identity absent from approved isolation review')
    contract = scope['execution_contract']
    for key, size in (('harness_commit', 40), ('harbor_commit', 40), ('harness_wheel_sha256', 64)):
        if not re.fullmatch('[0-9a-f]{'+str(size)+'}', contract.get(key, '')):
            raise ValueError('Frozen published collection execution contract required')
    return contract


def task_origin(origin, task, scope):
    """Bind a public source to its original split and a separate SFT split."""
    if (origin.get('kind') != 'published_harbor_task'
        or any(origin.get(k) != scope[k] for k in ('dataset', 'revision'))
        or origin.get('instance_id') != task['task_id']
        or origin.get('task_group') != task['task_group_id']
        or origin.get('source_split') != task.get('source_split')
        or not origin.get('source_split')
        or origin.get('source_task_path') != task.get('source_task_path')):
        raise ValueError('Public task identity/original split changed; never relabel it as authored')
    files = origin.get('source_files', [])
    names = [item['relative_path'] for item in files]
    if (len(set(names)) != len(names)
        or not {'instruction.md', 'task.toml', 'tests/test.sh'} <= set(names)
        or not any(name.startswith('environment/') for name in names)):
        raise ValueError('Original public instruction/environment/Verifier files required')
    for item in files:
        relative = Path(item['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(item['path']) != item['sha256']:
            raise ValueError('Published source file identity mismatch')
    if source_files_sha256(files) != task['source_files_sha256']:
        raise ValueError('Published source files differ from approved task')
def collection_scope(entry, identity):
    scope = checked_json(identity['approved_scope'], identity['approved_scope_sha256'])
    contract = approved_scope(scope)
    if any(identity.get(key) != contract[key] for key in contract):
        raise ValueError('Published collection differs from frozen execution contract')
    task = next((task for task in scope['tasks'] if task['task_id'] == entry['task_id']), None)
    if task is None or any(task.get(k) != entry[k] for k in ('task_group_id', 'split')):
        raise ValueError('Public task/group/SFT split absent from approved scope')
    task_origin(entry['origin'], task, scope)
    return contract
