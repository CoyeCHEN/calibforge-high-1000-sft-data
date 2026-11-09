"""Archive immutable native checkpoints; never copy live training tensors."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from .paths import workspace

def storage_settings():
    return workspace() / 'configs/checkpoint-storage.json'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree_directories(path):
    for directory in sorted((p for p in Path(path).rglob('*') if p.is_dir()), reverse=True):
        sync_directory(directory)
    sync_directory(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(path):
    result = []
    for item in sorted(Path(path).rglob('*')):
        if item.is_symlink():
            raise ValueError('Checkpoint symlinks are not accepted: ' + str(item))
        if item.is_file() and item.name not in ('.pin.json', 'ARCHIVE_COMPLETE.json', 'RESTORE_COMPLETE.json'):
            stat = item.stat()
            result.append(dict(path=str(item.relative_to(path)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return result


def validate_checkpoint(path, world_size=None, require_marker=False):
    path = Path(path)
    if world_size is None:
        marker=path/'CHECKPOINT_COMPLETE.json'
        if not marker.exists():raise ValueError('Explicit world size required before native completion')
        world_size=json.loads(marker.read_text())['identity']['world_size']
    if world_size not in(4,8):raise ValueError('Unaccepted checkpoint world size')
    state = json.loads((path / 'trainer_state.json').read_text())
    step = state.get('global_step', 0)
    if step < 1:
        raise ValueError('Checkpoint has no completed optimizer step')
    for name in ('config.json', 'scheduler.pt', 'training_args.bin', 'latest'):
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError('Incomplete checkpoint: ' + name)
    tag = (path / 'latest').read_text().strip()
    if tag != f'global_step{step}':
        raise ValueError('DeepSpeed latest tag/global_step mismatch')
    native = path / tag
    for rank in range(world_size):
        required = [path / f'rng_state_{rank}.pth',
                    native / f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt',
                    native / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt']
        if any(not item.is_file() or not item.stat().st_size for item in required):
            raise ValueError(f'Incomplete native checkpoint for rank {rank}')
    index = path / 'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if any(not (path / name).is_file() for name in weights):
        raise ValueError('Complete model export missing')
    marker = path / 'CHECKPOINT_COMPLETE.json'
    if require_marker and not marker.is_file():
        raise ValueError('Native save completion marker missing')
    if marker.exists():
        saved = json.loads(marker.read_text())
        for item in saved['files']:
            candidate = path / item['path']
            if not candidate.is_file() or candidate.stat().st_size != item['bytes']:
                raise ValueError('Native completion inventory mismatch')
    return state


def durable_completion(path, identity):
    """Called only after all native saves and all on_save callbacks have finished."""
    path = Path(path)
    validate_checkpoint(path,world_size=identity['world_size'])
    files = inventory(path)
    start = time.monotonic()
    for item in files:
        with (path / item['path']).open('rb') as stream:
            os.fsync(stream.fileno())
    sync_tree_directories(path)
    atomic_json(path / 'CHECKPOINT_COMPLETE.json', dict(identity=identity, files=files,
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return time.monotonic() - start


def enqueue(path, settings, job_id, pin_root=None):
    """Hardlink CLOSED immutable files so Trainer rotation cannot delete the copy source."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', job_id):
        raise ValueError('Unsafe job identity')
    path = Path(path).resolve()
    state = validate_checkpoint(path)
    if path.name != f'checkpoint-{state["global_step"]}':
        raise ValueError('Native checkpoint directory/global_step mismatch')
    key = f'{job_id}-checkpoint-{state["global_step"]}'
    pin_root = Path(pin_root or settings['pin_root'])
    pin_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if pin_root.stat().st_dev != path.stat().st_dev:
        raise ValueError('Pin source and pin directory must be on the same filesystem')
    pin = pin_root / key
    pin.mkdir(mode=0o750)  # Never overwrite a previous pin.
    files = inventory(path)
    for item in files:
        destination = pin / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        os.link(path / item['path'], destination)
    metadata = dict(key=key, job_id=job_id, checkpoint_name=path.name, original=str(path), pin=str(pin),
                    files=files, queued_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(pin / '.pin.json', metadata)
    queue = Path(settings['queue_root'])
    atomic_json(queue / (key + '.json'), metadata)
    return metadata


def copy_files(source, destination, workers=1):
    files = inventory(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o750)
    groups = [[] for _ in range(workers)]
    for index, item in enumerate(files):
        groups[index % workers].append(item['path'])
    def transfer(names):
        if names:
            subprocess.run(['rsync', '-a', '--whole-file', '--fsync', '--from0', '--files-from=-',
                            str(source) + '/', str(destination) + '/'],
                           input=('\0'.join(names) + '\0').encode(), check=True, timeout=7200)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(transfer, groups))
    return dict(seconds=time.monotonic() - start, bytes=sum(x['bytes'] for x in files), workers=workers)


def verify_archive(path):
    path = Path(path)
    marker = json.loads((path / 'ARCHIVE_COMPLETE.json').read_text())
    if {x['path'] for x in inventory(path)} != {x['path'] for x in marker['files']}:
        raise ValueError('Archive file inventory mismatch')
    for item in marker['files']:
        if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:
            raise ValueError('Unsafe archive member')
        candidate = path / item['path']
        if not candidate.is_file() or candidate.stat().st_size != item['bytes'] or sha256(candidate) != item['sha256']:
            raise ValueError('Archive integrity mismatch: ' + item['path'])
    validate_checkpoint(path)
    return marker


def archive_one(entry, settings):
    parent = Path(settings['archive_root']) / entry['job_id']
    if Path(settings['archive_root']).stat().st_dev != settings['archive_device']:
        raise OSError('Archive mount identity changed; refuse writing to an underlying filesystem')
    final = parent / entry['checkpoint_name']
    if final.exists():
        marker=verify_archive(final)
        if marker['key']!=entry['key']:raise ValueError('Published archive has a different job identity')
        pin=Path(entry['pin'])
        if pin.exists():
            if pin.resolve()==Path(entry['original']).resolve()or json.loads((pin/'.pin.json').read_text())['key']!=entry['key']:
                raise ValueError('Invalid pin ownership')
            shutil.rmtree(pin)
        return dict(archive=str(final),already_published=True,**marker['metrics'])
    pin = Path(entry['pin'])
    if pin.resolve() == Path(entry['original']).resolve() or json.loads((pin / '.pin.json').read_text())['key'] != entry['key']:
        raise ValueError('Invalid pin ownership/identity')
    validate_checkpoint(pin)
    before = inventory(pin)
    if before != entry['files']:
        raise ValueError('Pinned checkpoint changed before archival')
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = parent / ('.' + entry['checkpoint_name'] + '.partial')
    metrics = copy_files(pin, temporary, int(settings['transfer_workers']))
    start = time.monotonic()
    files = []
    for item in before:
        digest = sha256(pin / item['path'])
        if sha256(temporary / item['path']) != digest:
            raise ValueError('Archive checksum mismatch: ' + item['path'])
        files.append(item | dict(sha256=digest))
    if inventory(pin) != before:
        raise ValueError('Pinned source mutated during archival')
    metrics['verify_seconds'] = time.monotonic() - start
    marker = dict(key=entry['key'], job_id=entry['job_id'], files=files, metrics=metrics,
                  completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    sync_tree_directories(temporary)
    atomic_json(temporary / 'ARCHIVE_COMPLETE.json', marker)
    os.rename(temporary, final)
    sync_directory(parent)
    # Only unlink this service's newly created pin. The native checkpoint is untouched.
    shutil.rmtree(pin)
    return dict(archive=str(final), **metrics)


def worker(settings):
    queue = Path(settings['queue_root'])
    failed = Path(settings['failed_root'])
    queue.mkdir(parents=True, exist_ok=True, mode=0o750)
    failed.mkdir(parents=True, exist_ok=True, mode=0o750)
    with (queue.parent / 'archive-worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        for item in sorted(queue.glob('*.json')):
            entry = dict(key=item.stem)
            try:
                entry = json.loads(item.read_text())
                result = archive_one(entry, settings)
                atomic_json(failed.parent / 'archive-results' / item.name, dict(status='completed', **result))
                item.unlink()
                print(json.dumps(dict(status='completed', key=entry['key'], **result)), flush=True)
            except Exception as error:
                failures += 1
                entry.update(status='failed', error=type(error).__name__ + ': ' + str(error))
                atomic_json(item, entry)
                os.rename(item, failed / (item.stem + '-' + uuid.uuid4().hex[:8] + '.json'))
                print(json.dumps(dict(status='failed', key=entry['key'], error=entry['error'])), flush=True)
        return failures


def restore(archive, destination, workers=1):
    archive, destination = Path(archive), Path(destination)
    marker = verify_archive(archive)
    if destination.exists():
        raise FileExistsError('Restore destination preserved: ' + str(destination))
    temporary = destination.with_name('.' + destination.name + '.restoring')
    metrics = copy_files(archive, temporary, workers)
    for item in marker['files']:
        if sha256(temporary / item['path']) != item['sha256']:
            raise ValueError('Restored file mismatch: ' + item['path'])
    validate_checkpoint(temporary)
    sync_tree_directories(temporary)
    atomic_json(temporary / 'RESTORE_COMPLETE.json', dict(source=str(archive), files=marker['files'], metrics=metrics))
    os.rename(temporary, destination)
    sync_directory(destination.parent)
    validate_checkpoint(destination)
    return metrics
