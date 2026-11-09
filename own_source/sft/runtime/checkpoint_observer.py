"""Checkpoint identity and durability evidence around native save APIs."""
import ctypes
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch
from transformers import Trainer
from transformers.trainer_callback import CallbackHandler

from .checkpoint_archive import storage_settings, durable_completion, enqueue,atomic_json


class CheckpointObserver:
    def __init__(self, callback, optimizer_evidence, archive_enabled=False):
        self.callback = callback
        self.optimizer_evidence = optimizer_evidence
        self.archive_enabled = archive_enabled
        self.settings = json.loads(storage_settings().read_text()) if archive_enabled else None
        self.trainer = None
        self.saving = False
        self.save_started = None
        self.before = None
        self.driver = None

    def identity(self):
        cuda = torch.cuda.is_initialized()
        context = None
        if cuda:
            if self.driver is None:
                self.driver = ctypes.CDLL('libcuda.so.1')
                self.driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                self.driver.cuCtxGetCurrent.restype = ctypes.c_int
            pointer = ctypes.c_void_p()
            status = self.driver.cuCtxGetCurrent(ctypes.byref(pointer))
            if status:
                raise RuntimeError(f'cuCtxGetCurrent failed: {status}')
            context = hex(pointer.value) if pointer.value else None
        trainer = self.trainer
        return dict(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    world_size=torch.distributed.get_world_size()if torch.distributed.is_initialized()else 1,
                    stage=os.environ.get('QWEN35_STAGE'),config_sha256=os.environ.get('QWEN35_CONFIG_SHA256'),
                    cuda_context=context, device=torch.cuda.current_device() if cuda else None,
                    model_id=id(trainer.model) if trainer else None,
                    engine_id=id(trainer.model_wrapped) if trainer else None,
                    optimizer_id=id(trainer.optimizer) if trainer else None,
                    cuda_allocated=torch.cuda.memory_allocated() if cuda else 0,
                    cuda_reserved=torch.cuda.memory_reserved() if cuda else 0)

    def state_sample(self):
        return dict(identity=self.identity(),
                    weights={name: self.callback.weight(parameter) for name, parameter in self.callback.selected},
                    optimizer=self.optimizer_evidence(self.trainer.optimizer),
                    scheduler=self.trainer.lr_scheduler.state_dict())

    def install(self):
        observer = self
        native_train = Trainer.train

        @functools.wraps(native_train)
        def train(trainer, *args, **kwargs):
            observer.trainer = trainer
            return native_train(trainer, *args, **kwargs)
        Trainer.train = train

        native_save = Trainer._save_checkpoint
        @functools.wraps(native_save)
        def save_checkpoint(trainer, *args, **kwargs):
            failure = [None]
            if observer.archive_enabled and trainer.state.is_world_process_zero:
                try:
                    if Path(trainer.args.output_dir).stat().st_dev!=observer.settings['hot_device']:
                        raise OSError('Hot storage mount identity changed')
                    free = shutil.disk_usage(trainer.args.output_dir).free
                    if free < observer.settings['minimum_checkpoint_free_bytes']:
                        raise OSError('Insufficient checkpoint filesystem space before native save')
                except OSError as error:
                    # Broadcast even a stat/statvfs failure; do not strand other ranks
                    # in a collective while rank0 exits before the broadcast.
                    failure[0] = type(error).__name__+': '+str(error)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(failure, src=0)
            if failure[0]:
                raise RuntimeError(failure[0])
            observer.save_started = time.monotonic()
            observer.before = observer.state_sample()
            observer.saving = True
            observer.callback.record('checkpoint_save_begin', trainer.state, **observer.before)
            try:
                result = native_save(trainer, *args, **kwargs)
            except Exception as error:
                observer.callback.record('checkpoint_save_failed', trainer.state,
                                         error=type(error).__name__ + ': ' + str(error))
                observer.saving = False
                raise
            return result
        Trainer._save_checkpoint = save_checkpoint

        native_callbacks = CallbackHandler.on_save
        @functools.wraps(native_callbacks)
        def after_all_callbacks(handler, args, state, control):
            result = native_callbacks(handler, args, state, control)
            if not observer.saving:
                return result
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            folder = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            fsync_seconds = pin_seconds = 0.0
            failure = [None]
            archive_status = 'disabled'
            if state.is_world_process_zero:
                try:
                    fsync_seconds = durable_completion(folder, observer.identity())
                except Exception as error:
                    failure[0] = type(error).__name__ + ': ' + str(error)
                if not failure[0] and observer.archive_enabled:
                    try:
                        start = time.monotonic()
                        job_id = os.environ['QWEN35_JOB_ID']
                        enqueue(folder, observer.settings, job_id)
                        pin_seconds = time.monotonic() - start
                        archive_status = 'queued'
                        subprocess.run(['systemctl', '--user', 'start', '--no-block',
                                        observer.settings['archive_service']], check=True, timeout=15)
                    except Exception as error:
                        archive_status = 'failed'
                        observer.callback.record('checkpoint_archive_enqueue_failed', state,
                                                 error=type(error).__name__ + ': ' + str(error))
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(failure, src=0)
            if failure[0]:
                observer.callback.record('checkpoint_durability_failed', state, error=failure[0])
                raise RuntimeError('Local checkpoint durability failed: ' + failure[0])
            after = observer.state_sample()
            before = observer.before
            stable_fields = ('pid', 'process_start_ticks', 'cuda_context', 'model_id', 'engine_id', 'optimizer_id')
            stable = all(before['identity'][key] == after['identity'][key] for key in stable_fields)
            unchanged = before['weights'] == after['weights'] and before['optimizer'] == after['optimizer'] and before['scheduler'] == after['scheduler']
            if not stable or not unchanged:
                raise RuntimeError('Native checkpoint changed process/context/training state')
            observer.callback.record('checkpoint_save_complete', state, identity=after['identity'],
                                     process_context_engine_unchanged=stable, training_state_unchanged=unchanged,
                                     fsync_seconds_rank0=fsync_seconds, pin_seconds_rank0=pin_seconds,
                                     archive_status_rank0=archive_status,
                                     total_pause_seconds=time.monotonic() - observer.save_started)
            observer.saving = False
            if (observer.callback.runtime_recipe=='qwen35_772_b32_e5'
                and not observer.callback.bounded_acceptance and state.is_world_process_zero
                and state.epoch is not None and abs(state.epoch-3)<0.01):
                atomic_json(Path(args.output_dir)/'EPOCH3_DIAGNOSTIC_REQUEST.json',dict(
                    status='QUEUED_WITHOUT_TRAINING_WAIT',stage=os.environ.get('QWEN35_STAGE'),
                    checkpoint=str(folder),optimizer_step=state.global_step,epoch=state.epoch,
                    normal_epoch4_and_epoch5_continue=True))
            return result
        CallbackHandler.on_save = after_all_callbacks
        return self
