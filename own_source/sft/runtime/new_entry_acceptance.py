"""Accept current-source, recipe-specific native update/save/continue/resume records."""
import json
import math
from pathlib import Path
from .identity import training_source_sha256
from .checkpoint_archive import validate_checkpoint
from .config import validate_native


def capacity772(directory,configuration,inputs,dp):
    """Read native update/save/resume evidence from the current 772-row recipe."""
    root=Path(directory);ranks=[];indices=[];all_seen=set()
    validate_native(configuration,bounded=True)
    allowed={(r['input_sha256'],r['labels_sha256']):r for r in inputs['selected_rows']}
    for rank in range(dp):
        events=[json.loads(line)for line in(root/f'preflight-evidence-rank{rank}.jsonl').open()]
        steps=[e for e in events if e['event']=='optimizer_step']
        begins=[e for e in events if e['event']=='train_begin']
        assert [e['global_step']for e in steps]==[1,2,3]
        assert [e['global_step']for e in begins]==[0,2]
        for begin in begins:
            p=begin['parallelism']
            assert p['world_size']==p['data_parallel_group_size']==dp
            assert p['effective_global_batch']==32 and p['gradient_accumulation_steps']==32//dp
            assert begin['runtime_recipe']=='qwen35_772_b32_e5'
            assert begin['training_source_sha256']==training_source_sha256()
            assert begin['loss_only_official_liger_api'] and begin['visual_trainable']==0
        for e in steps:
            assert e['optimizer_moment_dtypes']==['torch.float32']
            assert e['master_partition_dtypes']==['torch.float32'] and e['fp32_master_partition_bytes']>0
            assert all(g['finite']for g in e['gradients'].values())
            assert len(e['model_input_batches'])==32//dp
            for batch in e['model_input_batches']:
                pair=(batch['input_sha256'],batch['labels_sha256']);r=allowed[pair]
                assert batch['nonpadding_tokens']==r['text_tokens']
                assert batch['supervised_tokens']==r['loss_tokens']
                assert batch['sequence_tokens']<=configuration['cutoff_len']
                all_seen.add(pair);indices.append(batch['canary_sample_index'])
        assert all(e['parameter_update_observed']for e in steps[1:])
        before=steps[1];resumed=begins[1]
        assert resumed['initial_weight_hashes']==before['weight_hashes']
        assert resumed['optimizer_state_sample_sha256']==before['optimizer_state_sample_sha256']
        assert resumed['scheduler_state']==before['scheduler_state']
        assert resumed['pid']!=begins[0]['pid']
        assert any(e['event']=='rng_restored'and e['global_step']==2 and all(e['matches'].values())for e in events)
        saves={e['global_step']:e for e in events if e['event']=='checkpoint_save_complete'}
        assert all(saves[s]['process_context_engine_unchanged']and saves[s]['training_state_unchanged']for s in(1,2))
        losses=[e['metrics']['loss']for e in events if e['event']=='metrics'and'loss'in e['metrics']]
        assert len(losses)>=3 and all(math.isfinite(float(v))for v in losses)
        ranks.append(dict(rank=rank,max_memory_allocated=max(e['peak_allocated_bytes']for e in steps),
            max_memory_reserved=max(e['peak_reserved_bytes']for e in steps),
            fp32_master_partition_bytes=steps[-1]['fp32_master_partition_bytes'],
            optimizer_moment_bytes=steps[-1]['optimizer_moment_bytes'],
            input_tokens=sum(b['nonpadding_tokens']for e in steps for b in e['model_input_batches']),
            elapsed_seconds=max(e['elapsed_seconds']for e in steps)))
    assert all_seen==set(allowed) and sorted(indices)==list(range(96))
    validate_checkpoint(root/'checkpoint-2',world_size=dp,require_marker=True)
    assert json.loads((root/'trainer_state.json').read_text())['global_step']==3
    return dict(passed=True,dp=dp,ga=32//dp,global_batch=32,context=configuration['cutoff_len'],
        ranks=ranks,maximum_real_input_tokens=max(r['text_tokens']for r in inputs['selected_rows']),
        source_sha256=training_source_sha256(),all_selected_real_rows_exercised=True,
        native_cross_process_model_adam_scheduler_rng_and_data_resume=True,
        formal_training=False,diagnostic_weights_must_not_initialize_formal_training=True)


def new_entry_acceptance(directory,length,minimum_supervision=1,*,configuration=None,input_contract=None):
    if not directory:return dict(passed=False,reason='MISSING_CURRENT_ENTRY_EVIDENCE')
    root=Path(directory);ranks=[];sample_indices=[]
    try:
        if configuration is not None:
            validate_native(configuration,bounded=configuration.get('max_steps')==3)
            if configuration['cutoff_len']!=length:raise ValueError('Acceptance length/config mismatch')
        appendix_e=configuration is not None and configuration.get('runtime_recipe')=='calibforge_appendix_e'
        ga=32 if appendix_e else 4
        lr=1e-5 if appendix_e else(2e-6 if length==65536 else 5e-7)
        warmup=0.05 if appendix_e else 0.03
        for rank in range(4):
            with (root/f'preflight-evidence-rank{rank}.jsonl').open()as stream:
                events=[json.loads(line)for line in stream]
            steps=[e for e in events if e['event']=='optimizer_step']
            if [e['global_step']for e in steps]!=[1,2,3]:raise ValueError('Expected unique steps 1/2/3')
            begins=[e for e in events if e['event']=='train_begin']
            if [e['global_step']for e in begins]!=[0,2]:raise ValueError('Expected initial and checkpoint-2 resume')
            for begin in begins:
                if (begin['training_source_sha256']!=training_source_sha256()or begin['gradient_accumulation_steps']!=ga
                    or begin['cutoff_len']!=length or begin['warmup_ratio']!=warmup
                    or begin['learning_rate']!=lr
                    or not begin['bf16']or begin['pure_bf16']):raise ValueError('Source/config identity mismatch')
                if appendix_e:
                    observed=begin['parallelism']
                    if (begin['runtime_recipe']!='calibforge_appendix_e' or begin['ignore_data_skip']
                        or observed['world_size']!=4 or observed['data_parallel_group_size']!=4
                        or sorted(observed['data_parallel_group_ranks'])!=[0,1,2,3]
                        or observed['microbatch_per_gpu']!=1 or observed['gradient_accumulation_steps']!=32
                        or observed['effective_global_batch']!=128):
                        raise ValueError('Missing actual CalibForge DP/global-batch or native data-resume evidence')
            saves={e['global_step']:e for e in events if e['event']=='checkpoint_save_complete'}
            for n in (1,2):
                if not saves[n]['process_context_engine_unchanged']or not saves[n]['training_state_unchanged']:raise ValueError('Save changed engine/state')
            chain=[steps[0],saves[1],steps[1]]
            for field in ('pid','process_start_ticks','cuda_context','engine_id','optimizer_id'):
                if len({e['identity'][field]for e in chain})!=1:raise ValueError('Save did not retain engine')
            if not chain[0]['identity']['cuda_context']:raise ValueError('Missing real CUDA evidence')
            if begins[1]['pid']==begins[0]['pid']:raise ValueError('Not a cross-process resume')
            seen_inputs=set()
            for step in steps:
                if len(step['model_input_batches'])!=ga:raise ValueError('Configured accumulation batches not observed')
                if any(b['supervised_tokens']<minimum_supervision for b in step['model_input_batches']):raise ValueError('Insufficient representative assistant target')
                if not all(v['finite']for v in step['gradients'].values()):raise ValueError('Nonfinite gradient')
                if step['optimizer_moment_dtypes']!=['torch.float32']:raise ValueError('Not FP32 Adam')
                if input_contract is not None:
                    for batch in step['model_input_batches']:
                        key=(batch['input_sha256'],batch['labels_sha256'])
                        expected=input_contract['allowed'][key]
                        if (batch['nonpadding_tokens']!=expected['text_tokens']
                            or batch['supervised_tokens']!=expected['loss_tokens']
                            or batch['sequence_tokens']>length):
                            raise ValueError('Actual canary input/labels differ from frozen corpus')
                        seen_inputs.add(key)
                        sample_indices.append(batch['canary_sample_index'])
            if not all(s['parameter_update_observed']for s in steps[1:]):raise ValueError('Missing nonzero-LR updates')
            if input_contract is not None:
                if not input_contract['required']<=seen_inputs:
                    raise ValueError('Each rank must exercise longest-input and largest-supervision corpus rows')
            elif length>=98304 and not all(min(b['nonpadding_tokens']for b in s['model_input_batches'])>=length-1072 for s in steps):raise ValueError('Not real long-context inputs')
            resumed=begins[1];before=steps[1]
            if (resumed['initial_weight_hashes']!=before['weight_hashes']or
                resumed['optimizer_state_sample_sha256']!=before['optimizer_state_sample_sha256']or
                resumed['scheduler_state']!=before['scheduler_state']):raise ValueError('Model/Adam/scheduler resume mismatch')
            if not any(e['event']=='rng_restored'and e['global_step']==2 and all(e['matches'].values())for e in events):raise ValueError('No native RNG restore proof')
            losses=[e['metrics']['loss']for e in events if e['event']=='metrics'and'loss'in e['metrics']]
            if len(losses)<3 or not all(math.isfinite(v)for v in losses):raise ValueError('Missing finite losses')
            ranks.append(dict(rank=rank,passed=True))
        if input_contract is not None and sorted(sample_indices)!=list(range(input_contract['diagnostic_exposures'])):
            raise ValueError('Native data position did not resume across unique diagnostic rows')
        validate_checkpoint(root/'checkpoint-2',require_marker=True)
        if json.loads((root/'trainer_state.json').read_text())['global_step']!=3:raise ValueError('Incomplete final step')
        return dict(passed=True,ranks=ranks,source_sha256=training_source_sha256(),
                    runtime_recipe='calibforge_appendix_e'if appendix_e else'agent_sft_v1',
                    gradient_accumulation_steps=ga)
    except (OSError,ValueError,KeyError,TypeError)as error:
        return dict(passed=False,reason=str(error),ranks=ranks)
