"""Read real rank evidence; missing GPU evidence remains unverified."""
import json
import math
from pathlib import Path
from .checkpoint_archive import validate_checkpoint

def continuous_acceptance(directory):
    folder=Path(directory);ranks=[]
    for rank in range(4):
        path=folder/f'preflight-evidence-rank{rank}.jsonl'
        events=[json.loads(line)for line in path.open()]if path.exists()else[]
        steps={e['global_step']:e for e in events if e['event']=='optimizer_step'}
        saves=[e for e in events if e['event']=='checkpoint_save_complete'and e['global_step']==1]
        begins=[e for e in events if e['event']=='train_begin']
        ends=[e for e in events if e['event']=='train_end']
        required=[steps.get(1),saves[-1]if saves else None,steps.get(2)]
        stable=all(required)and all(len({e['identity'][field]for e in required})==1
            for field in ('pid','process_start_ticks','cuda_context','engine_id','optimizer_id'))
        real_cuda=all(required)and all(e['identity'].get('cuda_context')and e['identity'].get('cuda_allocated',0)>0 for e in required)
        finite=all(required)and all(e.get('gradients')and all(g['finite']for g in e['gradients'].values())
            and any(e['changed'].values())and e['optimizer_moment_dtypes']==['torch.float32']for e in (steps[1],steps[2]))
        logs=[e['metrics']['loss']for e in events if e['event']=='metrics'and'loss'in e.get('metrics',{})]
        losses=len(logs)>=2 and all(math.isfinite(x)for x in logs)
        single_run=len(begins)==1 and begins[0]['global_step']==0 and len(ends)==1 and ends[0]['global_step']==2
        uninterrupted=not any(e['event']=='controlled_preflight_stop'for e in events)
        save_ok=bool(saves and saves[-1]['process_context_engine_unchanged']and saves[-1]['training_state_unchanged'])
        passed=bool(stable and real_cuda and finite and losses and single_run and uninterrupted and save_ok)
        ranks.append(dict(rank=rank,passed=passed,same_PID_context_engine=bool(stable),CUDA_retained=bool(real_cuda),
            finite_updates=bool(finite),single_training_lifecycle=single_run,normal_save=save_ok))
    complete=False
    try:validate_checkpoint(folder/'checkpoint-1',require_marker=True);complete=True
    except (OSError,ValueError,KeyError):pass
    final_progress=read(folder/'trainer_state.json').get('global_step')==2
    return dict(passed=all(r['passed']for r in ranks)and complete and final_progress,ranks=ranks,
                complete_native_checkpoint=complete,final_trainer_step2=final_progress)


def capacity(directory,name=""):
    directory=Path(directory);ranks=[]
    for rank in range(4):
        path=directory/f'preflight-evidence-rank{rank}.jsonl'
        events=[json.loads(x)for x in path.open()]if path.exists()else[]
        steps={e['global_step']:e for e in events if e['event']=='optimizer_step'}
        resumed=next((e for e in events if e['event']=='train_begin'and e['global_step']==1),None)
        initial=steps.get(1);final=steps.get(2)
        # The recovery profile reuses the original completed first update and
        # reads its missing Adam fingerprint from that exact saved checkpoint.
        initial_optimizer=next((e for e in events if e['event']=='checkpoint_optimizer_reference'),None)if name=='short-stateproof'else initial
        restored=bool(initial and resumed and
            initial['weight_hashes']==resumed['initial_weight_hashes']and
            initial_optimizer and initial_optimizer.get('optimizer_state_sample_sha256') and
            initial_optimizer['optimizer_state_sample_sha256']==resumed.get('optimizer_state_sample_sha256')and
            initial.get('scheduler_state') and initial.get('scheduler_state')==resumed.get('scheduler_state'))
        finite=bool(steps)and all(g['finite']for e in steps.values()for g in e.get('gradients',{}).values())
        types=bool(steps)and all(((initial_optimizer if step==1 else e)or{}).get('optimizer_moment_dtypes')==['torch.float32']for step,e in steps.items())
        checkpoints={}
        for step in (1,):
            checkpoint=directory/f'checkpoint-{step}'
            checkpoints[str(step)]=(checkpoint/'trainer_state.json').exists()and bool(list(checkpoint.rglob('*optim_states.pt')))
        full_checkpoint=all(checkpoints.values())
        final_state=directory/'trainer_state.json'
        final_progress=final_state.exists()and json.loads(final_state.read_text()).get('global_step')==2
        rng=next((e for e in events if e['event']=='rng_restored'and e['global_step']==1),None)
        rng_restored=bool(rng and all(rng.get('matches',{}).get(k,False)for k in ('python','numpy','torch_cpu','torch_cuda')))
        losses={key:[e['metrics'][key]for e in events if e['event']=='metrics'and key in e.get('metrics',{})]for key in ('loss','eval_loss')}
        valid_loss=all(values and all(math.isfinite(float(v))for v in values)for values in losses.values())
        model_inputs=bool(steps)and all(e.get('model_inputs_verified')and e.get('supervised_tokens_since_begin',0)>0 for e in steps.values())
        ranks.append(dict(rank=rank,optimizer_steps=sorted(steps),FP32_master_restore=restored,
            optimizer_state_sample_restore=restored,full_checkpoint_present=full_checkpoint,
            full_checkpoints_by_step=checkpoints,
            final_trainer_step2=final_progress,RNG_restore=rng_restored,
            actual_model_inputs_match_CPU_acceptance=model_inputs,
            finite_gradients=finite,FP32_optimizer_state=types,loss_recorded=valid_loss,
            passed=bool(initial and final and restored and finite and types and full_checkpoint and final_progress and rng_restored and valid_loss and model_inputs)))
    return dict(status='CONFIRMED'if all(r['passed']for r in ranks)else'UNVERIFIED',
        passed=all(r['passed']for r in ranks),ranks=ranks)


def read(path):return json.loads(path.read_text())if path.exists()else{}
