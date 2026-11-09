"""One predeclared validation stop rule, shared by both CalibForge initializations."""
import math


def calibforge_validation_health(history):
    evaluations=[e for e in history if 'eval_loss'in e]
    if any(not math.isfinite(float(e['eval_loss']))or e['eval_loss']<0 for e in evaluations):
        raise RuntimeError('Invalid validation loss')
    baselines=[e for e in evaluations if e.get('step')==0]
    if not baselines:
        raise RuntimeError('Native pre-training validation baseline missing')
    baseline=float(baselines[0]['eval_loss'])
    epochs={}
    for event in evaluations:
        epoch=float(event.get('epoch')or 0)
        # A resumed startup evaluation halfway through an epoch is not a second
        # completed epoch. Repeated evaluations at one epoch count only once.
        if epoch>=1 and abs(epoch-round(epoch))<1e-7:
            epochs[round(epoch)]=float(event['eval_loss'])
    latest=sorted(epochs)[-2:]
    severe=(len(latest)==2 and latest[1]==latest[0]+1
            and all(epochs[e]>=baseline*1.5 and epochs[e]>=baseline+0.5 for e in latest))
    return dict(rule='two_consecutive_epochs_ge_1_5x_baseline_and_plus_0_5_nats',
                baseline_eval_loss=baseline,completed_epoch_losses={str(e):epochs[e]for e in latest},
                severe_validation_regression=severe,
                scope='Predeclared engineering stop rule, not a paper parameter or Agent-capability guarantee')
