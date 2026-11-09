# Public Code Interpreter positive control

This study starts from the original post-trained Qwen3.5-4B, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. No old terminal M1/M2 weights or
trajectories enter training. The terminal project's large SFT remains paused.

## Frozen data and tool contract

- Dataset: `jupyter-agent/jupyter-agent-dataset`, revision
  `10a8dd461bb6c580fa25b83cd3b2f6810f1d6385`, actual split `non_thinking`.
- 2,000 unique training QA trajectories; 194 validation trajectories. Notebook,
  declared Kaggle identity, specific shared filenames and input paths form
  connected groups; a group cannot cross splits. Unknown upstream source aliases
  remain a provenance limitation. These are 2,000 trajectories, not 10,892 cells.
- Only release-labelled E2B execution records, complete sequential tool pairs,
  at least two Python calls, and a final answer agreeing verbatim with the
  released reference. The release does not include teacher HTTP or E2B sandbox
  identities. This filter is not an independent verifier of every answer.
- 10,892 training assistant decisions, 8,892 Python calls, 2,796,737 unique
  supervised tokens. Validation: 1,065 decisions, 871 calls, 273,159 tokens.
- Exact/4-gram screening and reviewed lexical candidates find no identified
  benchmark QA duplicate. Common public datasets and generic operations still
  overlap. For example, a source Titanic question asks the median after
  imputation, while CIBench asks the remaining missing-value count. This does
  not establish generalization to wholly unseen dataset families or rule out
  every possible semantic paraphrase.
- Train and evaluation expose exactly eight tools: `read`, `bash`, `edit`,
  `write`, `bash_job`, `finish`, `add_and_execute_jupyter_code_cell`,
  `final_answer`. The last two retain the public data's original semantics.
- Python runs in a persistent IPython kernel within a notebook. It shares files
  with the six actual Harness tools. Every new notebook gets a fresh Docker
  filesystem and kernel. Hidden benchmark notebooks, scores, model weights,
  home directories and credentials are not mounted.
- Thinking Off for training and generation, no compaction. Ordinary assistant
  explanation, function arguments (including descriptions) and EOS are labelled;
  system/user/real tool feedback and the supplied empty-thinking opener are
  masked. Each assistant target is supervised exactly once.

`encode_data.py` calls the production codec with the explicit
`published_messages` evidence contract. It does not pretend these are captured
teacher HTTP payloads. New builds capture tools through the complete archived
SGLang `_process_messages` entry. The historical inner-template capture added
`strict:false` and skipped router ordering; those old caches remain frozen.

## Training objective and recipe

The trainer is LF's native `run_exp` and DeepSpeed, with native Transformers token normalization. The candidate enables
`model_accepts_loss_kwargs` after LF attaches its processor. CPU DP4 GA8/GA2
controls pass; production Qwen/Liger/ZeRO3 admission is pending. For each real optimizer update:

```
objective = sum(shifted assistant CE across all DP ranks and GA micros)
            / sum(shifted supervised tokens across those same ranks/micros)
```

The candidate uses native `num_items_in_batch`, native distributed token counting,
and native loss scaling. Neither `get_batch_samples` nor `compute_loss` is
patched. Historical `normalization.py`, `audit_cpu_update.py`,
`audit_cuda_backbone.py` and `audit_runtime_identity.py` remain reference controls
for the frozen old run; they do not describe the candidate entry. Native online
Trainer loss is an update mean, not a corpus token-weighted mean. The candidate
does not emit historical `normalization-rank*.jsonl`; use native Trainer logs and
forward evidence, and evaluate frozen checkpoints separately.
Fixed-checkpoint validation uses all 194 unique rows and a global token sum.

Recipe: full language SFT, frozen vision, BF16 mixed compute with FP32 master and
Adam, ZeRO-3, activation checkpointing, FA2/Liger, LR `1e-5`, cosine, warmup 5%,
weight decay 0, two epochs, DP4/MB1/GA8/global32, seed/data seed 42. Window 262144,
packing off. Actual maximum input is 20916 tokens, so this does **not** claim a
real 256K activation-capacity test. All selected complete histories fit without
truncation. Expected native checkpoints: step 63 / epoch 1, step 126 / epoch 2.
Expected supervision exposure: 5,593,474 tokens; verify against actual rank IDs.

## Concrete deployment and entry points

Training host: `ruc-a800-18`, explicitly authorized GPUs 4–7.
M0 baseline and matched evaluation run on `ruc-a800-17` independently.

```
R=/ssd/chenkeyi/qwen35-ci-positive-control-20261108
T=/ssd/chenkeyi/qwen35-sft-772-20261101/env/train-r1
S=$R/source/training-frozen-r2/studies/ci_positive_control
```

The user explicitly authorized training in parallel with the frozen-M0 baseline.
The entry checks that policy, the original model identity, unchanged source/data,
live GPU availability, shared GPU locks, and native checkpoint state. Full M0
CIBench results remain required for the final comparison, not to begin training.

```bash
ssh ruc-a800-18 'systemctl --user status qwen35-ci-formal-train-20261108.service --no-pager'
ssh ruc-a800-18 'tail -n 20 /ssd/chenkeyi/qwen35-ci-positive-control-20261108/logs/formal-training.log'
```

Same-stage recovery explicitly adds
`--resume $R/runs/ci-e2b-2000-e2-r2/checkpoint-63` to that launcher and uses a new
service unit identity. It validates the complete native model, Adam, scheduler,
RNG and Trainer state, never a model-only HF export. Never restart a healthy
existing unit after a local disconnection.

The installed service automatically runs these separate export processes after
training completes. Do not invoke a second copy while that service is active:

```bash
PYTHONPATH=$R/source/training-frozen-r2 $T/bin/python $S/export_epoch.py $R 1 --device cuda:0
PYTHONPATH=$R/source/training-frozen-r2 $T/bin/python $S/export_epoch.py $R 2 --device cuda:0
```

They retain full native checkpoints and make separate HF snapshots, preserve the
checkpoint's native tokenizer/processor files, hash all files, then independently
load and generate on an explicitly authorized GPU. The installed causal-conv1d
fast path is CUDA-only; the original CPU smoke failed after training completed.
Use `--verify-existing` to validate that already exported immutable snapshot;
do not repeat training or overwrite weights. No native checkpoint metadata is
overwritten.

Serve M0/E1/E2 sequentially with the same frozen SGLang environment/parameters,
on legally free GPUs only. Use model IDs `qwen35-ci-m0`, `qwen35-ci-epoch1`,
`qwen35-ci-epoch2`, endpoint `127.0.0.1:18641`:

```bash
E=$R/source/evaluation-frozen-r8
bash $E/run_evaluation.sh $R epoch1 isolated-r3
bash $E/run_evaluation.sh $R epoch2 isolated-r3
$T/bin/python $S/score_validation.py $R epoch1
$T/bin/python $S/score_validation.py $R epoch2
```

The baseline runs in a detached user service. The existing
`qwen35-gpu-watch.timer` calls a read-only CI observer every 20 minutes. No new
GPU queue, provider, agent loop or scheduler is introduced. `observe.py` never
starts/stops training or inference. It deduplicates final and temporary outputs.

Host18 can use its existing validated `env/train-r1/bin/python` with `--python`;
the launcher does not reinstall packages or assume host17's venv path. Host/model/
data/config/source identities must be frozen again for the destination. Never
replace a venv interpreter with its resolved global-Python symlink target.

After an execution error, the existing Jupyter kernel-info readiness handshake
drains the interrupted request queue before another cell can be submitted.
Failure to recover readiness is infrastructure failure, not a fabricated result.

## Official evaluation boundaries

Pinned OpenCompass `6288ea0db61150b6c2c1e37c01808fef66748e8c`, CIBench
`5232d8cb145d2addc1c38d840b851f65368c424d`, Lagent
`2e469d7dd524b18dade138a54386c91b56ff5795`. Official all-prompt scope:
307 notebooks / 1818 questions / 26 library-language scopes, end-to-end
`infer_every`, no oracle history. The official CIReAct loop keeps its three-call
budget and first-successful-action question termination; 512 output tokens/call,
greedy T=0/top_p=1/top_k=-1, HTTP retry 0. All checkpoints share this protocol.

The adapter only handles native messages/functions and existing actions. Native
`isError=true` becomes an upstream API_ERROR with the original tool feedback, so
failed Bash commands do not falsely end a question as successful. The real exit-7
canary verifies recovery in the same kernel. Earlier partial outputs with the
incorrect status mapping are diagnostic, not primary baseline results. An
unknown function returns a real error observation; it is never guessed or aliased
to Python. The six terminal tools execute in the same isolated kernel environment.
The official IPython feedback, image capture and scoring operators remain intact.
The actual upstream IOPub inactivity timeout is 10s, distinct from the action's
120s timeout and HTTP 180s timeout. Record both, including resulting coverage.

Three tiny isolated upstream API/persistence repairs are provided as patches:
AgentOutputHandler inherits the existing ChatOutputHandler; CIBenchEvaluator uses
the current base output-directory API; intermediate saves write the same native
JSONL format that native restore reads. No scoring arithmetic or agent retry
budget changes. NLTK is fixed at 3.9.2 with the three official resource packages
cached along with the five required public Seaborn datasets in image
`sha256:f7f4237b989341a1a28acde581d818acbcec8a2ecb1d6411f930d9476f57b49d`.
Earlier host/unqualified-environment runs are diagnostic, not baseline scores.

DABStep public dev (10 questions) and the published author model are calibration,
not the primary score or a 75% acceptance threshold. Public context assets and
author weights are fixed and downloaded. Exact casefold answer agreement is
reported as a calibration diagnostic, not an official DABStep leaderboard score.
No answer key is sent to an agent or mounted in its environment.

After the primary SFT evaluation, lack of clear downstream improvement or a
large increase in unchanged repetition triggers the independent author TRL-path
control. That control must reconstruct from the same frozen raw messages/tools,
not reuse this cache as independent evidence, and match the target objective,
LR/order/token exposure before the same CIBench evaluation. This control is not
yet run, and no outcome is claimed before real training/evaluation completes.

The native-LF audit candidate also removes CI’s explicit `skip_logits` forward
partial: this study has no Trainer evaluation, and pinned Liger’s native
training default already selects fused loss. Bounded full Qwen4B DP4 GA8/GA2
controls used that default; native token-denominator differences were of the
same scale as one same-mode repeat pair. See `studies/native_lf/README.txt` for
raw error magnitudes, tokenizer identity, and Teacher-offload limitations.
# Tool projection must cross the full serving request entry

For new data builds, run `capture_server_projection.py` with the serving Python
environment against a new experiment root before `encode_data.py`. The capture
observes the tools actually passed to the tokenizer by `_process_messages`,
including `exclude_unset` and JSON key ordering. The old inner-template capture
added `strict=False` and missed router ordering. On two real original traces,
its encoded inputs matched 0/11 complete-router decision prefixes; the corrected
capture and actual encoder matched 11/11. Positive target token sequences stayed
identical. This is CPU input-contract evidence, not a new training/evaluation run
or proof of downstream causality.

The encoder checks the full-entry capture and projection file SHA and rejects
old or mismatched projection files. Rebuild into a new root. Keep frozen
experiments unchanged. This fix is separate from the new native LF recipe's
remaining LF-template versus SGLang-Jinja format difference.

`encode_data.py` always encodes the current input, saves its actual tokenizer via
native `save_pretrained` in `data/tokenizer`, and records class, package versions
and saved file hashes. The removed `--reuse-from` optimization accepted caches
from a tokenizer with different behavior: on real CI row7032, reuse matched only
3/7 complete-router prefixes; fresh encoding and archived serving reload of the
saved tokenizer matched 7/7. Positive targets stayed identical.

CI and Teacher tokenizers must not be assumed identical. On frozen CI row7032,
M0's TF5.8 Qwen2Tokenizer matches all seven old cache prefixes; the Teacher
snapshot matches three. In archived TF5.3, the native CI checkpoint tokenizer
also matches seven, whereas the old M0-restored serving assets match three.
New `export_epoch.py` therefore preserves checkpoint-native assets. Actual
before/after export assembly plus tokenizer loading passed the CPU regression;
the separate 4B generation smoke was excluded from that audit control.

The corrected legacy codec also completed all 8,391 rows/45,592 decision prefixes
against the complete archived frontend with its actual saved tokenizer. Positive
targets, non-system history, and all 514 replayed sampler denominators remain
identical; all target ranges shift by 36 tokens. This is CPU input-contract
evidence, separately from the new native LF data format and GPU update controls.
