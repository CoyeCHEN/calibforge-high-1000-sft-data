Native LF candidate — full CPU data and bounded Qwen4B GPU update controls

NEW INPUT-CONTRACT FINDING:
Historical CI capture called only SGLang's inner Jinja method. The full request
router also orders tool JSON keys and excludes unset defaults. Using its actual
logical eight-tool request, two old encoded traces match0/11 router decision
prefixes; new full-entry capture plus the actual encoder match11/11. Positive
target token vectors remain identical. New legacy CI builds require a fresh
full-entry capture and verified projection file; old artifacts remain unchanged.
The full8391 native data controls below used the previously frozen tool schema.
They still establish field/label/collator behavior for those exact inputs, but
are not a rebuilt corrected-schema corpus or corrected-data GPU admission.
The native LF tool-format difference below is separate from this capture error.

Pinned baseline: official LLaMA-Factory c683e59edd8f893abc5e1403184a1d0644817d2a,
Transformers5.8.0, Accelerate1.11.0. The clean source was not edited.
Use its native YAML entry; pinned src/train.py's JSON-file branch fails because
it calls json.load(Path). Existing project run_exp(dict) was unaffected.

CI whole trajectories:
  python3 /absolute/path/export_teacher_lf.py CI-raw-messages.jsonl /new/ci-data \
    --ci-union-null-padding
  torchrun --nproc_per_node=4 src/train.py /absolute/path/ci-native.yaml \
    dataset_dir=/new/ci-data output_dir=/absolute/path/new-output
Run from the clean LF checkout. The preferred data path uses the same thin
ShareGPT exporter as Teacher. The CI-only flag removes absent argument keys
padded with null by the published union schema; it must not be used for Teacher
provider payloads. All45592 real CI tool calls round-trip exactly through the
archived serving parser with their actual tool schemas after this cleanup.
All8391 native rows retain tool-call accompanying prose. Train/validation
supervised totals are11452337/274224, versus old11407810/273159.
The first target differs at token101: LF newline198 versus old double-newline271;
all positive target vectors differ. This is not an old-label equivalence claim.

The earlier raw-OpenAI control also encoded all8197train+194validation rows,
but dropped accompanying prose and separate reasoning_content. That was a
negative migration control, not the recommended data format. Its historical
TokenizersBackend differs from M0's default Qwen2Tokenizer under TF5.8: all-corpus
controls changed1 CI and27 MAX histories, while positive target sequences stayed
identical. Re-encoding the OpenAI control at cutoff262144 versus32768 was exact;
preferred ShareGPT max length31980 also fits the CI recipe's32768 cutoff.
The preferred ShareGPT path passed CPU encoding/collator/parser checks; its exact
data path has not run a production GPU update. The completed clean GPU CLI used
the earlier OpenAI control.

Teacher one real provider decision per row, preserve body and Thinking:
  python /absolute/path/export_teacher_lf.py T_MAX-raw-decisions.jsonl /new/max-data
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=4 src/train.py /absolute/path/teacher-decision-native.yaml \
    dataset_dir=/new/max-data output_dir=/new/max-output \
    deepspeed=/absolute/path/teacher-native-ds.json
Use the same commands with OFF raw decisions and new paths. Input JSONL has
messages=actual history plus current assistant, tools=list or serialized JSON.
Recorder private blocks first need their existing raw interchange reader.
The exporter writes native ShareGPT conversations and dataset_info.json. Tool
arguments stay in JSON strings in LF's native function slot, avoiding Arrow's
chunk-dependent nested struct inference. LF handles function serialization,
Thinking formatting, tokenization, mask_history, labels and padding.

Full native OpenAI JSON ingestion failed on real dynamic arguments. A two-real-row regression fails before conversion and loads after ShareGPT conversion.
The ShareGPT path passed all5201 decisions from64paired train tasks/arm. An
independent official Qwen Jinja reconstruction matches every old decision input/
labels SHA; every new target begins with the actual Thinking/body field prefix.
This removes the need for teacher_segments/project Qwen tokenization and prefix-
compatible merging from the NEW data path. Old files remain for frozen studies.

The native labels/input tensors still differ from historical ones: empty-think
opener supervision, end newline, BPE boundaries and tool formatting change.
On the first real CI prompt, native LF and the tokenizer's saved original Jinja
differ at token20: LF serializes flattened tools, while Jinja keeps the outer
type/function object. Identical system newline handling does not remove this
difference. Native labels are not automatically the current serving-prefix
contract; that alignment remains a migration acceptance item.
One decision per row also changes MAX optimizer steps12->70; OFF stays94 under
DP4/GA8 with sampler padding. This is a training design choice, not an equivalence
claim. Teacher cutoff stays262144: actual decision histories exceed128K.
The recipe uses native ZeRO3 CPU offload with three historical config values:
max_live_parameters=50M, max_reuse_distance=0, zero_force_ds_cpu_optimizer=false.
These keep the tested cache/torch CPU AdamW path. They are config values, not
optimizer code. Removing them and using the official LF example would also
let Accelerate map AdamW to DeepSpeedCPUAdam; that alternative is untested.
With unchanged real MAX/OFF longest+median rows, no saved-CPU hook, same old DS,
and the requested expandable allocator, one DP4 partial-GA1 update completed;
max torch allocated74.61GiB, train interval181.98s. The no-override attempt OOM.
This is capacity evidence for four rows, not full GA8/epoch/validation admission.
The allocator value was preserved in the executed launch command, not dumped
from the exited process; see the audit provenance supplement.

LOSS OBJECTIVE IS A SEPARATE CONTROL:
The historical objective is global shifted-supervised-token mean per update.
Clean LF disables model_accepts_loss_kwargs when a processor is attached. Thus
these clean YAML recipes use LF's default loss weighting, not proof of equality
to that historical token objective. average_tokens_across_devices alone cannot
undo the guard. The candidate CI entry uses token_mean.py to enable native HF
num_items_in_batch without replacing get_batch_samples/compute_loss. CPU DP4
GA8/GA2 passes. Bounded full Qwen4B/BF16/ZeRO3 with the unchanged CI tensors
compared native/wrapper/independent PyTorch CE: GA8 preclip/update relative
differences0.00540/0.00411 versus native repeat0.00520/0.00393; isolated GA2
0.00800/0.00528 versus repeat0.00758/0.00500. These are observed repeat-scale
differences, not bitwise or statistical equivalence. Teacher offload is not
covered. teacher-token-mean.patch is the equivalent change
for a COPY of the frozen external r9 entry; it sets the same native flag inline
and does not import a helper into the frozen source repository. It was not
applied to frozen source.

The clean original src/train.py also completed a four-A800 Qwen4B two-update
run on40 raw CI trajectories and saved a final model. This uses new native
OpenAI targets/default loss, separately from the frozen-tensor controls.
CI no longer installs skip_logits partial: official Liger defaults to loss-only
when training labels are present; CI evaluation stays disabled.

Deleted: six phase-timing patches and a duplicate native-save event. Remaining
identity/durability hooks and per-rank metrics have acceptance consumers in the
legacy candidate. The clean YAML paths do not install those custom callbacks.
Historical normalization.py/audit_* scripts remain old-run reference controls.
No GPU memory/speed benefit from removals is claimed without paired measurement.

Reports, scripts, per-decision values and independent review:
/home/cky/workspace/RSI/audits/native-lf-simplification-20261109/
No old checkpoint, frozen source, stopped evaluation, or remote branch changed.

TOKENIZER IDENTITY CORRECTION:
The snapshot/original class census above compares two candidate encodings; it
does not identify the historical CI data tokenizer. Frozen CI row7032 matches
TF5.8 M0 Qwen2Tokenizer on7/7 decision prefixes, the Teacher snapshot on3/7.
In the archived TF5.3 serving environment the native CI checkpoint matches7/7,
the old M0-restored serving assets3/7. The earlier claim that CI restoration was
necessary is withdrawn. New CI export preserves native checkpoint assets; the
old immutable export is untouched. Real export assembly and CPU token consumption
pass the before/after regression; 4B generation was deliberately not run.
The legacy candidate also deletes its custom cache-reuse branch and saves the
actual tokenizer with save_pretrained. With the same current M0 tokenizer,
reuse silently accepted a different tokenizer's cache (real7032 prefixes3/7),
while fresh encoding and serving reload of the saved snapshot match7/7.

Teacher's actual cached3362 training rows and5201 real decisions separately pass
the complete archived serving frontend on CPU, including all masks and prefixes.
The node18 replica was bound by all23 file hashes to actual node17 training caches.
Independent review checked all3362 masks/member coverage and reran4 key cached
prefixes. No claim of historical live-worker or production GPU verification.

The corrected legacy CI codec now also passed the full CPU contract using the
actual CI M0-default tokenizer: all8391 cache rows/all45592 complete-frontend
prefixes, identical positive targets and history after the first system segment,
all supervised ranges shifted36, all514 sampler denominators unchanged. This is
separate from the native LF ShareGPT data/template controls, and is not a new
corrected-data GPU forward or a causal regression experiment.

The official LF SGLangEngine input_ids interface was separately tested offline
on the first real native CI row. With the candidate preserve_thinking=False,
only1/6 native inference prefixes matches native SFT; historical empty-thinking
markers are removed. True matches6/6 when the supplied native aligned history
explicitly contains those markers, with both modes' training tensors unchanged.
True does not add missing client markers or fix nonempty CoT history generally.
No production config was changed based on this scoped inference control; it does
not establish LF OpenAI API or live backend compatibility.

LIGER COMPONENT CONTROL (optional candidate, GPU pending):
The matched full-Qwen normal window changed preclip gradients8.26% when only
ordinary RMSNorm was switched, versus0.52% same-mode repeat; SwiGLU was0.54%.
No kernel bug or downstream cause is established. Native Transformers can keep
only fused CE through use_liger_kernel/liger_kernel_config; no custom kernel
wrapper is needed. liger-ce-only.patch expresses that with native YAML fields.
CPU Trainer.train binding kept HF RMSNorm/MLP and installed only Liger forward,
but stopped before forward. Its four-GPU CLI test was refused because another
job occupied the GPUs. Do not mark this optional patch GPU-admitted.

TOKENIZER EXPORT:
New native data uses M0's default Qwen2Tokenizer under TF5.8; full original-vs-
historical-class comparisons changed1 CI and27 MAX histories, with all positive
target token sequences unchanged. The native saved tokenizer correctly round-
trips those real Unicode feedback texts in the archived serving TF5.3 runtime.
The old CI export repair restores M0 JSON to match the historical preencoded
cache. It is necessary for that old path but must not be transplanted into this
new native path: preserve the tokenizer that native training actually saved.
