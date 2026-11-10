# CalibForge High — own upstream-native LF debug package

**Current source baseline: `03074a83743c70394d59b4e17b07a0b122495e54` (`study/sds-fixed-legacy-20261110`), freezing the upstream simplification plus the later CI fixes. The active CalibForge training recipe is now five epochs.** The earlier mismatched SWE-Master release is withdrawn as a draft.

[Download the complete debug package](https://github.com/CoyeCHEN/calibforge-high-1000-sft-data/releases/download/native-lf-v2/calibforge-high-native-lf-debug-v2.tar.gz)

```bash
curl -L -o calibforge-high-native-lf-debug-v2.tar.gz https://github.com/CoyeCHEN/calibforge-high-1000-sft-data/releases/download/native-lf-v2/calibforge-high-native-lf-debug-v2.tar.gz
tar -xzf calibforge-high-native-lf-debug-v2.tar.gz
cd calibforge-high-native-lf-debug-v2
```

## What is included

- `raw_data/qualified.jsonl`: **674** complete raw messages/tools trajectories from the CalibForge Thinking High 1,000-task collection, with Thinking, prose, calls and feedback. 17 credential/private-key candidate trajectories are omitted from the public copy; originals are unchanged.
- `data/train.sharegpt.jsonl` / `validation.sharegpt.jsonl`: **638 / 35** trajectories converted by the unchanged own `export_teacher_lf.py` thin adapter. The extra raw trajectory exceeds the training length cap and is not in either split. Frozen task membership is retained; SWE-Master prompt/response/ranges are not used.
- `scripts/export_from_raw.py`: runnable raw-to-ShareGPT adapter. It stops after the last real assistant decision, omitting only trailing tool acknowledgments; no assistant target or preceding history is removed. Without this input-boundary step, native LF rejects529 rows and the first check fails (134/10 rows instead of638/35). See `metadata/TERMINAL_FEEDBACK_REGRESSION.json` and `EXPORT_REPORT.json`.
- `configs/native-text.env`: LF's native environment interface gives image/video/audio placeholders unused spellings for this text-only corpus. Otherwise literal `<image>` markup in real tool output causes a missing-image error. Text, tokenization rules and upstream LF source are unchanged; see `metadata/MEDIA_PLACEHOLDER_REGRESSION.json`.
- `data/lf-cache/`: actual Hugging Face dataset cache produced by **unmodified LF native preprocessing**, containing `input_ids`, `labels`, and `attention_mask`.
- `own_source/`: **122 files** matching the exact committed baseline `03074a83743c70394d59b4e17b07a0b122495e54` (parent `7d78bf219410e37f0bd36547263c2c7c694aada6`). `metadata/OWN_SOURCE_MANIFEST.json` fixes every included source hash; `OWN_CHANGES.patch` is the real diff.
- `framework/LLaMAFactory/src/`: unchanged pinned upstream LF source `c683e59edd8f893abc5e1403184a1d0644817d2a`. Its license is included.
- `model_assets/`: the actual official Qwen3.5-4B tokenizer/config/Jinja assets, without model weights. `runtime-tokenizer/` is the tokenizer saved after native LF template setup.
- `configs/`, `scripts/tokenize_native.py`, `REPRODUCE_CPU.sh`, and `metadata/`: native recipe, reproducible CPU entry, per-row input/label hashes, versions, counts and original Recorder identities.

**The entire original Recorder directory is not uploaded.** `RAW_RECORDER_INDEX.json` preserves identities/hashes for the published raw messages. Raw provider/runtime logs and environment configurations are outside this distribution.

## Baseline and five-epoch recipe

The previous package's 122 own-source files already match this fixed source commit byte for byte. This refresh makes that identity explicit, replaces the source diff with the exact parent-to-commit diff, and changes `num_train_epochs` from **1 to 5**. The new output path is `outputs/calibforge-high-native-e5`.

With the declared DP4 / microbatch1 / accumulation8 setup, 638 training rows imply **20 optimizer steps per epoch and 100 across five epochs**, including native sampler padding to 640 sampled rows per epoch. These are expected counts, not completed-run evidence. LR `1e-5`, warmup0.05, scheduler, kernels, DeepSpeed configuration, data split and token/label contents retain the previous recipe. Adam betas/epsilon, weight decay, gradient clipping, checkpoint contents, final-checkpoint selection and no-drop-tail behavior are now explicit in YAML at their existing locked defaults.

`metadata/BASELINE_03074a8.json` records the exact source, active entry and recipe. `metadata/TRAIN_CONFIG_AUDIT.json` lists every configuration field before and after the refresh. The fixed CI tool projection, cache-identity and tokenizer-export code is included for review; this CalibForge recipe still uses the documented thin ShareGPT exporter and native LF CLI.

The existing CPU data receipts remain valid for the unchanged tensors. This refresh has not executed a five-epoch GPU training or downstream evaluation. `historical_configs/` records earlier runs and is kept unchanged. The immutable `native-lf-v1` release remains a historical download; use `native-lf-v2` for this recipe.

Run `python scripts/check_package.py` before training to verify all packaged file hashes, the source snapshot, and the five-epoch recipe. Test answers, evaluation outputs and credentials are not added by this refresh.

## Actual processing path

```
raw messages/tools -> own_source/studies/native_lf/export_teacher_lf.py
                  -> native ShareGPT loader -> LF qwen3_5 formatter
                  -> native SFT processor/collator -> native Trainer/DeepSpeed
```

There is no call to SWE-Master, `teacher_segments.py`, a custom label builder in this active path. Older own-code files are included only so a reviewer can inspect the real simplification diff; they are not imported by the native training entry.

For this whole-trajectory dataset, `mask_history=false`, `enable_thinking=true`, `preserve_thinking=true`, cutoff262144 and packing=false. Every assistant target is supervised once. This is a dataset/configuration variant of the same native branch; the branch's separate Teacher A/B example uses one decision per row.

## Template distinction for debugging

The **training template is the registered LF `qwen3_5` ReasoningTemplate**, in `framework/LLaMAFactory/src/llamafactory/data/template.py`, together with `formatter.py` and `tool_utils.py`. `model_assets/chat_template.jinja` is the tokenizer's official Qwen Jinja. Those two formatting paths are not interchangeable; both are included deliberately.

## Reproduce / train

Use the pinned versions recorded in `requirements-observed.txt` with the included LF source. From the extracted package:

```bash
export PYTHONPATH="$PWD/framework/LLaMAFactory/src${PYTHONPATH:+:$PYTHONPATH}"
bash REPRODUCE_CPU.sh /absolute/path/to/new-native-data-check
source configs/native-text.env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=4 framework/LLaMAFactory/src/train.py configs/train-native.yaml
```

CPU reproduction passed on **673 rows**: native cache equals an uncapped native LF reference; all targets and label masks are valid and no truncation occurred. Input tokens/train=25330274, validation=1748018; supervised tokens/train=14357023, validation=1025918. This is CPU data evidence, not a production GPU forward/update/save or downstream-quality claim. No evaluation was restarted.

## Provenance and privacy

The raw benchmark is [CalibForge](https://huggingface.co/datasets/AweAI-Team/CalibForge), revision `c2fae4c1f536461142b2a89a8b9423a8bc376088`. Raw collection: 1,000 tasks, 932 recorded trajectories; 704 reward1, 214 reward0, 82 unscored. Completeness/text filtering yielded691; the public credential-filtered subset is674. Tokenizer assets: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.

The public files are scanned for supplied credentials and common credential/private-key formats. No credential values, raw environment files or model weights are included. Candidate matching does not assert that a security-task fixture was a live credential. Data/packaging use CC BY4.0; upstream assets retain their included licenses.

中文：这次交付是自己的回归上游精简版，不是师兄的SWE-Master处理。原始输入、原生LF处理后cache、模板与对应源码均可核对。1000是采集任务数；公开训练638条、留出35条，另有1条超长原始记录。完整原生Recorder目录未整体上传，索引中给出原件身份。
