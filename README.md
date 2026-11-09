# CalibForge High: 1,000-task SFT collection

Public synthetic trajectories collected with DeepSeek Thinking High and the six tools `read`, `bash`, `edit`, `write`, `bash_job`, and `finish`.

**1,000 is the number of attempted tasks, not the number of successful training trajectories.** The source collection has 932 recorded tasks, 704 reward-1 outcomes, 214 reward-0 outcomes and 82 unscored outcomes. Completeness and text-only filtering leaves 691 trajectories. One trajectory exceeds 262,144 tokens, so the frozen train/holdout split has 655/35 trajectories. Any additional credential-bearing exclusions are listed in the manifest.

## Download

[Download the complete data + tokenizer + template package](https://github.com/CoyeCHEN/calibforge-high-1000-sft-data/releases/download/data-v1/calibforge-high-1000-sft-data-v1.tar.gz)

```bash
curl -L -o calibforge-high-1000-sft-data-v1.tar.gz https://github.com/CoyeCHEN/calibforge-high-1000-sft-data/releases/download/data-v1/calibforge-high-1000-sft-data-v1.tar.gz
tar -xzf calibforge-high-1000-sft-data-v1.tar.gz
```

Published rows: full trajectories **674**, train **638**, holdout **35**. The package includes the longer raw trajectory; the processed split excludes it. No trajectory is truncated.

## Package files

- `data/trajectories.jsonl`: one JSON object per trajectory, with `task_id`, `trial_uuid`, `input`, and `tools`. Assistant messages retain `reasoning_content`, ordinary content and tool calls; real tool feedback is retained.
- `data/train.processed.jsonl` and `data/holdout.processed.jsonl`: frozen SWE-Master preprocessing output (`prompt`, `response`, `prompt_ids_len`, `response_ranges`, identity and token count). These files are not a generic LLaMA-Factory dataset format.
- `tokenizer/`: native tokenizer snapshot saved with Transformers 5.8.0; contains the actual official Qwen3.5 Jinja template. The bundled backend is selected to reproduce the frozen preprocessing counts. Use this snapshot with the pinned versions; a model-default AutoTokenizer can segment Unicode histories differently even with the same Jinja.
- `preprocessing/sft_data_pre_tokenize_toolcall.py`: the preprocessing source used to produce the frozen split. Its training-library dependencies come from the cited SWE-Master/OpenRLHF environment.
- `metadata/`: counts, split membership, source revisions and checksums.
- `PACKAGE_MANIFEST.json`: per-file sizes and SHA-256 hashes.

## Read raw messages

```python
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast

tok = AutoTokenizer.from_pretrained("tokenizer", local_files_only=True)
with open("data/trajectories.jsonl") as f:
    row = json.loads(next(f))
text = tok.apply_chat_template(row["input"], tools=row["tools"], tokenize=False)
```

Install `requirements-tokenizer.txt` for the pinned tokenizer versions. The bundled tokenizer was reloaded and checked against the rendered text, token count and valid response ranges of all 673 published processed rows. Token counts follow the native SWE-Master loading rule: strip final newlines and append a space plus tokenizer.eos_token when absent. This is a CPU data check, not a model-training or downstream-quality claim.

## Credential handling

Only messages/tools, processed data and explicitly selected tokenizer/source/license metadata are packaged. Provider environment files, credential-bearing HTTP request dumps, raw recorder journals, model weights and runtime logs are outside this distribution. Published text was scanned for supplied credentials and common credential formats. Candidate-bearing tasks, if found, are excluded consistently from raw and processed public files; private originals are unchanged. Candidates can include synthetic fixtures from security tasks; no assertion is made that they were live credentials. See `metadata/SECURITY_SCAN.json`.

## Attribution and license

The benchmark is [CalibForge](https://huggingface.co/datasets/AweAI-Team/CalibForge) by AweAI-Team, revision `c2fae4c1f536461142b2a89a8b9423a8bc376088`. Tokenizer assets originate from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Preprocessing derives from [SWE-Master](https://github.com/RUCAIBox/SWE-Master/tree/bc8904f44d62f10b5eb16189b22eca31e20cd48e).

Generated trajectory data and packaging are shared under CC BY 4.0. Third-party assets retain their upstream licenses in `LICENSES/`. Collection and packaging: CoyeCHEN.

中文：这是 CalibForge High 的1000任务采集包；1000不等于成功训练条数。包含完整 messages/tools、原生Qwen template/tokenizer及655/35的冻结预处理划分。公开包的实际数量、排除项与SHA见 metadata。
