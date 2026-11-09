#!/usr/bin/env bash
set -euo pipefail
ci_root=/ssd/chenkeyi/qwen35-ci-positive-control-20261108
ci_trainenv=/home/chenkeyi/workspace/RSI/runtime-evidence/qwen35-length-lf-audit-20261023/.venv
exec timeout 180s docker run --rm --name qwen35-ci-source-replay-20261108-r2 \
  --label chenkeyi.run=qwen35-ci-positive-control-20261108 \
  --network none --read-only --cpus 1 --memory 4g --pids-limit 128 --user 1002:1002 \
  --tmpfs /tmp:rw,nosuid,size=512m --env HOME=/tmp --env IPYTHONDIR=/tmp/ipython \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env CUDA_VISIBLE_DEVICES= \
  --env PYTHONPATH="$ci_root/eval-env/lib/python3.11/site-packages:$ci_trainenv/lib/python3.11/site-packages" \
  --mount type=bind,src="$ci_root/eval-env",dst="$ci_root/eval-env",readonly \
  --mount type=bind,src="$ci_trainenv",dst="$ci_trainenv",readonly \
  --mount type=bind,src="$ci_root/data/replay-input",dst=/home/user/input,readonly \
  --mount type=bind,src="$ci_root/data/replay-traces.jsonl",dst=/traces.jsonl,readonly \
  --mount type=bind,src="$ci_root/source/sft/studies/ci_positive_control/replay_cells.py",dst=/replay.py,readonly \
  --mount type=bind,src="$ci_root/checks/replay-work",dst=/workspace --workdir /workspace \
  python:3.11-slim /usr/local/bin/python /replay.py
