#!/usr/bin/env bash
set -euo pipefail
ci_root=/ssd/chenkeyi/qwen35-ci-positive-control-20261108
ci_connection=$1
ci_name=qwen35-ci-kernel-$(basename "$ci_connection" .json)
ci_trainenv=/home/chenkeyi/workspace/RSI/runtime-evidence/qwen35-length-lf-audit-20261023/.venv
ci_image=$(cat "$ci_root/environment/IMAGE_ID")
exec docker run --rm --init --pull never --name "$ci_name" \
  --label chenkeyi.run=qwen35-ci-positive-control-20261108 \
  --label chenkeyi.phase="${CI_PHASE:?CI_PHASE must identify this evaluation}" \
  --network host --user 1002:1002 --hostname ci-kernel --cpus 4 --memory 32g \
  --pids-limit 512 --shm-size 1g --tmpfs /tmp:rw,nosuid,size=4g \
  --env HOME=/tmp --env IPYTHONDIR=/tmp/ipython --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --env TF_NUM_INTRAOP_THREADS=1 --env TF_NUM_INTEROP_THREADS=1 \
  --env HTTP_PROXY=http://127.0.0.1:7890 --env HTTPS_PROXY=http://127.0.0.1:7890 \
  --env NO_PROXY=127.0.0.1,localhost,raw.githubusercontent.com \
  --env no_proxy=127.0.0.1,localhost,raw.githubusercontent.com \
  --env PYTHONPATH="$ci_root/eval-env/lib/python3.11/site-packages:$ci_trainenv/lib/python3.11/site-packages:/ci-runtime" \
  --mount type=bind,src="$ci_root/eval-env",dst="$ci_root/eval-env",readonly \
  --mount type=bind,src="$ci_trainenv",dst="$ci_trainenv",readonly \
  --mount type=bind,src="$ci_connection",dst=/ci-connection.json,readonly \
  --mount type=bind,src="$ci_root/source/open_harness_ai-0.3.0+rsi.summary.20261030-py3-none-any.whl",dst=/ci-runtime/open_harness_ai.whl,readonly \
  --mount type=bind,src="$ci_root/source/kernel-frozen/kernel_tools.py",dst=/ci-runtime/kernel_tools.py,readonly \
  "$ci_image" /usr/local/bin/python -m ipykernel_launcher -f /ci-connection.json
