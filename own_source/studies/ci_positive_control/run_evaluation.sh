#!/usr/bin/env bash
set -euo pipefail
root=$1
label=$2
edition=$3
export CI_ROOT="$root"
export CI_PHASE="cibench-$label-$edition"
export CI_EXECUTION_WORKDIR="$root/runtime/terminal-$label-$edition"
export CI_MODEL_REQUEST_LOG="$root/logs/requests-$label-$edition.jsonl"
export PATH="$root/eval-env/bin:$PATH"
ci_code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ci_code"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
mkdir -p "$CI_EXECUTION_WORKDIR"
cd "$root/source/opencompass-6288ea0db61150b6c2c1e37c01808fef66748e8c"
args=(run.py "$root/configs/cibench_${label}_${edition}.py" --debug)
if [[ -n ${4:-} ]]; then args+=(--reuse "$4"); fi
"$root/eval-env/bin/python" "${args[@]}"
"$root/eval-env/bin/python" "$ci_code/collect_evaluation.py" "$root" "$label" "$edition"
