#!/usr/bin/env bash
set -euo pipefail
export DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=
reproduce_output="${1:?Pass a new output directory}"
mkdir "$reproduce_output"
python scripts/export_from_raw.py --package-root . --output "$reproduce_output/sharegpt" --check-against data
python scripts/tokenize_native.py --package-root . --output "$reproduce_output/native-check"
