#!/usr/bin/env bash
# TuRBO HPO — Uber-Research/TuRBO (NeurIPS 2019). Uber Non-Commercial 라이선스.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f "data/data_preprocessed.csv" ]]; then
    python -m src.data.preprocess --config config/train.yaml
fi

python -m src.tuning.turbo_runner \
    --config config/hpo_turbo.yaml \
    --train-config config/train.yaml
