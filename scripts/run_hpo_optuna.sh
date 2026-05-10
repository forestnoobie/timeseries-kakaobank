#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f "data/data_preprocessed.csv" ]]; then
    python -m src.data.preprocess --config config/train.yaml
fi

python -m src.tuning.optuna_runner \
    --config config/hpo_optuna.yaml \
    --train-config config/train.yaml
