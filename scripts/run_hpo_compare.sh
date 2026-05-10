#!/usr/bin/env bash
# Run TuRBO + Optuna trial CSVs를 모아 비교 리포트 생성.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.tuning.compare \
    --turbo-config config/hpo_turbo.yaml \
    --optuna-config config/hpo_optuna.yaml \
    --train-config config/train.yaml \
    --out-dir outputs/hpo/comparison
