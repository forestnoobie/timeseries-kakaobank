#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-config/train.yaml}"

# preprocess가 선행 — preprocessed CSV 없으면 생성
if [[ ! -f "data/data_preprocessed.csv" ]]; then
    python -m src.data.preprocess --config "$CONFIG"
fi

python -m src.train.run_train --config "$CONFIG"
