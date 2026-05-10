#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.predict.run_predict --config "${1:-config/train.yaml}"
