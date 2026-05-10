#!/usr/bin/env bash
# CLAUDE.md §"Common commands" — EDA 노트북을 결정론적으로 재실행.
# Docker 안/밖에서 동일하게 동작하도록 작성.
set -euo pipefail

cd "$(dirname "$0")/.."

NOTEBOOK="outputs/analytics/notebook.ipynb"
BUILDER="outputs/analytics/_build_notebook.py"

if [[ ! -f "$NOTEBOOK" ]]; then
    if [[ -f "$BUILDER" ]]; then
        echo "[run_eda] notebook missing — regenerating from $BUILDER"
        python "$BUILDER"
    else
        echo "[run_eda] notebook not found: $NOTEBOOK" >&2
        exit 1
    fi
fi

# data/raw/dataset.csv 보장 (CLAUDE.md §1 경로 규약)
if [[ ! -f "data/raw/dataset.csv" ]]; then
    mkdir -p data/raw
    if [[ -f "dataset.csv" ]]; then
        cp dataset.csv data/raw/dataset.csv
        echo "[run_eda] seeded data/raw/dataset.csv from repo root"
    else
        echo "[run_eda] dataset.csv not found at repo root or data/raw/" >&2
        exit 2
    fi
fi

echo "[run_eda] executing $NOTEBOOK"
jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --inplace \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3

echo "[run_eda] done — see outputs/analytics/"
