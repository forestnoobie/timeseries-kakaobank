"""
Top-level orchestrator — CLAUDE.md 디렉터리 트리.

`python run.py all` 한 번으로 EDA→preprocess→train→HPO→compare→predict까지 수행.
세부 단계는 `python run.py <stage>` 또는 scripts/run_*.sh 직접 호출.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("run")

ROOT = Path(__file__).resolve().parent


def sh(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
    if rc != 0:
        raise SystemExit(f"step failed (rc={rc}): {' '.join(cmd)}")


def stage_eda() -> None:
    sh(["bash", "scripts/run_eda.sh"])


def stage_preprocess() -> None:
    sh([sys.executable, "-m", "src.data.preprocess", "--config", "config/train.yaml"])


def stage_train() -> None:
    sh([sys.executable, "-m", "src.train.run_train", "--config", "config/train.yaml"])


def stage_predict() -> None:
    sh([sys.executable, "-m", "src.predict.run_predict", "--config", "config/train.yaml"])


def stage_hpo_optuna() -> None:
    sh([sys.executable, "-m", "src.tuning.optuna_runner",
        "--config", "config/hpo_optuna.yaml",
        "--train-config", "config/train.yaml"])


def stage_hpo_turbo() -> None:
    sh([sys.executable, "-m", "src.tuning.turbo_runner",
        "--config", "config/hpo_turbo.yaml",
        "--train-config", "config/train.yaml"])


def stage_hpo_compare() -> None:
    sh([sys.executable, "-m", "src.tuning.compare",
        "--turbo-config", "config/hpo_turbo.yaml",
        "--optuna-config", "config/hpo_optuna.yaml",
        "--train-config", "config/train.yaml",
        "--out-dir", "outputs/hpo/comparison"])


STAGES = {
    "eda":          stage_eda,
    "preprocess":   stage_preprocess,
    "train":        stage_train,
    "predict":      stage_predict,
    "hpo_optuna":   stage_hpo_optuna,
    "hpo_turbo":    stage_hpo_turbo,
    "hpo_compare":  stage_hpo_compare,
}

ORDER = [
    "eda", "preprocess", "train", "predict",
    "hpo_optuna", "hpo_turbo", "hpo_compare",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Project orchestrator")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["all", "all_no_hpo", *STAGES.keys()],
                        help="실행할 단계. 'all'은 전체, 'all_no_hpo'는 HPO 제외 빠른 경로.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.stage == "all":
        for s in ORDER:
            STAGES[s]()
    elif args.stage == "all_no_hpo":
        for s in ["eda", "preprocess", "train", "predict"]:
            STAGES[s]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
