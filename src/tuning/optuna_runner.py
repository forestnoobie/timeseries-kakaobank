"""
Optuna runner — CLAUDE.md §5-5.

TPESampler + MedianPruner + sqlite 영속화.
TuRBO와 동일 cv_objective + 동일 예산 (공정 비교).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from src.features.build import FeatureConfig, build_features, feature_matrix
from src.train.run_train import make_splits
from src.tuning.cv_objective import TrainData, cv_rmse_lightgbm
from src.tuning.param_spec import LGBM_SPACE, suggest_optuna

logger = logging.getLogger(__name__)


def _build_data(train_cfg: dict) -> TrainData:
    df = pd.read_csv(train_cfg["paths"]["preprocessed_csv"], parse_dates=["date"])
    fcfg = FeatureConfig.from_dict(train_cfg["features"])
    df = build_features(df, fcfg)
    X, y = feature_matrix(df, target="value")
    splits = make_splits(X.index, train_cfg["split"]["train_end"], train_cfg["split"]["val_end"])
    learn_idx = splits.train_idx.union(splits.val_idx)
    fixed = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }
    return TrainData(X=X.loc[learn_idx], y=y.loc[learn_idx],
                     fixed_params=fixed,
                     n_splits=int(train_cfg["cv"]["n_splits"]))


def main(optuna_cfg_path: str, train_cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    optuna_cfg = yaml.safe_load(Path(optuna_cfg_path).read_text())
    train_cfg = yaml.safe_load(Path(train_cfg_path).read_text())

    seed = int(optuna_cfg.get("seed", 42))
    np.random.seed(seed)

    data = _build_data(train_cfg)
    storage_path = Path(optuna_cfg["optuna"]["storage_path"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=str(optuna_cfg["optuna"]["study_name"]),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    t0 = time.time()
    out_csv = Path(optuna_cfg["output"]["trials_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_optuna(trial)
        score = cv_rmse_lightgbm(params, data)
        # Pruner support: 단일 metric만 reporting (CV 평균이라 step 분리 불가)
        return score

    n_trials = int(optuna_cfg["optuna"]["n_trials"])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # ── trial dump ─────────────────────────────────────────────────────
    df_trials = study.trials_dataframe()
    df_trials["elapsed_s"] = (df_trials["datetime_complete"] - df_trials["datetime_start"]).dt.total_seconds()
    df_trials.to_csv(out_csv, index=False)

    best = study.best_trial
    Path(optuna_cfg["output"]["best_params_json"]).write_text(
        json.dumps({"best_rmse": float(best.value), "params": best.params}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Optuna best rmse=%.3f  params=%s", best.value, best.params)
    logger.info("walltime: %.1fs", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hpo_optuna.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    args = parser.parse_args()
    main(args.config, args.train_config)
