"""
TuRBO runner — CLAUDE.md §5-4.

Uber-Research/TuRBO (NeurIPS 2019). Uber Non-Commercial 라이선스.
n_init >= 2d 권장 (CLAUDE.md): 9차원 → 20.
영속화 부재: 매 trial CSV flush.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.features.build import FeatureConfig, build_features, feature_matrix
from src.train.run_train import make_splits
from src.tuning.cv_objective import TrainData, cv_rmse_lightgbm
from src.tuning.param_spec import LGBM_SPACE, coerce_types, decode_unit_vector, dim

logger = logging.getLogger(__name__)


def _build_data(train_cfg: dict) -> TrainData:
    df = pd.read_csv(train_cfg["paths"]["preprocessed_csv"], parse_dates=["date"])
    fcfg = FeatureConfig.from_dict(train_cfg["features"])
    df = build_features(df, fcfg)
    X, y = feature_matrix(df, target="value")
    splits = make_splits(X.index, train_cfg["split"]["train_end"], train_cfg["split"]["val_end"])
    # HPO는 train+val 영역만 사용 (test는 최종 평가용 — leakage 방지).
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


def main(turbo_cfg_path: str, train_cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    turbo_cfg = yaml.safe_load(Path(turbo_cfg_path).read_text())
    train_cfg = yaml.safe_load(Path(train_cfg_path).read_text())

    seed = int(turbo_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = _build_data(train_cfg)
    d = dim()
    logger.info("TuRBO dim=%d, n=%d", d, len(data.y))

    # ── trial 기록 + flush ────────────────────────────────────────────
    out_csv = Path(turbo_cfg["output"]["trials_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trial_records: list[dict] = []

    def write_csv() -> None:
        pd.DataFrame(trial_records).to_csv(out_csv, index=False)

    eval_count = {"n": 0}
    t0 = time.time()

    def objective(u: np.ndarray) -> float:
        # TuRBO는 다항-batch로 호출 가능 → u shape이 (d,)인지 (b,d)인지 분기.
        u = np.atleast_2d(u)
        scores: list[float] = []
        for row in u:
            params = decode_unit_vector(row)
            score = cv_rmse_lightgbm(params, data)
            scores.append(score)
            eval_count["n"] += 1
            trial_records.append({
                "trial": eval_count["n"],
                "elapsed_s": round(time.time() - t0, 2),
                "rmse": score,
                **params,
            })
            write_csv()  # 영속화 (CLAUDE.md §5-4 주의사항)
            logger.info("[turbo] trial %3d | rmse=%.3f | %s",
                        eval_count["n"], score, params)
        return np.asarray(scores).reshape(-1, 1) if u.shape[0] > 1 else float(scores[0])

    from turbo import Turbo1  # type: ignore[import-not-found]

    n_init = int(turbo_cfg["turbo"]["n_init"])
    max_evals = int(turbo_cfg["turbo"]["max_evals"])
    if n_init < 2 * d:
        logger.warning("n_init=%d < 2d=%d — bumping to 2d", n_init, 2 * d)
        n_init = 2 * d
    batch_size = int(turbo_cfg["turbo"]["batch_size"])

    turbo = Turbo1(
        f=objective,
        lb=np.zeros(d),
        ub=np.ones(d),
        n_init=n_init,
        max_evals=max_evals,
        batch_size=batch_size,
        verbose=True,
        use_ard=bool(turbo_cfg["turbo"].get("use_ard", True)),
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device=str(turbo_cfg["turbo"].get("device", "cpu")),
        dtype=str(turbo_cfg["turbo"].get("dtype", "float64")),
    )
    turbo.optimize()

    write_csv()

    # ── best params 저장 ─────────────────────────────────────────────
    df_trials = pd.DataFrame(trial_records).sort_values("rmse")
    best = df_trials.iloc[0]
    best_raw = {k: best[k] for k in df_trials.columns
                if k not in ("trial", "elapsed_s", "rmse")}
    # pandas 라운드트립으로 int → float 변환된 부분 복구
    best_params = coerce_types(best_raw)
    Path(turbo_cfg["output"]["best_params_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(turbo_cfg["output"]["best_params_json"]).write_text(
        json.dumps({"best_rmse": float(best["rmse"]), "params": best_params}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("TuRBO best rmse=%.3f params=%s", best["rmse"], best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hpo_turbo.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    args = parser.parse_args()
    main(args.config, args.train_config)
