"""
Shared CV objective — CLAUDE.md §5-3.

TuRBO와 Optuna *모두 동일한 함수*로 평가. 공정 비교의 핵심.

TimeSeriesSplit(n_splits=N) 평균 RMSE 반환.
seed 고정 → 결정론적 (TuRBO의 noise-free 가정 충족).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import rmse
from src.tuning.param_spec import coerce_types

logger = logging.getLogger(__name__)


@dataclass
class TrainData:
    X: pd.DataFrame
    y: pd.Series
    fixed_params: dict[str, Any]   # 고정값 (objective, metric, verbose 등)
    n_splits: int = 5
    seed: int = 42


def cv_rmse_lightgbm(params: dict[str, Any], data: TrainData) -> float:
    """LightGBM CV — 평균 fold RMSE.

    탐색 파라미터 + fixed_params 병합하여 학습.
    """
    import lightgbm as lgb  # type: ignore[import-not-found]

    # 정수 파라미터 보호 — LightGBM은 num_leaves=199.0 같은 float을 거부.
    params = coerce_types(params)
    merged = {**data.fixed_params, **params}
    n_estimators = merged.pop("n_estimators", 1000)
    early_stopping = merged.pop("early_stopping_rounds", 50)

    tscv = TimeSeriesSplit(n_splits=data.n_splits)
    fold_rmses: list[float] = []

    X_arr = data.X.values
    y_arr = data.y.values

    for tr, va in tscv.split(X_arr):
        X_tr, y_tr = X_arr[tr], y_arr[tr]
        X_va, y_va = X_arr[va], y_arr[va]
        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            random_state=data.seed,
            **merged,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(int(early_stopping), verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = model.predict(X_va)
        fold_rmses.append(rmse(y_va, pred))

    score = float(np.mean(fold_rmses))
    return score


__all__ = ["TrainData", "cv_rmse_lightgbm"]
