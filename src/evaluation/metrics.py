"""
Evaluation metrics — CLAUDE.md §3.

채택: RMSE (메인), MAE, MAPE, sMAPE, R².
모든 모델은 'vs Naive seasonal y(t-7)' 개선율과 함께 보고 (§3-3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """value > 0이라 정의됨 (CLAUDE.md §3-1). 0-division은 ε로 보호."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """대칭 MAPE — MAPE의 비대칭 문제 보정."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-9)) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_pred.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    # 표준 R² = 1 - SS_res / SS_tot_around_y_mean
    ss_tot_y = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot_y < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot_y


@dataclass(frozen=True)
class MetricBundle:
    rmse: float
    mae: float
    mape: float
    smape: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "mape": self.mape,
                "smape": self.smape, "r2": self.r2}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> MetricBundle:
    return MetricBundle(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        r2=r2(y_true, y_pred),
    )


def improvement_vs_baseline(
    metrics_model: MetricBundle,
    metrics_baseline: MetricBundle,
    *,
    metric: str = "rmse",
) -> float:
    """양수면 모델이 베이스라인보다 좋음(낮은 RMSE). %로 반환."""
    m = getattr(metrics_model, metric)
    b = getattr(metrics_baseline, metric)
    if b == 0:
        return float("nan")
    return float((b - m) / b * 100)


def metrics_table(
    runs: dict[str, MetricBundle],
    *,
    baseline_key: str = "naive_seasonal",
    primary_metric: str = "rmse",
) -> pd.DataFrame:
    """ {model_label: MetricBundle} → DataFrame with vs-baseline column."""
    rows = []
    base = runs.get(baseline_key)
    for label, mb in runs.items():
        row = {"model": label, **mb.to_dict()}
        if base is not None:
            row["vs_baseline_pct"] = improvement_vs_baseline(mb, base, metric=primary_metric)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.set_index("model")


__all__ = [
    "rmse", "mae", "mape", "smape", "r2",
    "MetricBundle", "compute_all",
    "improvement_vs_baseline", "metrics_table",
]
