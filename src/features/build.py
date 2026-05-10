"""
Feature engineering — CLAUDE.md §5 STEP 3.

가설 → 피처 매핑:
  H3 (요일 효과)         → calendar features + cyclic encoding
  H5 (lag-7 자기상관)    → value_lag_{1,7,14,28}
  H8 (holiday × dow)     → 명시적 상호작용 피처 (선형 모델용; 트리는 자동)

CLAUDE.md §6 데이터 누수 방지 원칙:
  - lag/rolling 모두 reduce 전에 *반드시* shift(1).
  - 학습 시점에 미래 정보가 새지 않도록 모든 변환은 시간순.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    calendar: bool = True
    cyclic_encoding: bool = True
    lags: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    rolling_std_windows: tuple[int, ...] = (7,)
    interaction_holiday_dow: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(
            calendar=bool(d.get("calendar", True)),
            cyclic_encoding=bool(d.get("cyclic_encoding", True)),
            lags=tuple(d.get("lags", (1, 7, 14, 28))),
            rolling_windows=tuple(d.get("rolling_windows", (7, 28))),
            rolling_std_windows=tuple(d.get("rolling_std_windows", (7,))),
            interaction_holiday_dow=bool(d.get("interaction_holiday_dow", True)),
        )


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["dayofweek"] = d.dayofweek
    df["month"] = d.month
    df["day"] = d.day
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["quarter"] = d.quarter
    df["year"] = d.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"] = d.is_month_end.astype(int)
    # H4 추세 흡수용 — 시간 인덱스
    df["t_index"] = (df["date"] - df["date"].min()).dt.days
    return df


def _add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    # 트리 모델은 불필요하지만 Ridge에 사용 — 피처 셋은 통일.
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lags(df: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    # CLAUDE.md §6: lag/rolling은 반드시 shift(1) 후 계산.
    # lag-k = value.shift(k) 자체가 이미 미래정보 차단(k≥1).
    base = df["value"].shift(1)  # 명시적 single-step gating
    for k in lags:
        df[f"value_lag_{k}"] = df["value"].shift(k)
    return df


def _add_rolling(
    df: pd.DataFrame,
    mean_windows: Iterable[int],
    std_windows: Iterable[int],
) -> pd.DataFrame:
    # rolling은 반드시 shift(1) 이후 적용 (현재 시점 미포함).
    base = df["value"].shift(1)
    for w in mean_windows:
        df[f"rolling_mean_{w}"] = base.rolling(w, min_periods=max(2, w // 2)).mean()
    for w in std_windows:
        df[f"rolling_std_{w}"] = base.rolling(w, min_periods=max(2, w // 2)).std()
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # H8: holiday × dayofweek. 7×2=14개 인디케이터.
    for d in range(7):
        df[f"int_h_dow{d}"] = ((df["dayofweek"] == d) & (df["holiday"] == 1)).astype(int)
    return df


def build_features(df: pd.DataFrame, fcfg: FeatureConfig) -> pd.DataFrame:
    """Add features to df. Caller decides what to drop (NaN rows from lags) downstream."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    if fcfg.calendar:
        df = _add_calendar(df)
    if fcfg.cyclic_encoding:
        df = _add_cyclic(df)
    df = _add_lags(df, fcfg.lags)
    df = _add_rolling(df, fcfg.rolling_windows, fcfg.rolling_std_windows)
    if fcfg.interaction_holiday_dow:
        df = _add_interactions(df)

    logger.info("features built: shape=%s", df.shape)
    return df


# 모델에 넣을 컬럼 셋 — date/value/log_value/imputation flag 등은 비-피처
NON_FEATURE_COLS = {
    "date",
    "value",
    "log_value",
    "was_imputed",
    "is_outlier_capped",
    "event",  # event_flag 사용
}


def feature_matrix(df: pd.DataFrame, *, target: str = "value") -> tuple[pd.DataFrame, pd.Series]:
    """피처 매트릭스 X와 타겟 y 분리. lag NaN 행 제거."""
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    sub = df.dropna(subset=feat_cols + [target]).copy()
    X = sub[feat_cols].copy()
    y = sub[target].copy()
    dates = sub["date"].copy()
    X.index = dates
    y.index = dates
    return X, y


__all__ = ["FeatureConfig", "build_features", "feature_matrix", "NON_FEATURE_COLS"]
