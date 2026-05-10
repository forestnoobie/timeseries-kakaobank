"""
Baselines — CLAUDE.md §5 STEP 4.

베이스라인은 어떤 모델이든 *반드시* 능가해야 함. 두 가지:
  1. NaiveSeasonal y(t-7)        : 강한 주간 계절성(H5) 활용
  2. GroupMean (dow × holiday)   : 가장 단순한 휴리스틱
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NaiveSeasonalPredictor:
    """y(t) := y(t-period). period=7 → 같은 요일 1주일 전.

    history는 fit에서 받은 (date → value) 매핑을 보존하고, predict에서 t-period가
    history에 있으면 그 값을 반환, 없으면 직전 가용 lag로 fallback.
    """
    period: int = 7

    def fit(self, dates: pd.Series, values: pd.Series) -> "NaiveSeasonalPredictor":
        self.history_ = pd.Series(values.values, index=pd.to_datetime(dates).values).sort_index()
        return self

    def predict(self, target_dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        target_dates = pd.to_datetime(pd.Series(target_dates) if not isinstance(target_dates, pd.DatetimeIndex) else target_dates)
        idx = pd.DatetimeIndex(target_dates)
        out: list[float] = []
        for d in idx:
            ref = d - pd.Timedelta(days=self.period)
            if ref in self.history_.index:
                out.append(float(self.history_.loc[ref]))
            else:
                # fallback: history에 있는 가장 가까운 과거 값
                past = self.history_.loc[: d - pd.Timedelta(days=1)]
                out.append(float(past.iloc[-1]) if len(past) > 0 else float("nan"))
        return np.asarray(out, dtype=float)


@dataclass
class GroupMeanPredictor:
    """(dayofweek, holiday) 그룹 평균.

    H1+H3 조합 — 가장 단순한 의미있는 베이스라인.
    """
    means_: dict[tuple[int, int], float] | None = None
    overall_mean_: float = 0.0

    def fit(self, df: pd.DataFrame, target_col: str = "value") -> "GroupMeanPredictor":
        df = df.copy()
        df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
        gb = df.groupby(["dow", "holiday"])[target_col].mean()
        self.means_ = {(int(k[0]), int(k[1])): float(v) for k, v in gb.items()}
        self.overall_mean_ = float(df[target_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
        out: list[float] = []
        for _, row in df.iterrows():
            key = (int(row["dow"]), int(row["holiday"]))
            out.append(self.means_.get(key, self.overall_mean_) if self.means_ is not None else self.overall_mean_)
        return np.asarray(out, dtype=float)


__all__ = ["NaiveSeasonalPredictor", "GroupMeanPredictor"]
