"""
Parameter spec — CLAUDE.md §5-2.

TuRBO와 Optuna가 *동일한 파라미터 공간*을 탐색하도록 단일 정의.
TuRBO는 [0,1]^d unit cube를 받기 때문에 decode 어댑터로 변환.
Optuna는 동일 명세에서 suggest_* 호출.

본 명세는 LightGBM의 9개 핵심 하이퍼파라미터 (CLAUDE.md §2-2 M4 가설 = "9차원").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ParamType = Literal["int", "float", "log_float"]


@dataclass(frozen=True)
class ParamDef:
    name: str
    kind: ParamType
    low: float
    high: float

    def decode_unit(self, u: float) -> Any:
        """[0,1] → 실제 값 (TuRBO용)."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "int":
            v = self.low + u * (self.high - self.low)
            return int(round(v))
        if self.kind == "float":
            return float(self.low + u * (self.high - self.low))
        if self.kind == "log_float":
            log_low, log_high = np.log(self.low), np.log(self.high)
            return float(np.exp(log_low + u * (log_high - log_low)))
        raise ValueError(self.kind)

    def suggest(self, trial) -> Any:
        """Optuna trial에서 suggest_*. 두 옵티마이저가 동일 공간을 보도록."""
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        if self.kind == "float":
            return trial.suggest_float(self.name, float(self.low), float(self.high))
        if self.kind == "log_float":
            return trial.suggest_float(self.name, float(self.low), float(self.high), log=True)
        raise ValueError(self.kind)


# ── LightGBM 9-차원 공간 ────────────────────────────────────────────────
# 범위는 LightGBM 표준 + M5 우승 레시피 + 본 데이터 크기(1500행) 보수적 조정.
LGBM_SPACE: list[ParamDef] = [
    ParamDef("learning_rate",    "log_float", 1e-3, 0.3),
    ParamDef("num_leaves",       "int",       8,    255),
    ParamDef("max_depth",        "int",       3,    12),
    ParamDef("min_child_samples","int",       5,    100),
    ParamDef("feature_fraction", "float",     0.5,  1.0),
    ParamDef("bagging_fraction", "float",     0.5,  1.0),
    ParamDef("bagging_freq",     "int",       0,    7),
    ParamDef("lambda_l1",        "log_float", 1e-4, 10.0),
    ParamDef("lambda_l2",        "log_float", 1e-4, 10.0),
]


def decode_unit_vector(u: np.ndarray, space: list[ParamDef] = LGBM_SPACE) -> dict[str, Any]:
    """TuRBO: [0,1]^d → params dict."""
    if len(u) != len(space):
        raise ValueError(f"dim mismatch: u={len(u)} vs space={len(space)}")
    return {p.name: p.decode_unit(u[i]) for i, p in enumerate(space)}


def suggest_optuna(trial, space: list[ParamDef] = LGBM_SPACE) -> dict[str, Any]:
    """Optuna: trial → params dict."""
    return {p.name: p.suggest(trial) for p in space}


def dim(space: list[ParamDef] = LGBM_SPACE) -> int:
    return len(space)


def coerce_types(params: dict[str, Any], space: list[ParamDef] = LGBM_SPACE) -> dict[str, Any]:
    """Pandas/JSON 라운드트립 후 int 파라미터가 float이 된 경우 강제 복구.

    LightGBM은 num_leaves=199.0을 거부 (str 변환 시 "199.0").
    """
    out = dict(params)
    for p in space:
        if p.name not in out:
            continue
        v = out[p.name]
        if p.kind == "int":
            out[p.name] = int(round(float(v)))
        elif p.kind in ("float", "log_float"):
            out[p.name] = float(v)
    return out


__all__ = ["ParamDef", "LGBM_SPACE", "decode_unit_vector", "suggest_optuna", "dim", "coerce_types"]
