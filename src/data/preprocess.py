"""
Preprocessing — CLAUDE.md §5 STEP 2.

EDA 근거 (outputs/analytics/notebook.ipynb 실행 결과):
  - H6: skew=+0.69 < 1.0 → 강한 우편향은 아님. log1p는 분산 안정화 옵션으로만.
  - H7: 이상치는 IQR 기준 다수. 상한이 휴일·event와 무관하면 winsorize 후보.
  - H9: 결측 11건. holiday rate (27%) ≈ overall (32%) → 강한 비-MCAR 증거 부족.
        그래도 안전을 위해 요일 평균(dow_mean)으로 보정 — 단순 0/forward-fill보다 robust.
  - event 컬럼: 스펙은 binary {0,1}이지만 실제 1건이 2 → event_flag로 정규화.

Step 2의 출력은 모델/피처와 독립한 '깨끗한 일별 시계열'.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    expected = {"date", "holiday", "event", "value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"raw csv missing columns: {missing}")
    return df


def normalize_event(df: pd.DataFrame, *, binarize: bool) -> pd.DataFrame:
    """event ∈ {0,1,2,...} → event_flag ∈ {0,1}.

    스펙은 binary지만 데이터에 2가 1건 존재 (notebook §1-1).
    binarize=False면 원본 보존.
    """
    df = df.copy()
    if binarize:
        df["event_flag"] = (df["event"] > 0).astype(int)
        n_anomaly = int((df["event"] > 1).sum())
        if n_anomaly > 0:
            logger.info("event>1 rows binarized: n=%d", n_anomaly)
    else:
        df["event_flag"] = df["event"]
    return df


def fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """결측 처리. CLAUDE.md §5 STEP 2 + H9.

    strategy:
      - linear     : 시계열 선형보간
      - dow_mean   : 동일 dayofweek 평균 (H3 요일효과 반영, robust 기본값)
      - forward_fill : ffill → bfill (가장 단순)
    """
    df = df.copy()
    n_missing = int(df["value"].isna().sum())
    if n_missing == 0:
        logger.info("missing=0; skip fill")
        return df

    if strategy == "linear":
        df["value"] = df["value"].interpolate("linear")
    elif strategy == "dow_mean":
        dow = df["date"].dt.dayofweek
        dow_means = df.groupby(dow)["value"].transform("mean")
        df["value"] = df["value"].fillna(dow_means)
    elif strategy == "forward_fill":
        df["value"] = df["value"].ffill().bfill()
    else:
        raise ValueError(f"unknown missing_strategy: {strategy}")

    remaining = int(df["value"].isna().sum())
    if remaining > 0:
        # 보수적 fallback — 어떤 전략도 끝/시작 결측을 남길 수 있음.
        df["value"] = df["value"].ffill().bfill()
    logger.info("missing_filled: n=%d, strategy=%s", n_missing, strategy)
    df["was_imputed"] = 0
    df.loc[df.index[df["value"].isna()], "was_imputed"] = 1  # noqa: E501  (안전 가드)
    return df


def handle_outliers(df: pd.DataFrame, strategy: str, quantiles: list[float]) -> pd.DataFrame:
    """이상치 처리. CLAUDE.md §5 STEP 2 + H7.

    휴일/event 일자는 무조건 보존 — 특수일은 정상 신호.
    strategy:
      - keep        : 그대로 (트리 모델 기본값)
      - winsorize   : non-special day만 quantile clipping
      - iqr_drop    : IQR 기준 제거 (시계열 끊김 → 비추)
    """
    df = df.copy()
    df["is_outlier_capped"] = 0
    if strategy == "keep":
        return df

    special = (df["holiday"] == 1) | (df["event_flag"] == 1)
    candidate = df[~special]

    if strategy == "winsorize":
        q_lo, q_hi = candidate["value"].quantile(quantiles)
        mask = (~special) & ((df["value"] < q_lo) | (df["value"] > q_hi))
        df.loc[mask, "value"] = df.loc[mask, "value"].clip(lower=q_lo, upper=q_hi)
        df.loc[mask, "is_outlier_capped"] = 1
        logger.info("winsorized n=%d (q_lo=%.1f, q_hi=%.1f)", int(mask.sum()), q_lo, q_hi)
    elif strategy == "iqr_drop":
        q1, q3 = candidate["value"].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (~special) & ((df["value"] < q1 - 1.5 * iqr) | (df["value"] > q3 + 1.5 * iqr))
        # drop 대신 NaN으로 두고 fill_missing 단계에서 채우는 편이 안전하나
        # 이 함수는 fill 이후 호출되므로 직접 결측 처리 — 시계열 보존.
        before = int(mask.sum())
        df.loc[mask, "value"] = np.nan
        df["value"] = df["value"].interpolate("linear").ffill().bfill()
        df.loc[mask, "is_outlier_capped"] = 1
        logger.info("iqr_dropped→interpolated n=%d", before)
    else:
        raise ValueError(f"unknown outlier_strategy: {strategy}")
    return df


def maybe_log1p(df: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    """log1p 변환 — H6 결과 + 분산 안정화.

    학습 시 적용. 예측은 expm1로 역변환 (CLAUDE.md §5 STEP 2 주석 패턴).
    여기서는 *원본*과 *log_value* 둘 다 보존하여 다운스트림이 선택.
    """
    df = df.copy()
    df["log_value"] = np.log1p(df["value"]) if enabled else np.nan
    return df


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = cfg["preprocess"]
    df = normalize_event(df, binarize=bool(p.get("binarize_event", True)))
    df = fill_missing(df, strategy=p.get("missing_strategy", "dow_mean"))
    df = handle_outliers(
        df,
        strategy=p.get("outlier_strategy", "keep"),
        quantiles=p.get("winsorize_quantiles", [0.001, 0.999]),
    )
    df = maybe_log1p(df, enabled=bool(p.get("log1p_target", False)))
    return df


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    cfg = yaml.safe_load(Path(config_path).read_text())
    raw_path = Path(cfg["paths"]["raw_csv"])
    out_path = Path(cfg["paths"]["preprocessed_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    logger.info("loaded raw: shape=%s, range=%s ~ %s",
                df.shape, df["date"].min().date(), df["date"].max().date())
    df = preprocess(df, cfg)
    df.to_csv(out_path, index=False)
    logger.info("saved %s (rows=%d, cols=%d)", out_path, len(df), df.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()
    main(args.config)
