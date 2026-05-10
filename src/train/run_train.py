"""
Training orchestrator — CLAUDE.md §5 STEP 4·6.

Pipeline:
  preprocess (별도 단계) → load preprocessed → build features →
  time-ordered split (config-fixed) → fit (baselines + Ridge + LightGBM + XGBoost) →
  predict on val + test → metrics table → persist artifacts.

산출물:
  - outputs/models/<model>.pkl     (학습된 모델)
  - outputs/predictions/metrics.csv (통합 지표)
  - outputs/predictions/predictions_<model>.csv (모든 split 예측)
  - outputs/analytics/feature_importance.md (LightGBM gain importance)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import MetricBundle, compute_all, metrics_table
from src.features.build import FeatureConfig, build_features, feature_matrix
from src.train.baselines import GroupMeanPredictor, NaiveSeasonalPredictor

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    train_idx: pd.DatetimeIndex
    val_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def make_splits(dates: pd.DatetimeIndex, train_end: str, val_end: str) -> Splits:
    train_end_d = pd.Timestamp(train_end)
    val_end_d = pd.Timestamp(val_end)
    train_idx = dates[dates <= train_end_d]
    val_idx = dates[(dates > train_end_d) & (dates <= val_end_d)]
    test_idx = dates[dates > val_end_d]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"empty split — train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
    return Splits(train_idx, val_idx, test_idx)


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _save_predictions(
    out_dir: Path,
    label: str,
    splits: Splits,
    y_true: pd.Series,
    preds: dict[str, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, idx in [("train", splits.train_idx), ("val", splits.val_idx), ("test", splits.test_idx)]:
        if split_name not in preds:
            continue
        for d, yt, yp in zip(idx, y_true.loc[idx].values, preds[split_name]):
            rows.append({"date": d, "split": split_name, "y_true": yt, "y_pred": yp})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"predictions_{label}.csv", index=False)


def fit_ridge(
    X: pd.DataFrame, y: pd.Series, splits: Splits, alpha: float
) -> tuple[Ridge, StandardScaler, dict[str, np.ndarray]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.loc[splits.train_idx])
    model = Ridge(alpha=alpha, random_state=0).fit(X_train, y.loc[splits.train_idx])

    preds: dict[str, np.ndarray] = {}
    for name, idx in [("train", splits.train_idx), ("val", splits.val_idx), ("test", splits.test_idx)]:
        preds[name] = model.predict(scaler.transform(X.loc[idx]))
    return model, scaler, preds


def fit_lightgbm(X: pd.DataFrame, y: pd.Series, splits: Splits, params: dict, esr: int):
    import lightgbm as lgb  # type: ignore[import-not-found]

    X_tr, y_tr = X.loc[splits.train_idx], y.loc[splits.train_idx]
    X_va, y_va = X.loc[splits.val_idx], y.loc[splits.val_idx]
    n_est = params.pop("n_estimators", 1000)
    model = lgb.LGBMRegressor(n_estimators=n_est, **params, random_state=42)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(0)],
    )
    preds = {
        "train": model.predict(X_tr),
        "val":   model.predict(X_va),
        "test":  model.predict(X.loc[splits.test_idx]),
    }
    return model, preds


def fit_xgboost(X: pd.DataFrame, y: pd.Series, splits: Splits, params: dict, esr: int):
    import xgboost as xgb  # type: ignore[import-not-found]

    X_tr, y_tr = X.loc[splits.train_idx], y.loc[splits.train_idx]
    X_va, y_va = X.loc[splits.val_idx], y.loc[splits.val_idx]
    n_est = params.pop("n_estimators", 1000)
    model = xgb.XGBRegressor(
        n_estimators=n_est,
        early_stopping_rounds=esr,
        random_state=42,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds = {
        "train": model.predict(X_tr),
        "val":   model.predict(X_va),
        "test":  model.predict(X.loc[splits.test_idx]),
    }
    return model, preds


def _save_lgbm_importance(model, X: pd.DataFrame, path: Path) -> None:
    imp = pd.DataFrame({
        "feature": X.columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LightGBM Feature Importance",
        "",
        "Gain importance — `outputs/analytics/feature_importance.md` 자동 생성.",
        "",
        "| rank | feature | gain | split |",
        "|---|---|---|---|",
    ]
    for i, row in imp.iterrows():
        lines.append(f"| {i+1} | `{row['feature']}` | {row['gain']:.0f} | {int(row['split'])} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    paths = cfg["paths"]
    df = pd.read_csv(paths["preprocessed_csv"], parse_dates=["date"])

    fcfg = FeatureConfig.from_dict(cfg["features"])
    df_feat = build_features(df, fcfg)
    X, y = feature_matrix(df_feat, target="value")

    splits = make_splits(X.index, cfg["split"]["train_end"], cfg["split"]["val_end"])
    logger.info("split sizes: train=%d  val=%d  test=%d",
                len(splits.train_idx), len(splits.val_idx), len(splits.test_idx))

    models_dir = Path(paths["models_dir"])
    preds_dir = Path(paths["predictions_dir"])
    runs: dict[str, MetricBundle] = {}
    val_runs: dict[str, MetricBundle] = {}

    # ── 1. Naive seasonal ────────────────────────────────────────────────
    if cfg["models"]["baselines"].get("naive_seasonal", True):
        logger.info("fit: naive_seasonal")
        train_dates = pd.Series(splits.train_idx)
        # 베이스라인은 *결측 보정된 원시 시계열 전체*를 history로 사용.
        full = df[["date", "value"]].copy().sort_values("date")
        ns = NaiveSeasonalPredictor(period=7).fit(full["date"], full["value"])
        preds = {
            "train": ns.predict(splits.train_idx),
            "val":   ns.predict(splits.val_idx),
            "test":  ns.predict(splits.test_idx),
        }
        _save_pickle(ns, models_dir / "naive_seasonal.pkl")
        _save_predictions(preds_dir, "naive_seasonal", splits, y, preds)
        runs["naive_seasonal"] = compute_all(y.loc[splits.test_idx].values, preds["test"])
        val_runs["naive_seasonal"] = compute_all(y.loc[splits.val_idx].values, preds["val"])

    # ── 2. Group mean ────────────────────────────────────────────────────
    if cfg["models"]["baselines"].get("group_mean", True):
        logger.info("fit: group_mean")
        train_df = df[df["date"].isin(splits.train_idx)]
        gm = GroupMeanPredictor().fit(train_df, target_col="value")
        preds = {}
        for name, idx in [("train", splits.train_idx), ("val", splits.val_idx), ("test", splits.test_idx)]:
            sub = df[df["date"].isin(idx)].copy()
            preds[name] = gm.predict(sub)
        _save_pickle(gm, models_dir / "group_mean.pkl")
        _save_predictions(preds_dir, "group_mean", splits, y, preds)
        runs["group_mean"] = compute_all(y.loc[splits.test_idx].values, preds["test"])
        val_runs["group_mean"] = compute_all(y.loc[splits.val_idx].values, preds["val"])

    # ── 3. Ridge ─────────────────────────────────────────────────────────
    if cfg["models"]["ridge"].get("enabled", True):
        logger.info("fit: ridge")
        alpha = float(cfg["models"]["ridge"].get("alpha", 1.0))
        ridge_model, scaler, preds = fit_ridge(X, y, splits, alpha=alpha)
        _save_pickle({"model": ridge_model, "scaler": scaler}, models_dir / "ridge.pkl")
        _save_predictions(preds_dir, "ridge", splits, y, preds)
        runs["ridge"] = compute_all(y.loc[splits.test_idx].values, preds["test"])
        val_runs["ridge"] = compute_all(y.loc[splits.val_idx].values, preds["val"])

    # ── 4. LightGBM ──────────────────────────────────────────────────────
    if cfg["models"]["lightgbm"].get("enabled", True):
        logger.info("fit: lightgbm")
        params = dict(cfg["models"]["lightgbm"]["params"])
        esr = int(cfg["models"]["lightgbm"].get("early_stopping_rounds", 50))
        lgbm_model, preds = fit_lightgbm(X, y, splits, params=params, esr=esr)
        _save_pickle(lgbm_model, models_dir / "lightgbm.pkl")
        _save_predictions(preds_dir, "lightgbm", splits, y, preds)
        runs["lightgbm"] = compute_all(y.loc[splits.test_idx].values, preds["test"])
        val_runs["lightgbm"] = compute_all(y.loc[splits.val_idx].values, preds["val"])
        _save_lgbm_importance(lgbm_model, X, Path(paths["feature_importance_path"]))

    # ── 5. XGBoost ───────────────────────────────────────────────────────
    if cfg["models"]["xgboost"].get("enabled", True):
        logger.info("fit: xgboost")
        params = dict(cfg["models"]["xgboost"]["params"])
        esr = int(cfg["models"]["xgboost"].get("early_stopping_rounds", 50))
        xgb_model, preds = fit_xgboost(X, y, splits, params=params, esr=esr)
        _save_pickle(xgb_model, models_dir / "xgboost.pkl")
        _save_predictions(preds_dir, "xgboost", splits, y, preds)
        runs["xgboost"] = compute_all(y.loc[splits.test_idx].values, preds["test"])
        val_runs["xgboost"] = compute_all(y.loc[splits.val_idx].values, preds["val"])

    # ── 통합 metrics 표 (test 기준) ────────────────────────────────────
    primary_baseline = cfg["reporting"]["primary_baseline"]
    primary_metric = cfg["reporting"]["primary_metric"]
    table = metrics_table(runs, baseline_key=primary_baseline, primary_metric=primary_metric)
    table_val = metrics_table(val_runs, baseline_key=primary_baseline, primary_metric=primary_metric)

    metrics_path = Path(paths["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_path)
    table_val.to_csv(metrics_path.with_name("metrics_val.csv"))
    logger.info("\n=== TEST metrics ===\n%s", table.round(3).to_string())
    logger.info("\n=== VAL metrics ===\n%s", table_val.round(3).to_string())
    logger.info("saved %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()
    main(args.config)
