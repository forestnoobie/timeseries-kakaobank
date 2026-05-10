"""
Predict + residual analysis + SHAP — CLAUDE.md §6-2, §6-3.

산출물 (outputs/predictions/):
  - residuals.csv                (모델별 잔차)
  - residual_plots.png           (시계열, 잔차 vs 예측, Q-Q)
  - top10_errors.csv             (가장 큰 오차 일자)
  - shap_summary.png             (LightGBM SHAP)

SHAP은 lightgbm/xgboost 학습된 모델만 분석 (선형 모델은 계수 자체가 해석).
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.features.build import FeatureConfig, build_features, feature_matrix
from src.train.run_train import make_splits

logger = logging.getLogger(__name__)


def _load_predictions(preds_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for path in preds_dir.glob("predictions_*.csv"):
        label = path.stem.replace("predictions_", "")
        df = pd.read_csv(path, parse_dates=["date"])
        out[label] = df
    return out


def _residual_plots(preds: dict[str, pd.DataFrame], out_path: Path, focus: str = "lightgbm") -> None:
    df = preds.get(focus)
    if df is None:
        logger.warning("no predictions for %s — skip residual plots", focus)
        return
    test = df[df["split"] == "test"].copy()
    test["resid"] = test["y_true"] - test["y_pred"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(test["date"], test["resid"], "o-", ms=3, color="C0")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_title(f"{focus} — residual time series (test)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].scatter(test["y_pred"], test["resid"], s=12, color="C0", alpha=0.7)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("y_pred"); axes[1].set_ylabel("resid")
    axes[1].set_title("residual vs prediction (heteroscedasticity)")

    stats.probplot(test["resid"], dist="norm", plot=axes[2])
    axes[2].set_title("residual Q-Q plot")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _top_errors(preds: dict[str, pd.DataFrame], out_path: Path, focus: str, k: int = 10) -> None:
    df = preds.get(focus)
    if df is None:
        return
    test = df[df["split"] == "test"].copy()
    test["abs_err"] = (test["y_true"] - test["y_pred"]).abs()
    top = test.sort_values("abs_err", ascending=False).head(k)
    top.to_csv(out_path, index=False)
    logger.info("top-%d errors (%s) → %s", k, focus, out_path)


def _residual_table(preds: dict[str, pd.DataFrame], out_path: Path) -> None:
    rows = []
    for label, df in preds.items():
        test = df[df["split"] == "test"]
        rows.append({
            "model": label,
            "resid_mean": (test["y_true"] - test["y_pred"]).mean(),
            "resid_std":  (test["y_true"] - test["y_pred"]).std(),
            "resid_skew": stats.skew(test["y_true"] - test["y_pred"]),
            "ljung_p":    sm_ljung_p(test["y_true"] - test["y_pred"]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def sm_ljung_p(resid: pd.Series) -> float:
    """Ljung-Box test: 잔차의 자기상관 잔존 여부.

    p < 0.05 → 자기상관 잔존 → 모델이 시계열 구조를 다 못 잡음.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        res = acorr_ljungbox(resid.dropna(), lags=[7], return_df=True)
        return float(res["lb_pvalue"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("ljung box failed: %s", exc)
        return float("nan")


def _shap_summary(model_path: Path, X_test: pd.DataFrame, out_path: Path) -> None:
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("shap not installed — skip SHAP summary")
        return
    if not model_path.exists():
        logger.warning("no model at %s — skip SHAP", model_path)
        return
    with model_path.open("rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    cfg = yaml.safe_load(Path(config_path).read_text())

    paths = cfg["paths"]
    preds_dir = Path(paths["predictions_dir"])
    preds = _load_predictions(preds_dir)
    logger.info("loaded predictions for: %s", list(preds.keys()))

    _residual_plots(preds, preds_dir / "residual_plots.png", focus="lightgbm")
    _top_errors(preds, preds_dir / "top10_errors_lightgbm.csv", focus="lightgbm")
    _residual_table(preds, preds_dir / "residual_summary.csv")

    # SHAP — LightGBM 모델 + test set
    df = pd.read_csv(paths["preprocessed_csv"], parse_dates=["date"])
    fcfg = FeatureConfig.from_dict(cfg["features"])
    df = build_features(df, fcfg)
    X, y = feature_matrix(df, target="value")
    splits = make_splits(X.index, cfg["split"]["train_end"], cfg["split"]["val_end"])
    X_test = X.loc[splits.test_idx]
    _shap_summary(Path(paths["models_dir"]) / "lightgbm.pkl", X_test,
                  preds_dir / "shap_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()
    main(args.config)
