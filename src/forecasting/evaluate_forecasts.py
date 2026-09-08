"""
evaluate_forecasts.py
=======================
Runs both forecasting models end-to-end on data/demand/demand.csv, scores
them with proper probabilistic-forecasting metrics (pinball/quantile loss and
empirical interval coverage — never just RMSE on the median, which throws
away everything about the uncertainty estimate), and saves two charts:

  outputs/forecasting/pooling_effect.png     — the hierarchical-pooling story:
      the new store's unpooled vs. pooled Bayesian forecast, side by side
  outputs/forecasting/forecast_comparison.png — Bayesian (pooled) vs. quantile-GBM
      median + 80% interval against actuals, across four representative series

Usage:
    python -m src.forecasting.evaluate_forecasts
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting.bayesian_structural_model import fit_and_forecast_all as bayes_fit_all
from src.forecasting.quantile_gbm_model import fit_and_forecast_all as gbm_fit_all

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "demand")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "forecasting")

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"


def pinball_loss(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    diff = y - yhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(y_true: np.ndarray, forecast: dict) -> dict:
    p10 = pinball_loss(y_true, forecast["q10"], 0.1)
    p50 = pinball_loss(y_true, forecast["q50"], 0.5)
    p90 = pinball_loss(y_true, forecast["q90"], 0.9)
    coverage = float(np.mean((y_true >= forecast["q10"]) & (y_true <= forecast["q90"])))
    return {"pinball10": p10, "pinball50": p50, "pinball90": p90,
            "pinball_total": p10 + p50 + p90, "coverage80": coverage}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    demand = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"), parse_dates=["date"])
    with open(os.path.join(DATA_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    test_horizon = manifest["test_horizon"]

    print("fitting hierarchical Bayesian structural models (per series MLE + shrinkage)...")
    bayes = bayes_fit_all(demand, test_horizon)
    print("fitting pooled quantile-GBM model...")
    gbm, _, _ = gbm_fit_all(demand, test_horizon)

    rows = []
    for sid, g in demand.groupby("series_id"):
        g = g.sort_values("day_index")
        y_true = g.iloc[-test_horizon:]["demand"].values.astype(float)
        for method, forecast in [
            ("bayes_unpooled", bayes[sid]["unpooled"]),
            ("bayes_pooled", bayes[sid]["pooled"]),
            ("gbm", gbm[sid]),
        ]:
            m = score(y_true, forecast)
            rows.append({"series_id": sid, "method": method, "n_train": bayes[sid]["n_train"], **m})

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "forecast_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")
    print(summary.pivot(index="series_id", columns="method", values="pinball_total").round(1).to_string())

    plot_pooling_effect(demand, bayes, test_horizon)
    plot_forecast_comparison(demand, bayes, gbm, test_horizon)


def _test_dates(g: pd.DataFrame, test_horizon: int) -> pd.Series:
    return g.sort_values("day_index").iloc[-test_horizon:]["date"]


def plot_pooling_effect(demand: pd.DataFrame, bayes: dict, test_horizon: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, sid in zip(axes, ["S5_P1", "S5_P2"]):
        g = demand[demand.series_id == sid].sort_values("day_index")
        train = g.iloc[: len(g) - test_horizon]
        test = g.iloc[-test_horizon:]
        dates = test["date"]

        ax.plot(train["date"], train["demand"], color=INK, linewidth=1, alpha=0.6, label="history (train)")
        ax.plot(dates, test["demand"], "o", color=INK, markersize=4, label="actual")

        for label, key, color, alpha in [("unpooled (per-series MLE)", "unpooled", "#d03b3b", 0.15),
                                          ("pooled (hierarchical shrinkage)", "pooled", BLUE, 0.25)]:
            f = bayes[sid][key]
            ax.plot(dates, f["q50"], color=color, linewidth=2, label=f"{label} median")
            ax.fill_between(dates, f["q10"], f["q90"], color=color, alpha=alpha, label=f"{label} 80% interval")

        ax.set_title(f"{sid}  (only {bayes[sid]['n_train']} days of training history)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Hierarchical pooling rescues a short, newly-opened store's forecast", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "pooling_effect.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_forecast_comparison(demand: pd.DataFrame, bayes: dict, gbm: dict, test_horizon: int) -> None:
    series_ids = ["S1_P1", "S4_P2", "S5_P1", "S5_P2"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, sid in zip(axes.flat, series_ids):
        g = demand[demand.series_id == sid].sort_values("day_index")
        test = g.iloc[-test_horizon:]
        dates = test["date"]
        y_true = test["demand"].values.astype(float)

        ax.plot(dates, y_true, "o-", color=INK, markersize=4, linewidth=1, label="actual")

        bf = bayes[sid]["pooled"]
        ax.plot(dates, bf["q50"], color=BLUE, linewidth=2, label="Bayesian (pooled) median")
        ax.fill_between(dates, bf["q10"], bf["q90"], color=BLUE, alpha=0.2, label="Bayesian 80% interval")

        gf = gbm[sid]
        ax.plot(dates, gf["q50"], color=ORANGE, linewidth=2, label="Quantile-GBM median")
        ax.fill_between(dates, gf["q10"], gf["q90"], color=ORANGE, alpha=0.2, label="GBM 80% interval")

        ax.set_title(f"{sid} (n_train={bayes[sid]['n_train']})")
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Bayesian structural model vs. quantile-GBM: 28-day-ahead forecasts", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "forecast_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
