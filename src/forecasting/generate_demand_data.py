"""
generate_demand_data.py
========================
Synthetic daily demand for a small retail chain: 5 stores x 2 products = 10
series, each with its own base level, a slow trend, weekly seasonality,
occasional promotions, and over-dispersed (Gamma-Poisson) count noise —
demand data is counts, not continuous, and it's rarely as clean as a Gaussian.

One store ("S5") is deliberately made a recent opening with only ~60 days of
history before the forecast window, instead of the ~670 days every other
store has. This is the scenario hierarchical/partial-pooling models are
actually for: a short, noisy series that can borrow statistical strength from
the other (structurally similar) series instead of being modeled in total
isolation. `bayesian_structural_model.py` demonstrates exactly that.

Usage:
    python -m src.forecasting.generate_demand_data --seed 11
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

N_DAYS = 700
TEST_HORIZON = 28
STORES = ["S1", "S2", "S3", "S4", "S5"]
PRODUCTS = ["P1", "P2"]
NEW_STORE = "S5"
NEW_STORE_START_DAY = 655  # only 17 days of history before the 28-day test window starts at day 672 —
# short enough that a per-series MLE genuinely overfits (see bayesian_structural_model.py's docstring
# and docs/ARCHITECTURE.md for exactly what goes wrong and how hierarchical shrinkage fixes it)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "demand")

# per-store base popularity and per-product multiplier — deliberately different scales
# so pooling has to do real work (it can't just copy one series onto another).
STORE_BASE = {"S1": 120, "S2": 45, "S3": 80, "S4": 200, "S5": 60}
PRODUCT_MULT = {"P1": 1.0, "P2": 0.6}
# weekend lift, Mon=0..Sun=6
WEEKDAY_EFFECT = np.array([0.95, 0.92, 0.95, 1.00, 1.10, 1.35, 1.25])
TREND_PER_DAY = {"S1": 0.03, "S2": -0.01, "S3": 0.015, "S4": 0.05, "S5": 0.08}
OVERDISPERSION_K = 8.0  # Gamma-Poisson shape — lower = more overdispersed than plain Poisson


def simulate_series(store: str, product: str, rng: np.random.Generator) -> pd.DataFrame:
    start_day = NEW_STORE_START_DAY if store == NEW_STORE else 0
    day_index = np.arange(start_day, N_DAYS)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")[day_index]
    weekday = dates.dayofweek.values

    base = STORE_BASE[store] * PRODUCT_MULT[product]
    trend = TREND_PER_DAY[store] * day_index
    seasonal = WEEKDAY_EFFECT[weekday]
    is_promo = rng.random(len(day_index)) < 0.06
    promo_mult = np.where(is_promo, 1.6, 1.0)

    mean_demand = np.clip((base + trend) * seasonal * promo_mult, 1.0, None)
    # Gamma-Poisson (negative-binomial) sampling for over-dispersed counts
    gamma_rate = rng.gamma(shape=OVERDISPERSION_K, scale=mean_demand / OVERDISPERSION_K)
    demand = rng.poisson(gamma_rate)

    return pd.DataFrame(
        {
            "date": dates,
            "store": store,
            "product": product,
            "series_id": f"{store}_{product}",
            "day_index": day_index,
            "weekday": weekday,
            "is_promo": is_promo,
            "demand": demand,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-store demand data.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = [simulate_series(store, product, rng) for store in STORES for product in PRODUCTS]
    demand = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(args.out_dir, "demand.csv")
    demand.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(demand)} rows across {demand['series_id'].nunique()} series)")

    manifest = {
        "n_days": N_DAYS,
        "test_horizon": TEST_HORIZON,
        "stores": STORES,
        "products": PRODUCTS,
        "new_store": NEW_STORE,
        "new_store_start_day": NEW_STORE_START_DAY,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote manifest.json")

    for sid, g in demand.groupby("series_id"):
        print(f"  {sid}: {len(g)} days, mean demand {g['demand'].mean():.1f}")


if __name__ == "__main__":
    main()
