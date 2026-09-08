"""
quantile_gbm_model.py
=======================
The "modern ML" probabilistic-forecasting baseline, contrasted against
`bayesian_structural_model.py`. Rather than a hand-rolled deep model
(TFT/N-BEATS need a GPU-friendly install and real training time — see the
README for why this project uses a lighter stand-in instead), this trains
three `sklearn.ensemble.GradientBoostingRegressor(loss="quantile")` models
(one each for the 10th/50th/90th percentile) **pooled across every
store-product series at once**, using lag/rolling/calendar/promo features
plus store & product identifiers as plain features.

That pooling strategy is the interesting contrast with the Bayesian model:
the structural model pools *statistical strength* across series through a
shared hyperparameter prior (shrinkage); this model pools *literally all the
training rows* into one global model and lets the tree ensemble learn
store/product effects as features. Two different, both legitimate, ways of
answering "how do related series help each other."

Forecasting 28 days ahead recursively feeds the model's own median (q50)
prediction back in as next-step lag features — the standard, if imperfect,
way to extend a lag-feature model past one step. Its real limitation (and
the reason the Bayesian model's interval widens with horizon while this one
mostly doesn't) is discussed in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = (0.1, 0.5, 0.9)
LAG_COLS = ["lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28"]
BASE_FEATURES = LAG_COLS + ["is_promo", "day_index"]


def _add_lag_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("day_index").copy()
    g["lag_1"] = g["demand"].shift(1)
    g["lag_7"] = g["demand"].shift(7)
    g["lag_14"] = g["demand"].shift(14)
    g["roll_mean_7"] = g["demand"].shift(1).rolling(7, min_periods=1).mean()
    g["roll_mean_28"] = g["demand"].shift(1).rolling(28, min_periods=1).mean()
    return g


def _one_hot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df[["weekday", "store", "product"]].astype(str), prefix=["wd", "store", "prod"])
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return out, list(dummies.columns)


def build_training_table(demand: pd.DataFrame, test_horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for _, g in demand.groupby("series_id"):
        g = _add_lag_features(g)
        frames.append(g.iloc[:-test_horizon])  # drop the held-out horizon from training
    train = pd.concat(frames, ignore_index=True)
    train, dummy_cols = _one_hot(train)
    train = train.dropna(subset=LAG_COLS)  # first ~14 rows of each series lack full lag history
    feature_cols = BASE_FEATURES + dummy_cols
    return train, feature_cols


def fit_quantile_models(train: pd.DataFrame, feature_cols: list[str], random_state: int = 0) -> dict:
    models = {}
    X = train[feature_cols].astype(float)
    y = train["demand"].astype(float)
    for q in QUANTILES:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=3,
                                           learning_rate=0.05, random_state=random_state)
        model.fit(X, y)
        models[q] = model
    return models


def recursive_forecast(series_df: pd.DataFrame, models: dict, feature_cols: list[str],
                        test_horizon: int, dummy_template: pd.DataFrame) -> dict:
    """Forecast `test_horizon` days ahead for one series, feeding the model's own q50 back in
    as the assumed value for lag/rolling features at the next step (recursive multi-step)."""
    g = series_df.sort_values("day_index").reset_index(drop=True)
    history = g.iloc[: len(g) - test_horizon].copy()
    future_known = g.iloc[len(g) - test_horizon :][["day_index", "weekday", "is_promo", "store", "product"]].reset_index(drop=True)

    working = history.copy()
    preds = {q: [] for q in QUANTILES}
    for step in range(test_horizon):
        next_row = future_known.iloc[step]
        working = pd.concat([working, pd.DataFrame([{**next_row.to_dict(), "demand": np.nan}])], ignore_index=True)
        working = _add_lag_features(working)
        feat_row, _ = _one_hot(working.tail(1))
        for col in feature_cols:
            if col not in feat_row.columns:
                feat_row[col] = 0.0
        X_next = feat_row[feature_cols].astype(float)

        row_preds = {q: max(0.0, float(models[q].predict(X_next)[0])) for q in QUANTILES}
        # enforce monotonic quantiles (GBMs fit independently can cross)
        q10, q50, q90 = sorted(row_preds.values())
        row_preds = {0.1: q10, 0.5: q50, 0.9: q90}
        for q in QUANTILES:
            preds[q].append(row_preds[q])
        working.loc[working.index[-1], "demand"] = row_preds[0.5]  # feed q50 back in as the assumed actual

    return {f"q{int(q*100)}": np.array(v) for q, v in preds.items()}


def fit_and_forecast_all(demand: pd.DataFrame, test_horizon: int) -> dict:
    train, feature_cols = build_training_table(demand, test_horizon)
    models = fit_quantile_models(train, feature_cols)
    results = {}
    for sid, g in demand.groupby("series_id"):
        results[sid] = recursive_forecast(g, models, feature_cols, test_horizon, train)
    return results, feature_cols, models
