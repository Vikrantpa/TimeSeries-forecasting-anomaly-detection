"""
bayesian_structural_model.py
=============================
A hierarchical Bayesian structural time series model, implemented from
scratch with a Kalman filter (no PyMC/Stan — this runs in milliseconds and
has zero heavyweight dependencies, at the cost of using a Gaussian/empirical-
Bayes approximation instead of full MCMC; see the docstring at the bottom and
docs/ARCHITECTURE.md for exactly what that trade-off means).

**The model** (per series, fit on log1p(demand) so forecasts stay positive
and multiplicative effects behave additively): a local linear trend
(level + trend, a "random walk with a drifting slope") plus a single
trigonometric harmonic capturing the weekly cycle. This is precisely the
"structural time series" decomposition BSTS software (Google's `bsts` R
package, `statsmodels.UnobservedComponents`) uses — level, trend, seasonal —
just written out by hand.

**The "hierarchical" part**: instead of fitting each of the 10 store-product
series in total isolation, each series' noise-variance hyperparameters are
shrunk toward the *across-series* average, with the shrinkage strength
inversely related to how much data that series has. A brand-new store with
60 days of history gets pulled hard toward what "typical" stores look like;
a store with 700 days of history is barely adjusted. This is a two-stage
empirical-Bayes approximation to a fully joint hierarchical model — the same
idea partial pooling always buys you (short/noisy groups borrow strength
from the population), computed without MCMC.

Usage (see also evaluate_forecasts.py, which drives this end-to-end):
    from src.forecasting.bayesian_structural_model import fit_and_forecast_all
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

SEASONAL_PERIOD = 7
THETA = 2 * np.pi / SEASONAL_PERIOD
N_STATE = 4  # [level, trend, s1, s2]

# empirical-Bayes shrinkage strength: weight = SHRINKAGE_C / (SHRINKAGE_C + n_obs)
SHRINKAGE_C = 250.0


# --------------------------------------------------------------------------- #
# State space matrices
# --------------------------------------------------------------------------- #
def transition_matrix() -> np.ndarray:
    c, s = np.cos(THETA), np.sin(THETA)
    return np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c],
        ]
    )


Z = np.array([1.0, 0.0, 1.0, 0.0])  # observation picks out level + seasonal-1


def build_Q_R(log_params: np.ndarray) -> tuple[np.ndarray, float]:
    """log_params = [log_q_level, log_q_trend, log_q_season, log_r_obs] -> (Q, R)."""
    q_level, q_trend, q_season, r_obs = np.exp(log_params)
    Q = np.diag([q_level, q_trend, q_season, q_season])
    return Q, r_obs


# --------------------------------------------------------------------------- #
# Kalman filter (univariate observation, 4-dim state)
# --------------------------------------------------------------------------- #
def kalman_filter(y: np.ndarray, Q: np.ndarray, r_obs: float) -> dict:
    """Returns filtered states/covariances at every time step and the data log-likelihood."""
    T_mat = transition_matrix()
    n = len(y)
    a = np.zeros((n + 1, N_STATE))
    P = np.zeros((n + 1, N_STATE, N_STATE))
    a[0] = np.array([y[0], 0.0, 0.0, 0.0])
    P[0] = np.eye(N_STATE) * 10.0  # diffuse-ish initial uncertainty

    loglik = 0.0
    for t in range(n):
        a_pred = T_mat @ a[t]
        P_pred = T_mat @ P[t] @ T_mat.T + Q

        v = y[t] - Z @ a_pred
        F = float(Z @ P_pred @ Z.T + r_obs)
        F = max(F, 1e-8)
        K = (P_pred @ Z.T) / F

        a[t + 1] = a_pred + K * v
        P[t + 1] = P_pred - np.outer(K, Z @ P_pred)
        loglik += -0.5 * (np.log(2 * np.pi * F) + v**2 / F)

    return {"a": a, "P": P, "loglik": loglik}


def negative_log_likelihood(log_params: np.ndarray, y: np.ndarray) -> float:
    Q, r_obs = build_Q_R(log_params)
    try:
        result = kalman_filter(y, Q, r_obs)
    except np.linalg.LinAlgError:
        return 1e10
    if not np.isfinite(result["loglik"]):
        return 1e10
    return -result["loglik"]


def fit_series_mle(y: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """MLE of the 4 log-hyperparameters for one series via the Kalman-filter likelihood."""
    if init is None:
        var0 = max(np.var(np.diff(y)), 1e-3)
        init = np.log([var0 * 0.05, var0 * 0.01, var0 * 0.05, var0 * 0.5])
    res = minimize(negative_log_likelihood, init, args=(y,), method="L-BFGS-B",
                    bounds=[(-12, 3)] * 4)
    return res.x


# --------------------------------------------------------------------------- #
# Hierarchical (empirical-Bayes) shrinkage across series
# --------------------------------------------------------------------------- #
def shrink_hyperparameters(raw_log_params: dict[str, np.ndarray], lengths: dict[str, int]) -> dict[str, np.ndarray]:
    all_params = np.stack(list(raw_log_params.values()))
    group_mean = all_params.mean(axis=0)
    shrunk = {}
    for series_id, theta in raw_log_params.items():
        n = lengths[series_id]
        w = SHRINKAGE_C / (SHRINKAGE_C + n)  # more data -> less shrinkage
        shrunk[series_id] = w * group_mean + (1 - w) * theta
    return shrunk, group_mean


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def forecast(y_log: np.ndarray, log_params: np.ndarray, horizon: int, quantiles=(0.1, 0.5, 0.9)) -> dict:
    """Filter through the observed data, then propagate the state forward `horizon` steps
    with no further updates, returning quantile forecasts back-transformed to demand units."""
    Q, r_obs = build_Q_R(log_params)
    fit = kalman_filter(y_log, Q, r_obs)
    a, P = fit["a"][-1], fit["P"][-1]
    T_mat = transition_matrix()

    means, variances = [], []
    for _ in range(horizon):
        a = T_mat @ a
        P = T_mat @ P @ T_mat.T + Q
        means.append(float(Z @ a))
        variances.append(float(Z @ P @ Z.T + r_obs))
    means, variances = np.array(means), np.array(variances)
    stds = np.sqrt(variances)

    out = {"loglik": fit["loglik"], "mean_log": means, "std_log": stds}
    for q in quantiles:
        z = norm.ppf(q)
        out[f"q{int(q * 100)}"] = np.expm1(means + z * stds)
    return out


def fit_and_forecast_all(demand: pd.DataFrame, test_horizon: int, quantiles=(0.1, 0.5, 0.9)) -> dict:
    """End-to-end: fit every series' MLE hyperparameters, shrink them hierarchically, forecast
    the held-out horizon with both the raw (unpooled) and shrunk (pooled) hyperparameters so the
    benefit of pooling can be measured directly (see evaluate_forecasts.py)."""
    raw_log_params, lengths, train_logs = {}, {}, {}
    for series_id, g in demand.groupby("series_id"):
        g = g.sort_values("day_index")
        train = g.iloc[: len(g) - test_horizon]
        y_log = np.log1p(train["demand"].values.astype(float))
        raw_log_params[series_id] = fit_series_mle(y_log)
        lengths[series_id] = len(train)
        train_logs[series_id] = y_log

    shrunk_log_params, group_mean = shrink_hyperparameters(raw_log_params, lengths)

    results = {}
    for series_id, y_log in train_logs.items():
        results[series_id] = {
            "unpooled": forecast(y_log, raw_log_params[series_id], test_horizon, quantiles),
            "pooled": forecast(y_log, shrunk_log_params[series_id], test_horizon, quantiles),
            "raw_log_params": raw_log_params[series_id],
            "shrunk_log_params": shrunk_log_params[series_id],
            "n_train": lengths[series_id],
        }
    results["_group_mean_log_params"] = group_mean
    return results
