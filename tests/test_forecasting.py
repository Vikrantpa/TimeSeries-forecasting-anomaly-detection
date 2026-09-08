"""
Tests for the forecasting module: the Kalman filter mechanics, the
hierarchical shrinkage direction, and the pinball-loss scoring function.
Run: pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.bayesian_structural_model import (
    build_Q_R,
    fit_series_mle,
    forecast,
    kalman_filter,
    shrink_hyperparameters,
)
from src.forecasting.evaluate_forecasts import pinball_loss


class TestKalmanFilter:
    def test_filtered_level_tracks_a_known_constant_level(self):
        rng = np.random.default_rng(0)
        true_level = 50.0
        y = true_level + rng.normal(0, 2.0, 300)
        log_params = np.log([1e-4, 1e-4, 1e-4, 4.0])  # near-zero state noise -> should converge to the mean
        Q, r_obs = build_Q_R(log_params)
        result = kalman_filter(y, Q, r_obs)
        filtered_level = result["a"][-1, 0]
        assert filtered_level == pytest.approx(true_level, abs=1.0)

    def test_forecast_variance_grows_with_horizon_when_state_noise_is_present(self):
        rng = np.random.default_rng(1)
        y = np.log1p(50 + rng.normal(0, 3, 100))
        log_params = np.log([0.01, 0.001, 0.001, 0.05])  # nonzero state noise
        out = forecast(y, log_params, horizon=10)
        # with no further observations, uncertainty about a random-walk state can only grow
        assert np.all(np.diff(out["std_log"]) >= -1e-9)

    def test_forecast_quantiles_are_monotonic(self):
        rng = np.random.default_rng(2)
        y = np.log1p(50 + rng.normal(0, 3, 100))
        log_params = fit_series_mle(y)
        out = forecast(y, log_params, horizon=14)
        assert np.all(out["q10"] <= out["q50"] + 1e-9)
        assert np.all(out["q50"] <= out["q90"] + 1e-9)


class TestHierarchicalShrinkage:
    def test_short_series_shrinks_more_than_long_series(self):
        raw = {"short": np.array([1.0, 1.0, 1.0, 1.0]), "long": np.array([-1.0, -1.0, -1.0, -1.0])}
        lengths = {"short": 10, "long": 2000}
        shrunk, group_mean = shrink_hyperparameters(raw, lengths)
        # the short series should move much closer to the group mean than the long one does
        short_move = np.abs(shrunk["short"] - raw["short"]).sum()
        long_move = np.abs(shrunk["long"] - raw["long"]).sum()
        assert short_move > long_move

    def test_shrinkage_stays_between_own_estimate_and_group_mean(self):
        raw = {"a": np.array([2.0, 2.0, 2.0, 2.0]), "b": np.array([-2.0, -2.0, -2.0, -2.0])}
        lengths = {"a": 50, "b": 50}
        shrunk, group_mean = shrink_hyperparameters(raw, lengths)
        for series_id in raw:
            lo, hi = sorted([raw[series_id][0], group_mean[0]])
            assert lo - 1e-9 <= shrunk[series_id][0] <= hi + 1e-9


class TestPinballLoss:
    def test_perfect_forecast_has_zero_loss(self):
        y = np.array([10.0, 20.0, 30.0])
        assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)

    def test_underforecasting_is_penalized_more_at_high_quantiles(self):
        y = np.array([100.0])
        low_forecast = np.array([50.0])
        # at q=0.9, under-forecasting (forecast below actual) should cost more than at q=0.1
        loss_q90 = pinball_loss(y, low_forecast, 0.9)
        loss_q10 = pinball_loss(y, low_forecast, 0.1)
        assert loss_q90 > loss_q10
