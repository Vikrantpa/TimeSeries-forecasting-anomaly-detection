"""
Tests for the anomaly detection module: each detector on a controlled
synthetic case, the ensemble vote logic, and the sustained-alarm helper used
by evaluate_detection.py.
Run: pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.detectors import (
    Reference,
    ensemble_vote,
    isolation_forest_detector,
    mahalanobis_detector,
    pca_detector,
    zscore_detector,
)
from src.anomaly_detection.evaluate_detection import first_sustained_alarm

SENSOR_COLS = ["s1", "s2", "s3"]


def _healthy_frame(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 1, n)
    s1 = 50 + 0.8 * factor + rng.normal(0, 0.5, n)
    s2 = 10 + 0.6 * factor + rng.normal(0, 0.3, n)
    s3 = 100 + 0.1 * factor + rng.normal(0, 2.0, n)
    return pd.DataFrame({"s1": s1, "s2": s2, "s3": s3})


class TestMahalanobisAndPCA:
    def test_flags_a_correlated_off_manifold_point_even_when_marginals_look_mild(self):
        healthy = _healthy_frame()
        ref = Reference(healthy, sensor_cols=SENSOR_COLS)

        # s1 up, s2 DOWN — breaks the normal positive co-movement, each marginal shift modest (~2 sigma)
        anomaly = pd.DataFrame({"s1": [51.6], "s2": [9.1], "s3": [100.0]})
        combined = pd.concat([healthy, anomaly], ignore_index=True)

        maha = mahalanobis_detector(combined, ref, alpha=0.01)
        z = zscore_detector(combined, ref, z_threshold=3.0)

        assert bool(maha["flag"].iloc[-1]) is True
        assert bool(z["flag"].iloc[-1]) is False  # this is exactly the case the naive baseline misses

    def test_pca_reconstruction_error_flags_the_same_anomaly(self):
        healthy = _healthy_frame()
        ref = Reference(healthy, sensor_cols=SENSOR_COLS)
        anomaly = pd.DataFrame({"s1": [51.6], "s2": [9.1], "s3": [100.0]})
        combined = pd.concat([healthy, anomaly], ignore_index=True)

        pca = pca_detector(combined, ref, healthy, n_components=2, percentile=99.0)
        assert bool(pca["flag"].iloc[-1]) is True

    def test_false_positive_rate_on_pure_healthy_data_is_low(self):
        healthy = _healthy_frame(seed=1)
        ref = Reference(healthy, sensor_cols=SENSOR_COLS)
        test = _healthy_frame(n=2000, seed=2)  # fresh healthy data, same distribution
        maha = mahalanobis_detector(test, ref, alpha=0.01)
        assert maha["flag"].mean() < 0.05  # should be close to alpha, comfortably under a loose bound


class TestIsolationForest:
    def test_flags_a_clear_outlier(self):
        healthy = _healthy_frame(seed=3)
        ref = Reference(healthy, sensor_cols=SENSOR_COLS)
        outlier = pd.DataFrame({"s1": [80.0], "s2": [-20.0], "s3": [200.0]})  # egregious, not subtle
        combined = pd.concat([healthy, outlier], ignore_index=True)
        result = isolation_forest_detector(combined, ref, healthy, contamination=0.01)
        assert bool(result["flag"].iloc[-1]) is True


class TestEnsembleVote:
    def test_requires_at_least_two_of_three_advanced_detectors(self):
        flags = {
            "mahalanobis": pd.Series([True, True, False, False]),
            "pca": pd.Series([True, False, True, False]),
            "isolation_forest": pd.Series([False, False, False, True]),
        }
        result = ensemble_vote(flags)
        # votes per index: [2, 1, 1, 1] -> only index 0 clears the >=2 bar
        assert result.tolist() == [True, False, False, False]


class TestSustainedAlarm:
    def test_finds_first_index_of_a_sustained_run(self):
        flags = np.array([True, False, True, True, True, False])
        assert first_sustained_alarm(flags, min_consecutive=3) == 2

    def test_returns_none_when_never_sustained(self):
        flags = np.array([True, False, True, False, True])
        assert first_sustained_alarm(flags, min_consecutive=3) is None
