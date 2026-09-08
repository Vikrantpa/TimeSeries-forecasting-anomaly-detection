"""
detectors.py
=============
Four anomaly detectors, from a naive per-sensor baseline up to a small
ensemble, all fit on a "healthy" reference window (the first
`healthy_training_window_hours` hours, pooled across every machine — the
assumption that mirrors real deployment: you calibrate against known-good
fleet behavior, then watch for departures from it):

1. **Per-sensor z-score** — flags an hour if ANY single sensor is more than
   `z_threshold` standard deviations from its own healthy mean. This is the
   naive baseline every plant already has (a threshold alarm per gauge), and
   it's blind to the exact failure mode this project injects: several
   sensors drifting together in a correlated way that's individually
   unremarkable but jointly very unusual.
2. **Mahalanobis distance** — the multivariate generalization of a z-score:
   distance from the healthy mean, scaled by the healthy *covariance*, so a
   correlated joint deviation stands out even when no single sensor has.
   Under the healthy-region Gaussian assumption, squared Mahalanobis
   distance ~ chi-square(p), which gives a principled statistical threshold
   rather than a hand-picked one.
3. **PCA reconstruction error** — fit PCA on standardized healthy data,
   project each hour onto the top components and back, and flag large
   reconstruction error. Complementary to Mahalanobis: it's robust to
   collinear sensors and tends to catch the same correlated-drift pattern
   from a different angle.
4. **Isolation Forest** — a modern tree-based anomaly detector, included as
   the nonparametric alternative that makes no Gaussian assumption at all.

`ensemble_vote()` flags an hour only when at least 2 of the 3 "advanced"
detectors (Mahalanobis, PCA, Isolation Forest) agree — cutting down on any
single method's false alarms without losing much sensitivity, which
`evaluate_detection.py` measures directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

SENSOR_COLS = ["temperature", "vibration_rms", "pressure", "current_draw", "rotation_speed"]


class Reference:
    """Statistics fit once on the healthy training window, reused by every detector."""

    def __init__(self, healthy: pd.DataFrame, sensor_cols: list[str] = SENSOR_COLS):
        self.sensor_cols = sensor_cols
        self.mean = healthy[sensor_cols].mean()
        self.std = healthy[sensor_cols].std().replace(0, 1e-6)
        self.cov = healthy[sensor_cols].cov()
        self.inv_cov = np.linalg.pinv(self.cov.values)

    def standardize(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[self.sensor_cols] - self.mean) / self.std).values


def fit_reference(sensors: pd.DataFrame, healthy_window_hours: int) -> Reference:
    healthy = sensors[sensors["hour"] < healthy_window_hours]
    return Reference(healthy)


def zscore_detector(df: pd.DataFrame, ref: Reference, z_threshold: float = 3.0) -> pd.DataFrame:
    z = ((df[ref.sensor_cols] - ref.mean) / ref.std).abs()
    score = z.max(axis=1).values
    return pd.DataFrame({"score": score, "flag": score > z_threshold}, index=df.index)


def mahalanobis_detector(df: pd.DataFrame, ref: Reference, alpha: float = 0.01) -> pd.DataFrame:
    diff = df[ref.sensor_cols].values - ref.mean.values
    d2 = np.einsum("ij,jk,ik->i", diff, ref.inv_cov, diff)
    threshold = chi2.ppf(1 - alpha, df=len(ref.sensor_cols))
    return pd.DataFrame({"score": d2, "flag": d2 > threshold}, index=df.index)


def pca_detector(df: pd.DataFrame, ref: Reference, healthy: pd.DataFrame, n_components: int = 3,
                  percentile: float = 99.5) -> pd.DataFrame:
    X_train = ref.standardize(healthy)
    pca = PCA(n_components=n_components, random_state=0).fit(X_train)
    recon_train = pca.inverse_transform(pca.transform(X_train))
    train_error = np.sum((X_train - recon_train) ** 2, axis=1)
    threshold = np.percentile(train_error, percentile)

    X = ref.standardize(df)
    recon = pca.inverse_transform(pca.transform(X))
    error = np.sum((X - recon) ** 2, axis=1)
    return pd.DataFrame({"score": error, "flag": error > threshold}, index=df.index)


def isolation_forest_detector(df: pd.DataFrame, ref: Reference, healthy: pd.DataFrame,
                               contamination: float = 0.01, random_state: int = 0) -> pd.DataFrame:
    X_train = ref.standardize(healthy)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    model.fit(X_train)

    X = ref.standardize(df)
    score = -model.score_samples(X)  # flip sign: higher = more anomalous, consistent with the other detectors
    train_score = -model.score_samples(X_train)
    threshold = np.percentile(train_score, 100 - contamination * 100)
    return pd.DataFrame({"score": score, "flag": score > threshold}, index=df.index)


def run_all_detectors(df: pd.DataFrame, ref: Reference, healthy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "zscore": zscore_detector(df, ref),
        "mahalanobis": mahalanobis_detector(df, ref),
        "pca": pca_detector(df, ref, healthy),
        "isolation_forest": isolation_forest_detector(df, ref, healthy),
    }


def ensemble_vote(detector_flags: dict[str, pd.Series], advanced_only=("mahalanobis", "pca", "isolation_forest"),
                   min_votes: int = 2) -> pd.Series:
    votes = sum(detector_flags[name].astype(int) for name in advanced_only)
    return votes >= min_votes
