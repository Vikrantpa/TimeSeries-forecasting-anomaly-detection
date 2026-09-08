# Architecture

## 1. Forecasting: the state space model

Each series is modeled in log1p-space (`y_t = log(1 + demand_t)`, transformed back
via `expm1` at the end — a monotonic transform, so **quantiles** carry through
exactly even though the mean would not) as a 4-dimensional linear-Gaussian state
space model:

```
state x_t = [level_t, trend_t, s1_t, s2_t]

level_t = level_{t-1} + trend_{t-1} + w_level
trend_t = trend_{t-1}                + w_trend
[s1_t, s2_t] = R(θ) [s1_{t-1}, s2_{t-1}]   + [w_s1, w_s2]     (θ = 2π/7, a weekly rotation)

y_t = level_t + s1_t + v_t
```

`R(θ)` is a 2D rotation matrix; rotating a 2D state at the weekly frequency is the
standard "trigonometric seasonal" component used in BSTS software (Google's `bsts`,
`statsmodels.UnobservedComponents`), just written out directly. `w_* ~ N(0, Q)` is
process noise (`Q = diag(q_level, q_trend, q_season, q_season)`), `v_t ~ N(0, R_obs)`
is observation noise.

### Kalman filter recursions (`kalman_filter()` in `bayesian_structural_model.py`)

For each time step, given the transition matrix `T` and observation vector `Z = [1,0,1,0]`:

```
predict:  a_pred = T @ a_{t-1}            P_pred = T @ P_{t-1} @ T' + Q
update:   v = y_t - Z @ a_pred            (innovation)
          F = Z @ P_pred @ Z' + R_obs     (innovation variance)
          K = P_pred @ Z' / F             (Kalman gain)
          a_t = a_pred + K v              P_t = P_pred - K Z P_pred
loglik += -0.5 (log(2π F) + v²/F)
```

Forecasting `h` steps ahead is the same predict step repeated with no update (no
new observations), so the state uncertainty `P` — and therefore the predictive
interval — necessarily widens with horizon (a property `tests/test_forecasting.py`
checks directly). This is a real, structural advantage over the recursive quantile-
GBM approach, whose interval width doesn't have this compounding-uncertainty
property built in (see below).

### Fitting the hyperparameters

`fit_series_mle()` finds `(q_level, q_trend, q_season, r_obs)` — in log-space, to
keep them positive — by maximizing the Kalman filter's data log-likelihood via
`scipy.optimize.minimize` (L-BFGS-B). With enough data this is well-identified.
With very little data (this project's new-store series has 17 days — under 3 full
weekly cycles) it is not: the optimizer can drive `q_level` up and `r_obs` down to
near zero, producing a state that interpolates the training data almost perfectly
but whose forecast variance compounds unchecked going forward. That's exactly what
happens in this repo's run (see `raw_log_params` for S5_P1 in a debug print, or
just look at `outputs/forecasting/pooling_effect.png`).

### Hierarchical shrinkage (the empirical-Bayes step)

```
group_mean = average of all series' raw log-hyperparameters
weight_i   = SHRINKAGE_C / (SHRINKAGE_C + n_obs_i)      (SHRINKAGE_C = 250)
shrunk_i   = weight_i * group_mean + (1 - weight_i) * raw_i
```

This is a two-stage empirical-Bayes approximation to a fully joint hierarchical
model, not MCMC over a joint posterior — the honest way to describe it is
"borrow the population's typical hyperparameters, weighted by how little you
trust your own series' estimate," which is the same statistical intuition behind
partial pooling, computed in closed form instead of by sampling. A series with
`n_obs = 2000` gets `weight ≈ 0.11` (barely shrunk); a series with `n_obs = 17`
gets `weight ≈ 0.94` (shrunk almost entirely to the population average). What it
buys you here specifically: the short series's *interval width* becomes sane
again, because `r_obs` and `q_level` — the two hyperparameters that were driven to
a degenerate extreme by 17 noisy points — get pulled back to what long, reliably-
estimated series say those values should typically be.

### Quantile-GBM's recursive forecasting limitation

`quantile_gbm_model.py` predicts one step at a time and feeds its own median
(`q50`) prediction back in as the next step's lag feature, since the true future
values aren't known. This is standard practice but has a real weakness: because
each step's *input* is a point estimate (not a distribution), the model's
predicted interval width doesn't automatically grow with horizon the way the
Kalman filter's does — uncertainty about "was my own recursive input right"
doesn't propagate into the quantile predictions. A more correct (and heavier)
fix is direct multi-horizon modeling (train a separate model per horizon) or
proper path simulation (sample many plausible future paths and take empirical
quantiles across them, rather than feeding back one point estimate).

## 2. Anomaly detection: the multivariate math

### Mahalanobis distance

Fit a mean vector `μ` and covariance matrix `Σ` on the healthy reference window.
For any new observation `x`:

```
d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Under the (approximate) assumption that healthy sensor readings are jointly
Gaussian, `d²` follows a chi-square distribution with `p` degrees of freedom
(`p` = number of sensors), which gives a principled statistical threshold —
`scipy.stats.chi2.ppf(1 - alpha, df=p)` — rather than a hand-tuned cutoff. This
is the entire mechanism that catches a correlated, "off-manifold" deviation
that no single sensor's z-score would flag: `Σ⁻¹` weights deviations *against*
the normal correlation structure much more heavily than deviations *along* it.

### PCA reconstruction error

Standardize the healthy data (mean/std from the reference window — necessary
since raw sensor units span very different scales, e.g. RPM vs. vibration),
fit PCA keeping the top `k` components, and for any observation compute:

```
error = || x_standardized - PCA⁻¹(PCA(x_standardized)) ||²
```

A point that lies off the healthy data's dominant subspace has high
reconstruction error even if its raw magnitude isn't extreme — a different
lens on the same "unusual joint pattern" idea as Mahalanobis distance, useful
because it degrades more gracefully when sensors are highly collinear (where
a raw covariance inverse can become numerically unstable).

### Why the injected fault is deliberately NOT just "big values"

The fault signature makes temperature and vibration rise while current draw
moves in the *opposite* direction from what the sensors' normal shared "load
factor" correlation would predict. Early in the fault, each individual sensor's
deviation is only ~1–2 standard deviations — comfortably under a typical 3-sigma
single-sensor alarm — while the *joint* pattern is already well outside the
healthy region's covariance structure. That gap is exactly what this project
measures: on machine M3, the per-sensor z-score baseline gives 65 hours of
advance warning before failure; Mahalanobis distance gives 115 hours. Same
underlying degradation, ~77% more lead time from using the right (multivariate)
lens on it.

### Evaluation metrics (`evaluate_detection.py`)

- **Precision/recall/F1**: each hour is a binary instance, ground truth = 1 for
  every hour from the true (hidden) `fault_start_hour` onward, 0 before it and
  for every hour of a healthy machine.
- **False positive rate**: fraction of hours flagged among healthy machines only
  — the cost side of the trade-off precision/recall alone can obscure.
- **Advance warning before failure**: `failure_hour - first_sustained_alarm_hour`,
  where a "sustained" alarm requires 3+ consecutive flagged hours (so a single
  noisy blip doesn't get credited as a detection) — this is the number a
  maintenance team actually cares about: how much runway did we have to act?

### Ensemble

`ensemble_vote()` flags an hour only when at least 2 of the 3 covariance-aware
detectors (Mahalanobis, PCA, Isolation Forest) agree. This trades a little
recall/lead-time for a lower false-positive rate than any single advanced
detector alone (0.49% vs. 0.71–1.00%), which is usually the right trade for an
alert a human has to act on.
