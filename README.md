# Probabilistic Demand Forecasting & Predictive-Maintenance Anomaly Detection

Two related time-series projects in one runnable repo:

1. **Probabilistic demand forecasting** — a hierarchical Bayesian structural time
   series model (Kalman filter, fit from scratch — no PyMC/Stan) compared against a
   pooled quantile-gradient-boosting model, both producing **uncertainty intervals,
   not just point forecasts**, across 10 store-product series.
2. **Multivariate anomaly detection for predictive maintenance** — four detectors
   (a naive per-sensor baseline, Mahalanobis distance, PCA reconstruction error,
   Isolation Forest) plus an ensemble, evaluated on synthetic multi-sensor machine
   logs by the metric that actually matters for maintenance: **how many hours of
   advance warning before failure** each one would have given.

Everything runs on synthetic data with `pip install` — no GPU, no PyMC/Stan build,
no cloud account.

## Quickstart

```bash
git clone <your-fork-url>
cd ts-forecasting-anomaly-detection
pip install -r requirements.txt

make pipeline   # generates both datasets, fits every model, saves charts + CSVs
make test       # pytest — Kalman filter, shrinkage direction, detectors, ensemble
```

Charts and result tables land in `outputs/forecasting/` and `outputs/anomaly_detection/`.

### Running pieces individually

```bash
# forecasting
python -m src.forecasting.generate_demand_data --seed 11
python -m src.forecasting.evaluate_forecasts        # fits both models, scores, plots

# anomaly detection
python -m src.anomaly_detection.generate_sensor_data --seed 5
python -m src.anomaly_detection.evaluate_detection   # fits all detectors, scores, plots
```

Sample data is already generated and committed under `data/` so you can explore the
evaluation scripts without regenerating it first; the generators reproduce it
byte-for-byte with the same `--seed`.

## 1. Demand forecasting: what's actually being compared

`src/forecasting/generate_demand_data.py` simulates 5 stores × 2 products (10
series) of daily demand — trend, weekly seasonality, occasional promotions,
over-dispersed (Gamma-Poisson) count noise. One store, **S5, only has 17 days of
history** before the 28-day forecast window — a recently-opened store, which is
exactly the scenario hierarchical pooling exists for.

**`bayesian_structural_model.py`** fits a local-level + local-trend + weekly-harmonic
state space model per series via a hand-written Kalman filter (in log1p space, so
forecasts stay positive), with hyperparameters found by maximum likelihood. The
**hierarchical** part: each series' hyperparameters are then shrunk toward the
across-series average, with shrinkage strength `250 / (250 + n_obs)` — a short
series gets pulled hard toward "what typical stores look like"; a 700-day series
barely moves. This is a two-stage empirical-Bayes approximation to a full joint
hierarchical model, not MCMC — see `docs/ARCHITECTURE.md` for exactly what that
trade-off buys and costs.

**`quantile_gbm_model.py`** is the "modern ML" alternative: a single
`GradientBoostingRegressor(loss="quantile")` per quantile, pooled across **all**
series at once (store/product as features), with lag/rolling/calendar/promo
features. It pools information differently — one global model with entity features,
instead of a shared prior — which is the real point of comparison.

### What actually happened when I ran it

With only 17 training days, the per-series MLE finds a degenerate fit: it explains
almost all variation as random-walk noise in the *level* and treats observation
noise as ≈0, which lets it interpolate the 17 points almost perfectly but causes
the forecast uncertainty to explode as that noise compounds forward with nothing
to stabilize it — by day 28 its 80%-interval upper bound is **> 20,000** for a
series whose actual values are ~100–200. Hierarchical shrinkage pulls the
hyperparameters back to sane, population-typical values and the interval stays
usable. Total pinball loss (lower is better) for that series: **446.4 (unpooled) →
37.5 (pooled)**, an ~92% reduction. See `outputs/forecasting/pooling_effect.png`.

| series | n_train | bayes (unpooled) | bayes (pooled) | quantile-GBM |
|---|---|---|---|---|
| S1_P1 | 672 | 44.6 | 44.1 | **42.9** |
| S4_P1 | 672 | 87.6 | 86.4 | **63.3** |
| S5_P1 | 17 | 446.4 | **37.5** | 35.4 |
| S5_P2 | 17 | 53.7 | 42.4 | **22.5** |

(full table in `outputs/forecasting/forecast_summary.csv`, pinball loss summed
across the 10th/50th/90th percentile, lower is better.) Two honest takeaways: pooling
overwhelmingly matters for the short series and is nearly a no-op for the long ones
(exactly as it should be); and the quantile-GBM is competitive-to-better everywhere,
partly because it can see the known-in-advance promotion calendar as a feature,
which the univariate structural model in this repo can't (a real limitation,
discussed in `docs/ARCHITECTURE.md`).

## 2. Predictive maintenance: what actually happened

`src/anomaly_detection/generate_sensor_data.py` simulates 6 machines logging 5
correlated sensors hourly for 60 days. Two machines get an injected fault: starting
at a hidden hour, temperature and vibration rise while current draw moves the
*opposite* way from what their normal correlation would predict — individually
each sensor's shift stays modest for a long stretch, but the **joint pattern** is
unusual well before any single sensor crosses a naive threshold. That gap is the
entire reason multivariate detection exists.

| machine | detector | advance warning before failure | recall |
|---|---|---|---|
| M3 | per-sensor z-score (baseline) | 65 hours | 0.30 |
| M3 | **Mahalanobis distance** | **115 hours** | 0.55 |
| M3 | PCA reconstruction error | 103 hours | 0.46 |
| M3 | Isolation Forest | 91 hours | 0.40 |
| M5 | per-sensor z-score (baseline) | 64 hours | 0.40 |
| M5 | **Mahalanobis distance** | **91 hours** | 0.57 |

("Advance warning" = hours between the first *sustained* alarm — 3+ consecutive
flagged hours, so a single noisy blip doesn't count — and the actual failure.) The
multivariate detector gives roughly **50–80% more lead time** than the naive
per-sensor baseline, at a comparable or lower false-positive rate on the four
healthy machines (0.49–1.00% of hours flagged, see
`outputs/anomaly_detection/detection_summary.csv`). `detection_traces.png` plots
the raw sensors and the Mahalanobis distance trace with the true fault onset,
failure point, and ensemble alarm all marked.

## Project structure

```
ts-forecasting-anomaly-detection/
├── data/
│   ├── demand/                       # generated + committed sample data
│   └── sensors/
├── src/
│   ├── forecasting/
│   │   ├── generate_demand_data.py
│   │   ├── bayesian_structural_model.py   # Kalman filter + hierarchical shrinkage
│   │   ├── quantile_gbm_model.py          # pooled quantile GBM
│   │   └── evaluate_forecasts.py          # pinball loss, coverage, fan charts
│   └── anomaly_detection/
│       ├── generate_sensor_data.py
│       ├── detectors.py                   # z-score, Mahalanobis, PCA, Isolation Forest, ensemble
│       └── evaluate_detection.py          # precision/recall, advance-warning metric, plots
├── tests/
│   ├── test_forecasting.py
│   └── test_anomaly_detection.py
├── outputs/forecasting/ , outputs/anomaly_detection/   # charts + result CSVs (generated)
├── docs/ARCHITECTURE.md              # the math: Kalman filter, shrinkage, Mahalanobis/PCA
├── .github/workflows/ci.yml
├── Makefile
└── requirements.txt
```

## Limitations & production swaps

- **Forecasting model** → for real scale, the natural next steps are Meta's Prophet
  (fast, handles holidays/regressors out of the box), GluonTS/AWS DeepAR, or an
  actual Temporal Fusion Transformer / N-BEATS (PyTorch Forecasting, Darts,
  NeuralForecast) — this project's from-scratch Kalman filter is a deliberately
  lightweight, dependency-free stand-in that still produces genuine probabilistic
  intervals; it does not use GPU-trained deep nets because that would need a much
  heavier install and real training time for a demo. For full hierarchical Bayesian
  inference (rather than this project's empirical-Bayes approximation), PyMC or Stan
  with a proper joint model and MCMC is the real thing.
- **Quantile GBM** here is recursive multi-step (feeds its own median forecast back
  in as next-step lag features) — a known, imperfect way to extend past one step;
  a direct multi-horizon model (separate model per horizon) or a proper
  probabilistic simulation (sampling paths rather than feeding back a point
  estimate) would be the production fix.
- **Anomaly detection** → at real fleet scale, an autoencoder or a graph neural
  network over sensor topology generalizes further than PCA/Mahalanobis, and
  a real deployment would also model label lag (you often don't know an alarm was
  a true or false positive until well after the fact) and seasonal/regime changes
  in "normal" behavior, neither of which this project's fixed healthy-reference-window
  approach handles.
- Both modules assume **known, complete data** — no missing sensor readings, no
  ingestion delay, no clock skew across machines — because that's its own separate
  (and substantial) engineering problem, not a modeling one.

## License

MIT — see [LICENSE](LICENSE).
