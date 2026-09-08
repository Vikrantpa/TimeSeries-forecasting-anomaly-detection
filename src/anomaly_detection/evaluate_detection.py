"""
evaluate_detection.py
=======================
Runs every detector (+ the ensemble) on the full synthetic fleet and reports
the metrics that actually matter for predictive maintenance — not just
precision/recall, but **how many hours of advance warning** each detector
would have given before the machine actually failed, and how often each one
cries wolf on machines with nothing wrong.

"Detected" here means a *sustained* alarm (>= 3 consecutive flagged hours),
not a single flagged hour — a lone flagged hour is exactly the kind of noise
a real alerting system needs to suppress, so crediting a detector for one
lucky hour would overstate how useful it actually is.

Usage:
    python -m src.anomaly_detection.evaluate_detection
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.anomaly_detection.detectors import ensemble_vote, fit_reference, run_all_detectors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sensors")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "anomaly_detection")

BLUE, ORANGE, RED, INK, GRAY = "#2a78d6", "#eb6834", "#d03b3b", "#0b0b0b", "#898781"
MIN_CONSECUTIVE = 3


def first_sustained_alarm(flags: np.ndarray, min_consecutive: int = MIN_CONSECUTIVE) -> int | None:
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= min_consecutive:
            return i - min_consecutive + 1
    return None


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sensors = pd.read_csv(os.path.join(DATA_DIR, "sensors.csv"))
    with open(os.path.join(DATA_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    faulty = manifest["faulty_machines"]
    healthy_window = manifest["healthy_training_window_hours"]

    ref = fit_reference(sensors, healthy_window)
    healthy_ref_df = sensors[sensors["hour"] < healthy_window]

    all_flags: dict[str, dict[str, pd.DataFrame]] = {}
    rows = []
    for machine, g in sensors.groupby("machine_id"):
        df = g.sort_values("hour").reset_index(drop=True)
        detectors = run_all_detectors(df, ref, healthy_ref_df)
        detectors["ensemble"] = pd.DataFrame(
            {"flag": ensemble_vote({k: v["flag"] for k, v in detectors.items()}),
             "score": sum(detectors[k]["flag"].astype(int) for k in ("mahalanobis", "pca", "isolation_forest"))}
        )
        all_flags[machine] = detectors

        is_faulty = machine in faulty
        y_true = np.zeros(len(df), dtype=bool)
        if is_faulty:
            y_true[df["hour"] >= faulty[machine]["fault_start_hour"]] = True

        for name, d in detectors.items():
            y_pred = d["flag"].values
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            fpr = float((y_pred & ~y_true).sum() / max((~y_true).sum(), 1))

            advance_warning = None
            if is_faulty:
                alarm_idx = first_sustained_alarm(y_pred)
                if alarm_idx is not None:
                    advance_warning = faulty[machine]["failure_hour"] - df["hour"].iloc[alarm_idx]

            rows.append({
                "machine": machine, "detector": name, "is_faulty": is_faulty,
                "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                "false_positive_rate": round(fpr, 4),
                "advance_warning_hours": advance_warning,
            })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "detection_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}\n")

    print("=== False positive rate on HEALTHY machines (lower is better) ===")
    fpr_table = summary[~summary["is_faulty"]].groupby("detector")["false_positive_rate"].mean().sort_values()
    print(fpr_table.round(4).to_string())

    print("\n=== Advance warning before failure, in hours (higher is better) ===")
    warn_table = summary[summary["is_faulty"]][["machine", "detector", "advance_warning_hours", "recall", "f1"]]
    print(warn_table.to_string(index=False))

    plot_detection_traces(sensors, faulty, all_flags)


def plot_detection_traces(sensors: pd.DataFrame, faulty: dict, all_flags: dict) -> None:
    fig, axes = plt.subplots(len(faulty), 2, figsize=(13, 4.5 * len(faulty)))
    if len(faulty) == 1:
        axes = axes.reshape(1, -1)

    for row, (machine, meta) in enumerate(faulty.items()):
        df = sensors[sensors.machine_id == machine].sort_values("hour").reset_index(drop=True)
        detectors = all_flags[machine]

        ax = axes[row, 0]
        ax.plot(df["hour"], df["temperature"], color=INK, linewidth=1, label="temperature")
        ax2 = ax.twinx()
        ax2.plot(df["hour"], df["vibration_rms"], color=ORANGE, linewidth=1, label="vibration_rms")
        ax.axvline(meta["fault_start_hour"], color=GRAY, linestyle="--", linewidth=1, label="true fault onset")
        ax.axvline(meta["failure_hour"], color=RED, linestyle="--", linewidth=1, label="failure")
        ax.set_title(f"{machine}: raw sensors")
        ax.set_ylabel("temperature (°C)")
        ax2.set_ylabel("vibration_rms")
        ax.legend(fontsize=7, loc="upper left")

        ax = axes[row, 1]
        maha = detectors["mahalanobis"]["score"].values
        ax.plot(df["hour"], maha, color=BLUE, linewidth=1, label="Mahalanobis distance²")
        threshold = maha[df["hour"] < 400].mean() + 4 * maha[df["hour"] < 400].std()
        ax.axhline(threshold, color=GRAY, linestyle=":", linewidth=1, label="~healthy range")
        ax.axvline(meta["fault_start_hour"], color=GRAY, linestyle="--", linewidth=1)
        ax.axvline(meta["failure_hour"], color=RED, linestyle="--", linewidth=1)

        alarm_idx = None
        ens_flags = detectors["ensemble"]["flag"].values
        run = 0
        for i, fl in enumerate(ens_flags):
            run = run + 1 if fl else 0
            if run >= MIN_CONSECUTIVE:
                alarm_idx = i - MIN_CONSECUTIVE + 1
                break
        if alarm_idx is not None:
            ax.axvline(df["hour"].iloc[alarm_idx], color="#1baf7a", linewidth=2,
                       label=f"ensemble sustained alarm (hour {df['hour'].iloc[alarm_idx]})")
        ax.set_title(f"{machine}: Mahalanobis distance + ensemble alarm")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Predictive maintenance: correlated multivariate drift ahead of failure", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "detection_traces.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
