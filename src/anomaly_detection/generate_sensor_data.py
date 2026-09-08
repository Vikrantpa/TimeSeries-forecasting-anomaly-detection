"""
generate_sensor_data.py
=========================
Synthetic hourly multivariate sensor logs for a fleet of 6 machines over 60
days (1,440 hours): temperature, vibration_rms, pressure, current_draw, and
rotation_speed. The five sensors share a one-factor correlation structure (a
common "load" factor drives temperature/vibration/current up together, with
rotation_speed weakly anti-correlated) — this matters because it's exactly
the kind of correlated multivariate signature a single-sensor threshold
can't see, but Mahalanobis distance / PCA reconstruction error can.

Two of the six machines ("M3", "M5") get an injected fault: starting at a
hidden `fault_start_hour`, a slow, accelerating (quadratic) degradation
appears jointly across temperature/vibration/current — a stand-in for
something like bearing wear — culminating in a `failure_hour` terminal
event (a sharp erratic spike), after which that machine's log is truncated
(it went offline). The other four machines run clean for the full window,
which is what lets `evaluate_detection.py` measure each detector's
false-positive rate on data with nothing wrong.

The ground truth (`fault_start_hour`, `failure_hour`) is saved to
manifest.json — in the real world you'd only learn `failure_hour` (when the
machine actually broke), never the true onset. That's exactly the point of
`evaluate_detection.py`'s "hours of advance warning before failure" metric.

Usage:
    python -m src.anomaly_detection.generate_sensor_data --seed 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

N_HOURS = 1440  # 60 days
MACHINES = ["M1", "M2", "M3", "M4", "M5", "M6"]
FAULTY_MACHINES = {"M3": {"fault_start_hour": 620, "failure_hour": 860},
                   "M5": {"fault_start_hour": 940, "failure_hour": 1120}}
SENSORS = ["temperature", "vibration_rms", "pressure", "current_draw", "rotation_speed"]

BASELINE = {"temperature": 65.0, "vibration_rms": 0.8, "pressure": 100.0, "current_draw": 15.0, "rotation_speed": 1800.0}
DAILY_AMPLITUDE = {"temperature": 3.0, "vibration_rms": 0.05, "pressure": 2.0, "current_draw": 1.2, "rotation_speed": 15.0}
IDIOSYNCRATIC_STD = {"temperature": 0.8, "vibration_rms": 0.04, "pressure": 1.8, "current_draw": 0.4, "rotation_speed": 8.0}
# one-factor loadings (shared "load" factor) -> guarantees a valid PSD covariance by construction
FACTOR_LOADING = {"temperature": 0.9, "vibration_rms": 0.03, "pressure": 0.3, "current_draw": 0.6, "rotation_speed": -3.0}
FACTOR_STD = 1.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sensors")


def simulate_machine(machine_id: str, rng: np.random.Generator) -> pd.DataFrame:
    hours = np.arange(N_HOURS)
    hour_of_day = hours % 24
    daily_cycle = np.sin(2 * np.pi * (hour_of_day - 9) / 24)  # peaks mid-shift

    common_factor = rng.normal(0, FACTOR_STD, N_HOURS)
    slow_drift = np.cumsum(rng.normal(0, 0.002, N_HOURS))  # tiny common wear, same for every machine

    data = {"hour": hours, "machine_id": machine_id}
    for sensor in SENSORS:
        idiosyncratic = rng.normal(0, IDIOSYNCRATIC_STD[sensor], N_HOURS)
        values = (
            BASELINE[sensor]
            + DAILY_AMPLITUDE[sensor] * daily_cycle
            + FACTOR_LOADING[sensor] * common_factor
            + idiosyncratic
            + (slow_drift if sensor != "rotation_speed" else -slow_drift * 50)
        )
        data[sensor] = values
    df = pd.DataFrame(data)

    fault = FAULTY_MACHINES.get(machine_id)
    if fault is None:
        df["has_fault"] = False
        return df

    start, fail = fault["fault_start_hour"], fault["failure_hour"]
    ramp = np.zeros(N_HOURS)
    in_fault = hours >= start
    progress = np.clip((hours - start) / max(fail - start, 1), 0, 1)
    ramp[in_fault] = progress[in_fault] ** 2  # quadratic: slow onset, accelerating

    # The early fault signature is deliberately "off-manifold" rather than just large: temperature
    # and vibration rise together while current_draw moves the *opposite* way from what their normal
    # positive correlation (via the shared load factor) would predict. That's what makes it hard for
    # a per-sensor threshold to catch early (each sensor's shift stays modest for a while) but easy
    # for a covariance-aware detector to catch early (the joint pattern is unusual even when no single
    # sensor is, which is the entire point of this module).
    df["temperature"] += ramp * 7.0
    df["vibration_rms"] += ramp * 0.35
    df["current_draw"] -= ramp * 3.2
    df["rotation_speed"] -= ramp * 55.0

    # terminal failure burst: a few hours of erratic spiking right at/after failure_hour
    terminal_start, terminal_end = fail, min(fail + 6, N_HOURS)
    terminal_mask = (hours >= terminal_start) & (hours < terminal_end)
    n_terminal = terminal_mask.sum()
    if n_terminal:
        df.loc[terminal_mask, "temperature"] += rng.normal(15, 5, n_terminal)
        df.loc[terminal_mask, "vibration_rms"] += rng.normal(3.0, 1.0, n_terminal)
        df.loc[terminal_mask, "current_draw"] += rng.normal(6.0, 2.0, n_terminal)
        df.loc[terminal_mask, "rotation_speed"] -= rng.normal(200, 60, n_terminal)

    # machine goes offline shortly after the terminal burst — truncate its log
    cutoff = min(fail + 8, N_HOURS)
    df = df.iloc[:cutoff].copy()
    df["has_fault"] = True
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multivariate sensor logs.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = [simulate_machine(m, rng) for m in MACHINES]
    sensors = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(args.out_dir, "sensors.csv")
    sensors.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sensors)} rows, {sensors['machine_id'].nunique()} machines)")

    manifest = {
        "n_hours": N_HOURS,
        "machines": MACHINES,
        "faulty_machines": FAULTY_MACHINES,
        "healthy_training_window_hours": 400,  # first N hours assumed fault-free for every machine — used to fit detectors
        "sensors": SENSORS,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote manifest.json")

    for m, g in sensors.groupby("machine_id"):
        fault_note = f"FAULTY (starts {FAULTY_MACHINES[m]['fault_start_hour']}h, fails {FAULTY_MACHINES[m]['failure_hour']}h)" if m in FAULTY_MACHINES else "healthy"
        print(f"  {m}: {len(g)} hours — {fault_note}")


if __name__ == "__main__":
    main()
