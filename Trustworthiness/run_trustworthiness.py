"""
Run the trustworthiness stage.

For every base model it computes calibration before and after temperature
scaling, and it builds RAPS conformal prediction sets with a ninety percent
coverage target. It saves a calibration results table, a conformal results
table, reliability diagrams and abstention plots.

It uses the calibration split to fit the temperature and to calibrate the
conformal threshold, and the test split to report the final numbers. Run the
test probability step first.

Run:
    python Trustworthiness/predict_test.py        (once)
    python Trustworthiness/run_trustworthiness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import (
    CM_PRED_DIR, RESULTS_DIR, PLOTS_DIR, BASE_MODELS,
    ECE_N_BINS, ALPHA,
)
from calibration import calibrate_model
from conformal import run_conformal, plot_abstention_by_confidence

_SUFFIX = {"val": "val", "calibration": "cal", "test": "test"}


def _load(safe, split):
    f = CM_PRED_DIR / f"{safe}_{_SUFFIX[split]}_probs.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Missing {f}. Run predict_test.py and the classical stage first.")
    d = pd.read_csv(f)
    return d["y_true"].values, d["y_prob"].values


def main():
    cal_rows, conf_rows = [], []

    for safe, label, _ in BASE_MODELS:
        print(f"\n=== {label} ===")
        y_cal, p_cal = _load(safe, "calibration")
        y_test, p_test = _load(safe, "test")

        # Calibration
        cal, p_test_cal, p_cal_cal = calibrate_model(
            y_cal, p_cal, y_test, p_test, label,
            n_bins=ECE_N_BINS, plot_path=PLOTS_DIR / f"reliability_{safe}.png")
        cal_rows.append(cal)
        print(f"  temperature {cal['temperature']}, "
              f"ECE {cal['ece_before']} -> {cal['ece_after']}")

        # Conformal prediction on the raw model probabilities. The coverage
        # guarantee holds on the model scores directly, and keeping it separate
        # from temperature scaling avoids the two analyses interacting.
        ev, sets_test = run_conformal(
            y_cal, p_cal, y_test, p_test, alpha=ALPHA, label=label)
        conf_rows.append(ev)
        print(f"  conformal qhat {ev['qhat']}, coverage {ev['coverage']}, "
              f"avg set size {ev['avg_set_size']}, abstention {ev['abstention']}")
        plot_abstention_by_confidence(
            p_test, sets_test, label, PLOTS_DIR / f"abstention_{safe}.png")

    cal_df = pd.DataFrame(cal_rows)
    conf_df = pd.DataFrame(conf_rows).sort_values("abstention")
    cal_df.to_csv(RESULTS_DIR / "calibration_results.csv", index=False)
    conf_df.to_csv(RESULTS_DIR / "conformal_results.csv", index=False)

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS (test set)")
    print("=" * 70)
    print(cal_df.to_string(index=False))
    print("\n" + "=" * 70)
    print(f"CONFORMAL RESULTS (RAPS, target coverage {1-ALPHA:.0%}, test set)")
    print("=" * 70)
    print(conf_df.to_string(index=False))
    print(f"\nSaved tables and plots under {RESULTS_DIR}")
    print("\nTrustworthiness stage complete.")


if __name__ == "__main__":
    main()
