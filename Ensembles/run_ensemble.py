"""
Run the ensemble stage.

Loads the per model validation and calibration probabilities, builds the
weighted voting and stacking ensembles, evaluates them on the validation set,
and saves the ensemble probabilities, the learned weights and coefficients, and
a results table. The ensemble validation and calibration probabilities are saved
because the trustworthiness and evaluation stages may use them.

Run:
    python Ensembles/run_ensemble.py
or from inside Ensembles/:
    python run_ensemble.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Ensembles"))

import numpy as np
import pandas as pd
import joblib

from config import (
    BASE_MODELS, MODEL_LABELS, MODELS_DIR, RESULTS_DIR, PRED_OUT_DIR,
)
from metrics import compute_all_metrics, tune_threshold, print_metrics, results_to_dataframe
from ensemble import (
    load_prob_matrix, weighted_average, optimise_weights, train_stacking,
)


def _evaluate(y, y_prob, label):
    t, _, thr = tune_threshold(y, y_prob, metric="f2")
    m = compute_all_metrics(y, (y_prob >= t).astype(int), y_prob, t)
    print_metrics(m, label)
    return m, t, thr


def main():
    print("Loading per model probabilities...")
    P_val, y_val = load_prob_matrix("val")
    P_cal, y_cal = load_prob_matrix("calibration")
    print(f"  val matrix {P_val.shape}, calibration matrix {P_cal.shape}")

    results = {}

    # --- Uniform average (reference) ---
    print("\n--- Uniform average ---")
    yv = weighted_average(P_val, np.ones(P_val.shape[1]))
    results["Uniform Average"], t_u, _ = _evaluate(y_val, yv, "Uniform Average")

    # --- Weighted voting ---
    print("\n--- Weighted voting (Nelder-Mead) ---")
    w = optimise_weights(P_val, y_val, n_restarts=10)
    yv_w = weighted_average(P_val, w)
    yc_w = weighted_average(P_cal, w)
    results["Weighted Voting"], t_w, _ = _evaluate(y_val, yv_w, "Weighted Voting")
    weights_df = pd.DataFrame({"model": [MODEL_LABELS[m] for m in BASE_MODELS], "weight": w}) \
        .sort_values("weight", ascending=False)
    print("  Learned weights:")
    print(weights_df.to_string(index=False))
    weights_df.to_csv(RESULTS_DIR / "weighted_voting_weights.csv", index=False)

    # --- Stacking ---
    print("\n--- Stacking (logistic regression meta learner) ---")
    meta, scaler, best_c, cv_f2, coef = train_stacking(P_val, y_val)
    yv_s = meta.predict_proba(scaler.transform(P_val))[:, 1]
    yc_s = meta.predict_proba(scaler.transform(P_cal))[:, 1]
    results["Stacking"], t_s, _ = _evaluate(y_val, yv_s, "Stacking")
    print("  Meta learner coefficients:")
    print(coef.to_string(index=False))
    coef.to_csv(RESULTS_DIR / "stacking_coefficients.csv", index=False)
    joblib.dump({"meta": meta, "scaler": scaler, "C": best_c}, MODELS_DIR / "stacking.joblib")

    # --- Save ensemble probabilities for later stages ---
    pd.DataFrame({"y_true": y_val, "weighted_prob": yv_w, "stacking_prob": yv_s}) \
        .to_csv(PRED_OUT_DIR / "ensemble_val_probs.csv", index=False)
    pd.DataFrame({"y_true": y_cal, "weighted_prob": yc_w, "stacking_prob": yc_s}) \
        .to_csv(PRED_OUT_DIR / "ensemble_cal_probs.csv", index=False)

    # --- Results table ---
    df = results_to_dataframe(results)
    cols = ["model", "f2", "f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy", "threshold"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS (validation set, sorted by F2)")
    print("=" * 80)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 80)
    df.to_csv(RESULTS_DIR / "ensemble_val_results.csv", index=False)
    print(f"\nSaved results, weights, coefficients and probabilities under {RESULTS_DIR}")
    print("\nEnsemble stage complete.")


if __name__ == "__main__":
    main()
