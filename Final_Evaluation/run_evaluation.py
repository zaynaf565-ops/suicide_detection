"""
Final evaluation on the held out test set.

This is the only place the test set is scored. For every base model and for the
two ensembles it reports the metrics at the threshold that was tuned on the
validation set, never on the test set. It adds two things the validation results
did not have. First, bootstrap confidence intervals on the F2 score, so the
reader can see whether the differences between models are real or within noise.
Second, a McNemar test between the strongest models, which checks whether their
errors differ in a statistically meaningful way.

Run:
    python Final_Evaluation/run_evaluation.py
Requires the test probabilities from Trustworthiness/predict_test.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from config import (
    CM_PRED_DIR, CM_RESULTS, ENS_RESULTS, ENS_MODELS, RESULTS_DIR, PLOTS_DIR,
    RANDOM_SEED, FBETA, N_BOOTSTRAP, CI_LEVEL, BASE_MODELS,
)
from metrics import compute_all_metrics

plt.rcParams.update({"font.family": "serif", "font.size": 10})


# ---------------------------------------------------------------------------
# Load tuned thresholds from the saved results tables
# ---------------------------------------------------------------------------
def _classical_thresholds():
    files = sorted(CM_RESULTS.glob("classical_ml_val_results_*.csv"))
    if not files:
        return {}
    df = pd.read_csv(files[-1])
    return dict(zip(df["model"], df["threshold"]))


def _ensemble_thresholds():
    f = ENS_RESULTS / "ensemble_val_results.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return dict(zip(df["model"], df["threshold"]))


# ---------------------------------------------------------------------------
# Build test probabilities for base models and ensembles
# ---------------------------------------------------------------------------
def load_test_probs():
    probs, y = {}, None
    for safe, label in BASE_MODELS:
        f = CM_PRED_DIR / f"{safe}_test_probs.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"Missing {f}. Run Trustworthiness/predict_test.py first.")
        d = pd.read_csv(f)
        probs[label] = d["y_prob"].values
        if y is None:
            y = d["y_true"].values
    return probs, y


def add_ensemble_probs(probs):
    """Build ensemble test probabilities from the base model test probabilities."""
    order = [label for _, label in BASE_MODELS]
    P = np.column_stack([probs[l] for l in order])

    wf = ENS_RESULTS / "weighted_voting_weights.csv"
    if wf.exists():
        wdf = pd.read_csv(wf).set_index("model").reindex(order)
        w = wdf["weight"].values
        probs["Weighted Voting"] = (P @ w) / w.sum()

    sf = ENS_MODELS / "stacking.joblib"
    if sf.exists():
        obj = joblib.load(sf)
        meta, scaler = obj["meta"], obj["scaler"]
        probs["Stacking"] = meta.predict_proba(scaler.transform(P))[:, 1]
    return probs


# ---------------------------------------------------------------------------
# Bootstrap and McNemar
# ---------------------------------------------------------------------------
def fbeta(y, yhat, beta=FBETA):
    tp = np.sum((yhat == 1) & (y == 1))
    fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * prec + rec
    return (1 + b2) * prec * rec / denom if denom else 0.0


def bootstrap_f2_ci(y, y_prob, thresh, n=N_BOOTSTRAP, level=CI_LEVEL, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    yhat = (y_prob >= thresh).astype(int)
    N = len(y)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        scores[i] = fbeta(y[idx], yhat[idx])
    lo = np.percentile(scores, 100 * (1 - level) / 2)
    hi = np.percentile(scores, 100 * (1 + level) / 2)
    return float(scores.mean()), float(lo), float(hi)


def mcnemar(y, pred_a, pred_b):
    """Two sided McNemar test on the discordant pairs."""
    from scipy.stats import chi2
    a_correct = pred_a == y
    b_correct = pred_b == y
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)   # continuity corrected
    p = float(chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": round(stat, 3), "p_value": p}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    probs, y = load_test_probs()
    probs = add_ensemble_probs(probs)

    thresholds = {**_classical_thresholds(), **_ensemble_thresholds()}
    default_t = 0.5

    rows, preds = [], {}
    for label, p in probs.items():
        t = float(thresholds.get(label, default_t))
        yhat = (p >= t).astype(int)
        preds[label] = yhat
        m = compute_all_metrics(y, yhat, p, t)
        mean_f2, lo, hi = bootstrap_f2_ci(y, p, t)
        m.update({"model": label, "f2_boot_mean": round(mean_f2, 4),
                  "f2_ci_low": round(lo, 4), "f2_ci_high": round(hi, 4)})
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("f2", ascending=False)
    cols = ["model", "f2", "f2_ci_low", "f2_ci_high", "recall", "precision",
            "f1", "roc_auc", "accuracy", "threshold"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 90)
    print("FINAL TEST RESULTS (sorted by F2, with bootstrap 95% CI on F2)")
    print("=" * 90)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    df.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)

    # F2 bar chart with bootstrap CI
    dfp = df.sort_values("f2")
    err = np.vstack([dfp["f2"] - dfp["f2_ci_low"], dfp["f2_ci_high"] - dfp["f2"]])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(dfp["model"], dfp["f2"], xerr=err, color="#4c72b0",
            edgecolor="black", linewidth=0.5, error_kw=dict(ecolor="#333333", capsize=3))
    for yy, v in enumerate(dfp["f2"]):
        ax.text(v - 0.005, yy, f"{v:.4f}", va="center", ha="right", color="white", fontsize=8)
    ax.set_xlabel("Test F2 score with 95% bootstrap confidence interval")
    ax.set_xlim(0.85, 0.96); ax.set_title("Final test performance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "f2_bootstrap_ci.png", dpi=160); plt.close()
    print(f"\nSaved F2 bootstrap CI chart")

    # McNemar between the top model and the others
    top = df.iloc[0]["model"]
    mc_rows = []
    for label in probs:
        if label == top:
            continue
        r = mcnemar(y, preds[top], preds[label])
        r["model_a"], r["model_b"] = top, label
        mc_rows.append(r)
    mc = pd.DataFrame(mc_rows)[["model_a", "model_b", "n01", "n10", "statistic", "p_value"]]
    print("\n" + "=" * 90)
    print(f"McNEMAR TEST, {top} versus each other model (test set)")
    print("=" * 90)
    print(mc.to_string(index=False, float_format="{:.4g}".format))
    mc.to_csv(RESULTS_DIR / "mcnemar_results.csv", index=False)

    print(f"\nSaved all results under {RESULTS_DIR}")
    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()
