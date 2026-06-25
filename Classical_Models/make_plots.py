"""
Qualitative results for the classical models.

Run this after training. It reads the saved results table and per model
probability files and produces a set of comparison plots:

  1. F2 comparison across models
  2. Threshold tuning curves (F2, precision, recall against threshold)
  3. ROC curves for all models
  4. Precision recall curves for all models
  5. Confusion matrices at the tuned threshold

Calibration reliability diagrams, SHAP and LIME plots, and conformal abstention
plots are not produced here. They belong to the trustworthiness and
explainability phases later in the project.

Run:
    python make_plots.py    (from inside Classical_Models/)
"""
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from config import RESULTS_DIR, PRED_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "serif"


def _models_available():
    files = sorted(glob.glob(str(PRED_DIR / "*_val_probs.csv")))
    return [Path(f).name.replace("_val_probs.csv", "") for f in files]


def _nice(name):
    return name.replace("_", " ").title()


def plot_f2_comparison():
    res = sorted(glob.glob(str(RESULTS_DIR / "classical_ml_val_results_*.csv")))
    if not res:
        print("  no results table found, skipping F2 comparison")
        return
    df = pd.read_csv(res[-1]).sort_values("f2")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["model"], df["f2"], color="#4c72b0", edgecolor="black", linewidth=0.5)
    for y, v in enumerate(df["f2"]):
        ax.text(v - 0.01, y, f"{v:.4f}", va="center", ha="right", color="white", fontsize=9)
    ax.set_xlabel("Validation F2 score"); ax.set_xlim(0, 1)
    ax.set_title("Classical model comparison by F2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "f2_comparison.png", dpi=160); plt.close()
    print("  saved f2_comparison.png")


def plot_threshold_curves():
    models = _models_available()
    n = len(models)
    if n == 0:
        return
    cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        f = PRED_DIR / f"{m}_threshold_curve.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        ax.plot(d["threshold"], d["f2"], label="F2", color="#c44e52", lw=1.6)
        ax.plot(d["threshold"], d["precision"], label="Precision", color="#4c72b0", lw=1)
        ax.plot(d["threshold"], d["recall"], label="Recall", color="#55a868", lw=1)
        best = d.loc[d["f2"].idxmax(), "threshold"]
        ax.axvline(best, ls="--", color="black", lw=0.8)
        ax.set_title(_nice(m)); ax.set_xlabel("Threshold"); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Threshold tuning curves (validation)", y=1.02)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "threshold_curves.png", dpi=160, bbox_inches="tight"); plt.close()
    print("  saved threshold_curves.png")


def plot_roc_pr():
    models = _models_available()
    if not models:
        return
    figr, axr = plt.subplots(figsize=(5.6, 5))
    figp, axp = plt.subplots(figsize=(5.6, 5))
    for m in models:
        d = pd.read_csv(PRED_DIR / f"{m}_val_probs.csv")
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        axr.plot(fpr, tpr, lw=1.3, label=f"{_nice(m)} ({auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(d["y_true"], d["y_prob"])
        axp.plot(rec, prec, lw=1.3, label=_nice(m))
    axr.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8)
    axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
    axr.set_title("ROC curves (validation)"); axr.legend(fontsize=8, loc="lower right")
    axr.spines[["top", "right"]].set_visible(False)
    figr.tight_layout(); figr.savefig(PLOTS_DIR / "roc_curves.png", dpi=160); plt.close(figr)
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    axp.set_title("Precision recall curves (validation)"); axp.legend(fontsize=8, loc="lower left")
    axp.spines[["top", "right"]].set_visible(False)
    figp.tight_layout(); figp.savefig(PLOTS_DIR / "pr_curves.png", dpi=160); plt.close(figp)
    print("  saved roc_curves.png and pr_curves.png")


def plot_confusion_matrices():
    models = _models_available()
    n = len(models)
    if n == 0:
        return
    cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        d = pd.read_csv(PRED_DIR / f"{m}_val_probs.csv")
        cm = confusion_matrix(d["y_true"], d["y_pred_tuned"])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(_nice(m), fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["non", "suicide"], fontsize=8)
        ax.set_yticklabels(["non", "suicide"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices at the tuned threshold (validation)", y=1.02)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight"); plt.close()
    print("  saved confusion_matrices.png")


def main():
    print("Generating plots...")
    plot_f2_comparison()
    plot_threshold_curves()
    plot_roc_pr()
    plot_confusion_matrices()
    print(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
