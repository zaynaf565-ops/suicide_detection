import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, classification_report,
)
from config import FBETA


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true     : true binary labels (0/1)
    y_pred     : predicted binary labels at the given threshold
    y_prob     : predicted probabilities for the positive class (optional)
    threshold  : decision threshold used to produce y_pred (logged only)

    Returns
    -------
    dict of metric name -> value
    """
    metrics = {
        "threshold":  threshold,
        "f2":         fbeta_score(y_true, y_pred, beta=FBETA, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "accuracy":   accuracy_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Find the decision threshold that maximises a given metric on a
    validation set.

    Parameters
    ----------
    y_true     : true binary labels
    y_prob     : predicted probabilities for the positive class
    metric     : metric to optimise ('f2', 'f1', or 'recall')
    thresholds : candidate thresholds to evaluate (default: 0.01 to 0.99)

    Returns
    -------
    (best_threshold, best_score, scores_df)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "threshold": t,
            "f2":        fbeta_score(y_true, y_pred, beta=FBETA, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
        })

    df = pd.DataFrame(records)
    best_idx   = df[metric].idxmax()
    best_row   = df.loc[best_idx]
    return float(best_row["threshold"]), float(best_row[metric]), df


def print_metrics(metrics: dict, model_name: str = ""):
    """Pretty-print a metrics dictionary."""
    header = f"  {model_name}" if model_name else ""
    print(f"\n{'='*55}{header}")
    print(f"  {'Metric':<18} {'Value':>10}")
    print(f"  {'-'*30}")
    display_order = [
        "f2", "f1", "precision", "recall", "accuracy",
        "roc_auc", "pr_auc", "threshold",
        "tp", "fp", "tn", "fn",
    ]
    for key in display_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<18} {val:>10.4f}")
            else:
                print(f"  {key:<18} {val:>10}")
    print(f"{'='*55}\n")


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert a dict of {model_name: metrics_dict} to a tidy DataFrame
    sorted by F2 descending.
    """
    rows = []
    for name, m in results.items():
        row = {"model": name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f2", ascending=False).reset_index(drop=True)
    return df
