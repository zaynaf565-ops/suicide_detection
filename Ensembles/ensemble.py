"""
Ensemble methods over the five classical models.

Two ensembles are built, both following the approach in the original pipeline.

  Weighted voting. A weighted average of the model probabilities, where the
  weights are found by Nelder-Mead search to maximise validation F2. Weights are
  squared during the search so they stay non negative, and several random
  restarts are used to avoid a poor local optimum. A uniform average is also
  reported as a simple reference.

  Stacking. A logistic regression meta learner trained on the model
  probabilities, with its regularisation strength chosen by cross validation.
  The meta learner is trained on the validation probabilities, which is known to
  carry a risk of overfitting because that same set was used for threshold
  tuning. This is kept deliberately and examined honestly in the results, since
  it is an informative negative finding rather than something to hide.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, fbeta_score

from config import RANDOM_SEED, FBETA, BASE_MODELS, MODEL_LABELS, PRED_IN_DIR
from metrics import compute_all_metrics, tune_threshold, print_metrics

_COARSE = np.arange(0.05, 1.00, 0.05)

# The classical stage saved calibration files as "..._cal_probs.csv", so map the
# split name to the suffix actually used in the filenames.
_SPLIT_SUFFIX = {"val": "val", "calibration": "cal"}


# ---------------------------------------------------------------------------
# Load the saved per model probabilities into matrices
# ---------------------------------------------------------------------------
def load_prob_matrix(split: str):
    """Return (prob_matrix [N, M], y_true [N]) for a split, columns ordered by BASE_MODELS."""
    suffix = _SPLIT_SUFFIX.get(split, split)
    cols, y = [], None
    for name in BASE_MODELS:
        f = PRED_IN_DIR / f"{name}_{suffix}_probs.csv"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}. Run the classical models stage first.")
        d = pd.read_csv(f)
        cols.append(d["y_prob"].values)
        if y is None:
            y = d["y_true"].values
    return np.column_stack(cols), y


# ---------------------------------------------------------------------------
# Weighted voting
# ---------------------------------------------------------------------------
def weighted_average(prob_matrix, weights):
    w = np.clip(np.asarray(weights, float), 0, None)
    return prob_matrix.mean(axis=1) if w.sum() == 0 else (prob_matrix @ w) / w.sum()


def _neg_f2(raw_w, P, y):
    y_prob = weighted_average(P, raw_w ** 2)
    _, f2, _ = tune_threshold(y, y_prob, metric="f2", thresholds=_COARSE)
    return -f2


def optimise_weights(P, y, n_restarts=10):
    M = P.shape[1]
    rng = np.random.default_rng(RANDOM_SEED)
    best = {"f2": -np.inf, "w": None}
    for i in range(n_restarts):
        x0 = np.ones(M) / M if i == 0 else rng.dirichlet(np.ones(M))
        res = minimize(_neg_f2, x0, args=(P, y), method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
        w = res.x ** 2
        w = w / w.sum() if w.sum() > 0 else np.ones(M) / M
        if -res.fun > best["f2"]:
            best = {"f2": -res.fun, "w": w}
    print(f"  Weighted voting: best val F2 (coarse) = {best['f2']:.4f}")
    return best["w"]


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------
def train_stacking(P, y):
    scaler = StandardScaler()
    Xm = scaler.fit_transform(P)
    f2s = make_scorer(fbeta_score, beta=FBETA, zero_division=0)
    best_c, best_cv = 1.0, -np.inf
    for c in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        meta = LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
        cv = cross_val_score(meta, Xm, y, cv=5, scoring=f2s).mean()
        if cv > best_cv:
            best_cv, best_c = cv, c
    print(f"  Stacking: selected C = {best_c}  (CV F2 = {best_cv:.4f})")
    meta = LogisticRegression(C=best_c, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
    meta.fit(Xm, y)
    coef = pd.DataFrame({
        "model": [MODEL_LABELS[m] for m in BASE_MODELS],
        "coefficient": meta.coef_[0],
    }).sort_values("coefficient", ascending=False)
    return meta, scaler, best_c, float(best_cv), coef