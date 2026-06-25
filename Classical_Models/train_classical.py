"""
Phase 1, step 2: train the five classical models.

This script loads the artifacts saved by the preprocessing phase (TF-IDF
matrices and handcrafted feature matrices) and joins them into the model inputs,
rather than recomputing preprocessing. It then trains each model with a
randomised hyperparameter search scored on F2, tunes the decision threshold on
the validation set, and saves the model, its probabilities and a results table.

The probabilities for the validation and calibration splits are saved because
the ensemble and conformal prediction stages later in the project need them.

Run from the project root:
    python Classical_Models/scripts/run_training.py
or directly:
    python train_classical.py   (from inside Classical_Models/)
"""
import sys
import json
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz, hstack, csr_matrix

# Progress bar. Falls back to a no-op wrapper if tqdm is not installed.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier

from config import (
    TFIDF_DIR, FEAT_DIR, PRE_STATS, MODELS_DIR, RESULTS_DIR, PRED_DIR,
    RANDOM_SEED, FBETA, N_ITER_SEARCH, CV_FOLDS,
)
from metrics import (
    compute_all_metrics, tune_threshold, print_metrics, results_to_dataframe,
)


# ---------------------------------------------------------------------------
# Load preprocessing artifacts
# ---------------------------------------------------------------------------
def _load_split(name: str):
    tfidf = load_npz(TFIDF_DIR / f"{name}_tfidf.npz")
    full  = np.load(FEAT_DIR / f"{name}_features_full.npy")
    mnb   = np.load(FEAT_DIR / f"{name}_features_mnb.npy")
    y     = np.load(FEAT_DIR / f"{name}_labels.npy")
    X_full = hstack([tfidf, csr_matrix(full)]).tocsr()
    X_mnb  = hstack([tfidf, csr_matrix(mnb)]).tocsr()
    return X_full, X_mnb, y


def load_artifacts() -> dict:
    missing = [p for p in [TFIDF_DIR, FEAT_DIR] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Preprocessing outputs not found at {missing}.\n"
            "Run the preprocessing phase first, or fix PRE_OUTPUTS in config.py."
        )
    print("Loading preprocessing artifacts...")
    data = {}
    for name in ("train", "val", "calibration"):
        Xf, Xm, y = _load_split(name)
        data[f"X_{name}_full"] = Xf
        data[f"X_{name}_mnb"]  = Xm
        data[f"y_{name}"]      = y
        print(f"  {name:<12} full {Xf.shape}  mnb {Xm.shape}")

    slang = None
    if PRE_STATS.exists():
        slang = json.load(open(PRE_STATS)).get("slang_normalisation")
    data["slang_normalisation"] = slang
    print(f"  slang normalisation in these features: {slang}")
    return data


# ---------------------------------------------------------------------------
# Model definitions (kept the same as the original pipeline, which matched the
# search spaces stated in the proposal)
# ---------------------------------------------------------------------------
def get_model_configs():
    return [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_SEED, solver="lbfgs", penalty="l2", n_jobs=1),
         {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}, "full"),

        ("Multinomial Naive Bayes",
         MultinomialNB(),
         {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}, "mnb"),

        ("Linear SVM",
         CalibratedClassifierCV(
             LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
             cv=3, method="sigmoid"),
         {"estimator__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}, "full"),

        ("Random Forest",
         RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
         {"n_estimators": [50, 100, 200], "max_depth": [20, 40, None],
          "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]}, "full"),

        ("XGBoost",
         XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
                       verbosity=0, tree_method="hist"),
         {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
          "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.7, 0.8, 1.0],
          "colsample_bytree": [0.7, 0.8, 1.0]}, "full"),
    ]


# ---------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(data: dict) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    f2_scorer = make_scorer(fbeta_score, beta=FBETA, zero_division=0)
    results = {}
    configs = get_model_configs()
    n_models = len(configs)
    overall_t0 = time.time()

    for i, (name, estimator, param_dist, key) in enumerate(
            tqdm(configs, desc="Overall progress", unit="model"), start=1):
        elapsed = time.time() - overall_t0
        print(f"\n{'='*60}\n  [{i}/{n_models}] Training: {name}"
              f"   (elapsed so far {elapsed/60:.1f} min)\n{'='*60}")
        print(f"  Running {N_ITER_SEARCH} search iterations x {CV_FOLDS} folds "
              f"= {N_ITER_SEARCH*CV_FOLDS} fits. This is the slow part.")
        X_train, X_val, X_cal = data[f"X_train_{key}"], data[f"X_val_{key}"], data[f"X_calibration_{key}"]
        y_train, y_val, y_cal = data["y_train"], data["y_val"], data["y_calibration"]

        t0 = time.time()
        search = RandomizedSearchCV(
            estimator, param_dist, n_iter=N_ITER_SEARCH, scoring=f2_scorer,
            cv=cv, random_state=RANDOM_SEED, n_jobs=1, refit=True, verbose=1,
        )
        search.fit(X_train, y_train)
        train_time = time.time() - t0
        model = search.best_estimator_
        print(f"  Done in {train_time/60:.1f} min")
        print(f"  Best params:   {search.best_params_}")
        print(f"  CV F2 (train): {search.best_score_:.4f}")

        # Threshold tuning on validation
        y_prob_val = model.predict_proba(X_val)[:, 1]
        best_t, best_f2, thresh_df = tune_threshold(y_val, y_prob_val, metric="f2")
        print(f"  Optimal threshold: {best_t:.2f}  (Val F2: {best_f2:.4f})")

        y_pred_val = (y_prob_val >= best_t).astype(int)
        m = compute_all_metrics(y_val, y_pred_val, y_prob_val, best_t)
        m["train_time_s"] = round(train_time, 1)
        m["cv_f2"]        = round(float(search.best_score_), 4)
        m["best_params"]  = str(search.best_params_)
        print_metrics(m, name)
        results[name] = m

        # Save model, threshold curve, and probabilities for later stages
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe}.joblib")
        thresh_df.to_csv(PRED_DIR / f"{safe}_threshold_curve.csv", index=False)
        pd.DataFrame({"y_true": y_val, "y_prob": y_prob_val, "y_pred_tuned": y_pred_val}) \
            .to_csv(PRED_DIR / f"{safe}_val_probs.csv", index=False)
        y_prob_cal = model.predict_proba(X_cal)[:, 1]
        pd.DataFrame({"y_true": y_cal, "y_prob": y_prob_cal}) \
            .to_csv(PRED_DIR / f"{safe}_cal_probs.csv", index=False)
        print(f"  Saved model and probabilities.")

    return results


def save_results_table(results: dict, slang):
    df = results_to_dataframe(results)
    df.insert(1, "slang_normalisation", slang)
    cols = ["model", "slang_normalisation", "f2", "f1", "precision", "recall",
            "roc_auc", "pr_auc", "accuracy", "threshold", "cv_f2", "train_time_s"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 80)
    print("CLASSICAL ML RESULTS (validation set, sorted by F2)")
    print("=" * 80)
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 80)
    tag = "slang_on" if slang else "slang_off"
    out = RESULTS_DIR / f"classical_ml_val_results_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


def main():
    data = load_artifacts()
    results = train_and_evaluate(data)
    save_results_table(results, data.get("slang_normalisation"))
    print("\nClassical model training complete.")


if __name__ == "__main__":
    main()
