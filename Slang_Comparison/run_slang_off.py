"""
Controlled comparison of slang normalisation, on versus off.

This trains the five models twice on identical train and validation splits, once
with slang normalisation on and once with it off, using the same fixed
hyperparameters on both sides so that the only difference is the slang switch.
It then reports the validation F2 for each model under both settings and the
gain from the improvement.

Using fixed hyperparameters, rather than a fresh search on each side, keeps the
comparison fair and keeps the run time reasonable. The hyperparameters are the
best settings found earlier by the full search.

Run from inside the folder:
    python run_slang_off.py
Set RAW_CSV below if your dataset sits elsewhere.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

import pipeline as pp
from metrics import compute_all_metrics, tune_threshold

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW_CSV = REPO / "Pre_Processing" / "data" / "raw" / "Suicide_Detection.csv"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
SEED = 42

# Best settings found by the earlier full search.
def make_models():
    return [
        ("Logistic Regression",
         LogisticRegression(C=10, max_iter=1000, class_weight="balanced", random_state=SEED), "full"),
        ("Multinomial Naive Bayes",
         MultinomialNB(alpha=0.001), "mnb"),
        ("Linear SVM",
         CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED), cv=3), "full"),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10,
                                max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1), "full"),
        ("XGBoost",
         XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.7,
                       colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                       random_state=SEED, n_jobs=-1, verbosity=0), "full"),
    ]


def build_branch(train, val, use_slang):
    tag = "on" if use_slang else "off"
    print(f"\n  Building features, slang {tag} ...")
    ctr = [pp.clean_text(t, use_slang) for t in tqdm(train["text"], desc=f"  clean train ({tag})")]
    cva = [pp.clean_text(t, use_slang) for t in tqdm(val["text"], desc=f"  clean val   ({tag})")]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=3, strip_accents="unicode")
    Xtr_tfidf = vec.fit_transform(ctr); Xva_tfidf = vec.transform(cva)
    ftr = pp.features_matrix(train["text"], use_slang, for_mnb=False)
    fva = pp.features_matrix(val["text"],   use_slang, for_mnb=False)
    ftr_m = pp.features_matrix(train["text"], use_slang, for_mnb=True)
    fva_m = pp.features_matrix(val["text"],   use_slang, for_mnb=True)
    return {
        "full": (hstack([Xtr_tfidf, csr_matrix(ftr)]).tocsr(), hstack([Xva_tfidf, csr_matrix(fva)]).tocsr()),
        "mnb":  (hstack([Xtr_tfidf, csr_matrix(ftr_m)]).tocsr(), hstack([Xva_tfidf, csr_matrix(fva_m)]).tocsr()),
    }


def train_eval(branch, y_train, y_val):
    rows = {}
    for name, model, key in make_models():
        Xtr, Xva = branch[key]
        model.fit(Xtr, y_train)
        p = model.predict_proba(Xva)[:, 1]
        t, f2, _ = tune_threshold(y_val, p, metric="f2")
        m = compute_all_metrics(y_val, (p >= t).astype(int), p, t)
        rows[name] = m["f2"]
        print(f"    {name:<26} val F2 = {m['f2']:.4f}  (threshold {t:.2f})")
    return rows


def main():
    t0 = time.time()
    print("Loading and splitting (identical split for both branches)...")
    train, val = pp.load_filter_split(RAW_CSV)
    y_train, y_val = train["label"].values, val["label"].values
    print(f"  train {len(train):,}, val {len(val):,}")

    print("\n=== Branch 1: slang OFF ===")
    off = build_branch(train, val, use_slang=False)
    f2_off = train_eval(off, y_train, y_val)

    print("\n=== Branch 2: slang ON ===")
    on = build_branch(train, val, use_slang=True)
    f2_on = train_eval(on, y_train, y_val)

    # Comparison table
    models = list(f2_on.keys())
    df = pd.DataFrame({
        "model": models,
        "f2_slang_off": [round(f2_off[m], 4) for m in models],
        "f2_slang_on":  [round(f2_on[m], 4) for m in models],
    })
    df["gain"] = (df["f2_slang_on"] - df["f2_slang_off"]).round(4)
    df = df.sort_values("f2_slang_on", ascending=False)
    df.to_csv(RESULTS / "slang_comparison.csv", index=False)
    print("\n" + "=" * 70)
    print("SLANG NORMALISATION COMPARISON (validation F2)")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n  mean gain across models: {df['gain'].mean():.4f}")

    # Grouped bar chart
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, df["f2_slang_off"], w, label="Slang off", color="#bbbbbb", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, df["f2_slang_on"],  w, label="Slang on",  color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Validation F2"); ax.set_ylim(0.85, 0.96)
    ax.set_title("Effect of slang normalisation on F2")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(RESULTS / "slang_comparison.png", dpi=160); plt.close()
    print(f"  saved chart and table under {RESULTS}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
