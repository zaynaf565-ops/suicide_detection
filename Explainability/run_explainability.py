"""
Run the explainability stage, with progress bars.

Three parts.
  Global importance. For every model, the features that drive it most. XGBoost
  uses SHAP, Random Forest uses its built in importance (which is fast and avoids
  the very slow SHAP pass on deep trees), linear models use their coefficients,
  and Naive Bayes uses the difference in log probabilities. Each is a top feature
  bar chart.
  Local explanations. LIME explanations for a few example posts.
  Agreement. How far SHAP and LIME agree on the same posts for XGBoost, by
  Jaccard overlap of the top tokens and Spearman rank correlation.

Each slow part shows a percentage progress bar.

Run:
    python Explainability/run_explainability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, hstack, csr_matrix

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x=None, **k):
        return x if x is not None else range(0)

from config import (
    VECTORIZER, TFIDF_DIR, FEAT_DIR, SPLITS_DIR, CM_MODELS, RESULTS_DIR, PLOTS_DIR,
    RANDOM_SEED, TOP_FEATURES, HANDCRAFTED_NAMES, BASE_MODELS,
)
import text_features as tf

# ---------------------------------------------------------------------------
# Speed settings. These are intentionally modest so the stage finishes in a
# reasonable time on a laptop. Raise them if you want finer detail.
# ---------------------------------------------------------------------------
SHAP_SAMPLE      = 100     # rows for the XGBoost SHAP pass
LIME_EXAMPLES    = 3       # number of posts shown as full LIME figures
LIME_NUM_SAMPLES = 150     # perturbations per LIME explanation
AGREEMENT_SAMPLE = 12      # posts used for the SHAP versus LIME agreement
AGREEMENT_TOPK   = 10      # top tokens compared per post

plt.rcParams.update({"font.family": "serif", "font.size": 10})
rng = np.random.default_rng(RANDOM_SEED)


def _feature_names(for_mnb=False):
    vocab = list(joblib.load(VECTORIZER).get_feature_names_out())
    hc = [c for c in HANDCRAFTED_NAMES if not (for_mnb and c in {"vader_neg", "vader_pos", "vader_compound"})]
    return np.array(vocab + hc)


def _load_sample(key, n):
    tfidf = load_npz(TFIDF_DIR / "test_tfidf.npz")
    feat = np.load(FEAT_DIR / f"test_features_{'mnb' if key == 'mnb' else 'full'}.npy")
    idx = rng.choice(tfidf.shape[0], size=min(n, tfidf.shape[0]), replace=False)
    X = hstack([tfidf[idx], csr_matrix(feat[idx])]).tocsr()
    return X


def _save_bar(names, vals, title, path):
    k = len(names)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(names[::-1], vals[::-1], color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.set_title(title); ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------
def global_importance():
    print("\n[1/3] Global feature importance")
    summary = {}
    for safe, label, kind, key in tqdm(BASE_MODELS, desc="  models", unit="model"):
        mpath = CM_MODELS / f"{safe}.joblib"
        if not mpath.exists():
            continue
        model = joblib.load(mpath)
        names = _feature_names(for_mnb=(key == "mnb"))

        if safe == "xgboost":
            import shap
            X = _load_sample(key, SHAP_SAMPLE).toarray().astype(np.float32)
            expl = shap.TreeExplainer(model)
            # batched so the bar moves
            chunks = np.array_split(np.arange(X.shape[0]), 10)
            acc = np.zeros(X.shape[1])
            for c in tqdm(chunks, desc="  XGBoost SHAP", leave=False):
                sv = expl.shap_values(X[c], check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[1]
                elif getattr(sv, "ndim", 2) == 3:
                    sv = sv[:, :, 1]
                acc += np.abs(sv).sum(axis=0).ravel()
            imp = acc / X.shape[0]
        elif kind == "tree":   # Random Forest, use the fast built in importance
            imp = np.asarray(model.feature_importances_).ravel()
        elif kind == "linear":
            if hasattr(model, "coef_"):
                imp = np.abs(np.asarray(model.coef_).ravel())
            else:
                coefs = [clf.estimator.coef_.ravel() for clf in model.calibrated_classifiers_]
                imp = np.abs(np.mean(coefs, axis=0))
        else:  # nb
            imp = np.abs(model.feature_log_prob_[1] - model.feature_log_prob_[0])

        imp = np.asarray(imp).ravel()
        k = min(TOP_FEATURES, len(imp))
        top = np.argsort(imp)[-k:][::-1]
        summary[label] = list(zip(names[top], np.round(imp[top], 4)))
        _save_bar(names[top], imp[top], f"Top {k} features, {label}", PLOTS_DIR / f"importance_{safe}.png")

    rows = [{"model": m, "rank": r, "feature": n, "importance": v}
            for m, feats in summary.items() for r, (n, v) in enumerate(feats, 1)]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "global_importance.csv", index=False)
    print("  global importance done")


# ---------------------------------------------------------------------------
# LIME pipeline and agreement
# ---------------------------------------------------------------------------
class TextPipeline:
    def __init__(self, model, for_mnb=False):
        self.model = model
        self.vec = joblib.load(VECTORIZER)
        self.for_mnb = for_mnb

    def predict_proba(self, texts):
        cleaned = [tf.clean_text(t) for t in texts]
        Xt = self.vec.transform(cleaned)
        F = tf.features_matrix(texts, for_mnb=self.for_mnb)
        X = hstack([Xt, csr_matrix(F)]).tocsr()
        return self.model.predict_proba(X)


def _example_posts(n):
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    pos = df[df["label"] == 1].sample(n // 2 + n % 2, random_state=RANDOM_SEED)
    neg = df[df["label"] == 0].sample(n // 2, random_state=RANDOM_SEED)
    return pd.concat([pos, neg]).reset_index(drop=True)


def lime_and_agreement():
    try:
        import shap
        from lime.lime_text import LimeTextExplainer
        from scipy.stats import spearmanr
    except Exception as e:
        print(f"\n  lime or shap not installed ({e}); skipping LIME and agreement.")
        return

    model = joblib.load(CM_MODELS / "xgboost.joblib")
    pipe = TextPipeline(model, for_mnb=False)
    explainer = LimeTextExplainer(class_names=["non", "suicide"])
    vocab = joblib.load(VECTORIZER).get_feature_names_out()
    tree_expl = shap.TreeExplainer(model)

    posts = _example_posts(max(LIME_EXAMPLES, AGREEMENT_SAMPLE))

    print(f"\n[2/3] LIME example explanations ({LIME_EXAMPLES} posts)")
    for i in tqdm(range(min(LIME_EXAMPLES, len(posts))), desc="  LIME examples"):
        text = str(posts.iloc[i]["text"])
        exp = explainer.explain_instance(text, pipe.predict_proba,
                                         num_features=10, num_samples=LIME_NUM_SAMPLES)
        fig = exp.as_pyplot_figure(); fig.set_size_inches(6, 4)
        plt.tight_layout(); fig.savefig(PLOTS_DIR / f"lime_example_{i+1}.png", dpi=150); plt.close(fig)

    print(f"\n[3/3] SHAP versus LIME agreement ({AGREEMENT_SAMPLE} posts)")
    jacc, spear = [], []
    for i in tqdm(range(min(AGREEMENT_SAMPLE, len(posts))), desc="  agreement"):
        text = str(posts.iloc[i]["text"])
        cleaned = tf.clean_text(text)
        Xt = pipe.vec.transform([cleaned])
        F = tf.features_matrix([text], for_mnb=False)
        X = hstack([Xt, csr_matrix(F)]).tocsr()
        sv = tree_expl.shap_values(X, check_additivity=False)
        sv = (sv[1] if isinstance(sv, list) else sv).ravel()
        present = Xt.nonzero()[1]
        if len(present) < 2:
            continue
        shap_top = sorted({vocab[j]: abs(sv[j]) for j in present}.items(),
                          key=lambda kv: kv[1], reverse=True)[:AGREEMENT_TOPK]
        shap_top = [w for w, _ in shap_top]
        exp = explainer.explain_instance(text, pipe.predict_proba,
                                         num_features=AGREEMENT_TOPK, num_samples=LIME_NUM_SAMPLES)
        lime_top = [w for w, _ in exp.as_list()]
        sset, lset = set(shap_top), set(lime_top)
        if sset or lset:
            jacc.append(len(sset & lset) / len(sset | lset))
        shared = list(sset & lset)
        if len(shared) >= 2:
            sr = {w: r for r, w in enumerate(shap_top)}
            lr = {w: r for r, w in enumerate(lime_top)}
            rho, _ = spearmanr([sr[w] for w in shared], [lr[w] for w in shared])
            if not np.isnan(rho):
                spear.append(rho)

    res = {"n_posts": len(jacc),
           "mean_jaccard": round(float(np.mean(jacc)), 4) if jacc else None,
           "mean_spearman": round(float(np.mean(spear)), 4) if spear else None}
    pd.DataFrame([res]).to_csv(RESULTS_DIR / "shap_lime_agreement.csv", index=False)
    print(f"  agreement over {res['n_posts']} posts: "
          f"mean Jaccard {res['mean_jaccard']}, mean Spearman {res['mean_spearman']}")
    if jacc:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.hist(jacc, bins=10, color="#4c72b0", edgecolor="white")
        ax.set_xlabel("Jaccard overlap of top tokens (SHAP vs LIME)")
        ax.set_ylabel("Posts"); ax.set_title("SHAP and LIME agreement")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(PLOTS_DIR / "shap_lime_agreement.png", dpi=150); plt.close()


def main():
    global_importance()
    lime_and_agreement()
    print(f"\nSaved all explainability outputs under {RESULTS_DIR}")
    print("Explainability stage complete.")


if __name__ == "__main__":
    main()