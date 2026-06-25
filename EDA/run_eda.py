"""
Exploratory data analysis figures.

Three figures that describe the data and the features, separate from any model.

  Word clouds. The most frequent cleaned words in suicide posts and in non
  suicide posts, side by side, to show the surface vocabulary difference.

  TF-IDF top token heatmap. The tokens with the largest difference in mean
  TF-IDF weight between the two classes, which shows which words carry the most
  class signal in the bag of words representation.

  Feature correlation heatmap. The correlation between the nineteen handcrafted
  features, which shows where features overlap and where they add new
  information.

Run:
    python EDA/run_eda.py
Reads the preprocessing outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

plt.rcParams.update({"font.family": "serif", "font.size": 10})

ROOT_DIR  = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
PRE_OUT   = REPO_ROOT / "Pre_Processing" / "outputs"
CLEAN_CSV = PRE_OUT / "clean" / "train_clean.csv"
TFIDF_NPZ = PRE_OUT / "tfidf" / "train_tfidf.npz"
VECTORIZER = PRE_OUT / "tfidf_vectorizer.joblib"
FEAT_NPY  = PRE_OUT / "features" / "train_features_full.npy"
LAB_NPY   = PRE_OUT / "features" / "train_labels.npy"
OUT = ROOT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "word_count", "char_count", "avg_word_length",
    "exclamation_count", "question_count", "exclamation_density", "question_density",
    "ellipsis_count", "uppercase_ratio",
    "first_person_count", "first_person_density",
    "negation_count", "negation_density",
    "hopelessness_count", "hopelessness_density",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]


def word_clouds():
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud not installed, skipping. Install with: pip install wordcloud")
        return
    if not CLEAN_CSV.exists():
        print(f"  {CLEAN_CSV} not found, skipping word clouds")
        return
    df = pd.read_csv(CLEAN_CSV).dropna(subset=["clean_text"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lab, title, cmap in [
        (axes[0], 1, "Suicide posts", "Reds"),
        (axes[1], 0, "Non suicide posts", "Blues"),
    ]:
        text = " ".join(df[df["label"] == lab]["clean_text"].astype(str).values)
        wc = WordCloud(width=800, height=500, background_color="white",
                       colormap=cmap, max_words=120).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(OUT / "wordclouds.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved wordclouds.png")


def tfidf_heatmap(top_k=25):
    if not (TFIDF_NPZ.exists() and VECTORIZER.exists() and LAB_NPY.exists()):
        print("  TF-IDF artifacts not found, skipping heatmap")
        return
    X = load_npz(TFIDF_NPZ)
    y = np.load(LAB_NPY)
    vocab = joblib.load(VECTORIZER).get_feature_names_out()
    mean_pos = np.asarray(X[y == 1].mean(axis=0)).ravel()
    mean_neg = np.asarray(X[y == 0].mean(axis=0)).ravel()
    diff = mean_pos - mean_neg
    top_pos = np.argsort(diff)[-top_k:][::-1]
    top_neg = np.argsort(diff)[:top_k]
    idx = np.concatenate([top_pos, top_neg])
    mat = np.column_stack([mean_pos[idx], mean_neg[idx]])
    labels = [vocab[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 11))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Suicide", "Non suicide"])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.axhline(top_k - 0.5, color="black", lw=1)
    ax.set_title(f"Mean TF-IDF weight, top {top_k} tokens per class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean TF-IDF")
    plt.tight_layout(); plt.savefig(OUT / "tfidf_top_tokens_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved tfidf_top_tokens_heatmap.png")


def feature_correlation():
    if not FEAT_NPY.exists():
        print("  feature matrix not found, skipping correlation heatmap")
        return
    F = np.load(FEAT_NPY)
    df = pd.DataFrame(F, columns=FEATURE_NAMES)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES))); ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURE_NAMES))); ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")
    ax.set_title("Correlation between the nineteen handcrafted features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(OUT / "feature_correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  saved feature_correlation_heatmap.png")


def main():
    print("Generating EDA figures...")
    word_clouds()
    tfidf_heatmap()
    feature_correlation()
    print(f"Saved figures under {OUT}")


if __name__ == "__main__":
    main()
