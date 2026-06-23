"""
Phase 1, step 1: run the complete preprocessing end to end.

What this does, in order:
  1. Load the raw CSV, filter it, build the four way split and save the splits.
  2. Clean the text for every split (with slang normalisation if enabled).
  3. Fit TF-IDF on train only and transform every split.
  4. Build the nineteen handcrafted features for every split, full and MNB safe.
  5. Save everything to the outputs folder and write a stats file.

Run from the project root:
    python scripts/run_preprocessing.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.sparse import save_npz

from config import OUTPUTS_DIR, SPLITS_DIR, ENABLE_SLANG_NORMALISATION, TEXT_COL, LABEL_COL
from data.load_data import load_and_filter, make_splits, load_split
from preprocessing.text_cleaning import clean_series, build_tfidf
from preprocessing.feature_engineering import build_feature_matrix

CLEAN_DIR = OUTPUTS_DIR / "clean"
TFIDF_DIR = OUTPUTS_DIR / "tfidf"
FEAT_DIR  = OUTPUTS_DIR / "features"
for d in (CLEAN_DIR, TFIDF_DIR, FEAT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPLIT_NAMES = ["train", "val", "calibration", "test"]


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Slang normalisation enabled: {ENABLE_SLANG_NORMALISATION}")
    print("=" * 60)

    # 1. Build splits if they do not exist yet
    if not (SPLITS_DIR / "train.csv").exists():
        frame = load_and_filter()
        make_splits(frame, save=True)
    else:
        print("Splits already exist, loading them.")

    splits = {name: load_split(name) for name in SPLIT_NAMES}

    # 2. Clean text for each split
    print("\nCleaning text...")
    clean_texts = {}
    for name in SPLIT_NAMES:
        ts = time.time()
        cleaned = clean_series(splits[name][TEXT_COL])
        clean_texts[name] = cleaned
        out = splits[name].copy()
        out["clean_text"] = cleaned
        out[[ "clean_text", LABEL_COL]].to_csv(CLEAN_DIR / f"{name}_clean.csv", index=False)
        print(f"  {name:<12} cleaned in {time.time()-ts:5.1f}s")

    # 3. TF-IDF, fit on train only
    print("\nBuilding TF-IDF...")
    tfidf = build_tfidf(
        clean_texts["train"],
        other_splits={n: clean_texts[n] for n in SPLIT_NAMES if n != "train"},
        save=True,
    )
    for name in SPLIT_NAMES:
        save_npz(TFIDF_DIR / f"{name}_tfidf.npz", tfidf[name])

    # 4. Handcrafted features, full and MNB safe
    print("\nBuilding handcrafted features...")
    for name in SPLIT_NAMES:
        ts = time.time()
        full = build_feature_matrix(splits[name][TEXT_COL], for_mnb=False)
        mnb  = build_feature_matrix(splits[name][TEXT_COL], for_mnb=True)
        np.save(FEAT_DIR / f"{name}_features_full.npy", full)
        np.save(FEAT_DIR / f"{name}_features_mnb.npy", mnb)
        np.save(FEAT_DIR / f"{name}_labels.npy", splits[name][LABEL_COL].values)
        print(f"  {name:<12} features in {time.time()-ts:5.1f}s")

    # 5. Stats file
    stats = {
        "slang_normalisation": ENABLE_SLANG_NORMALISATION,
        "tfidf_vocab_size": int(len(tfidf["vectorizer"].vocabulary_)),
        "splits": {n: {"rows": int(len(splits[n])),
                       "suicide": int(splits[n][LABEL_COL].sum())} for n in SPLIT_NAMES},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(OUTPUTS_DIR / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Preprocessing complete")
    print(json.dumps(stats, indent=2))
    print(f"All outputs saved under {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
