"""
Generate test set probabilities for each saved classical model.

The training stage saved validation and calibration probabilities but not test
probabilities, since the test set is held back until the final evaluation. The
calibration and conformal stages, and the final evaluation, all need test
probabilities, so this script produces them once.

It loads each saved model and the saved test features, predicts, and writes one
file per model next to the existing probability files. No model is retrained.

Run:
    python Trustworthiness/predict_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz, hstack, csr_matrix

from config import TFIDF_DIR, FEAT_DIR, CM_MODELS_DIR, CM_PRED_DIR, BASE_MODELS


def _load_test_inputs():
    tfidf = load_npz(TFIDF_DIR / "test_tfidf.npz")
    full  = np.load(FEAT_DIR / "test_features_full.npy")
    mnb   = np.load(FEAT_DIR / "test_features_mnb.npy")
    y     = np.load(FEAT_DIR / "test_labels.npy")
    X_full = hstack([tfidf, csr_matrix(full)]).tocsr()
    X_mnb  = hstack([tfidf, csr_matrix(mnb)]).tocsr()
    return X_full, X_mnb, y


def main():
    print("Loading test features...")
    X_full, X_mnb, y = _load_test_inputs()
    print(f"  full {X_full.shape}, mnb {X_mnb.shape}")

    for safe, label, key in BASE_MODELS:
        model_path = CM_MODELS_DIR / f"{safe}.joblib"
        if not model_path.exists():
            print(f"  skipping {label}, model file not found at {model_path}")
            continue
        model = joblib.load(model_path)
        X = X_mnb if key == "mnb" else X_full
        y_prob = model.predict_proba(X)[:, 1]
        out = CM_PRED_DIR / f"{safe}_test_probs.csv"
        pd.DataFrame({"y_true": y, "y_prob": y_prob}).to_csv(out, index=False)
        print(f"  {label:<26} test probabilities saved to {out.name}")

    print("\nTest probabilities complete.")


if __name__ == "__main__":
    main()
