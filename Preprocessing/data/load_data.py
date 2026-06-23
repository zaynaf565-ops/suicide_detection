"""
Load the raw corpus, filter it and build the four way stratified split.

The split order matters. The test set is removed first so that nothing downstream
can ever touch it. The calibration set is removed next and kept aside only for
conformal prediction. Whatever is left becomes train and validation. Every split
keeps the same class balance through stratification.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_CSV, SPLITS_DIR, TEXT_COL, LABEL_COL, RAW_CLASS_COL, POSITIVE_LABEL,
    RANDOM_SEED, TEST_SIZE, CAL_SIZE, VAL_FRACTION, MIN_WORDS,
)


def load_and_filter() -> pd.DataFrame:
    """Read the raw CSV and apply the three filtering rules."""
    df = pd.read_csv(RAW_CSV, encoding="latin-1")

    # Drop the Pushshift index column if present
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")

    n_raw = len(df)

    # 1. Null or whitespace only text
    df = df[df[TEXT_COL].notna()]
    df = df[df[TEXT_COL].astype(str).str.strip().str.len() > 0]
    n_after_null = len(df)

    # 2. Posts shorter than MIN_WORDS
    word_counts = df[TEXT_COL].astype(str).str.split().str.len()
    df = df[word_counts >= MIN_WORDS]
    n_after_short = len(df)

    # 3. Exact duplicate posts
    df = df.drop_duplicates(subset=[TEXT_COL])
    n_after_dup = len(df)

    # Map the raw class column to a 0/1 label
    df[LABEL_COL] = (df[RAW_CLASS_COL].astype(str).str.strip() == POSITIVE_LABEL).astype(int)
    df = df[[TEXT_COL, LABEL_COL]].reset_index(drop=True)

    print("Filtering summary")
    print(f"  raw posts            {n_raw:,}")
    print(f"  after null removal   {n_after_null:,}  (removed {n_raw - n_after_null:,})")
    print(f"  after short removal  {n_after_short:,}  (removed {n_after_null - n_after_short:,})")
    print(f"  after dup removal    {n_after_dup:,}  (removed {n_after_short - n_after_dup:,})")
    print(f"  class balance        {df[LABEL_COL].value_counts().to_dict()}")
    return df


def make_splits(df: pd.DataFrame, save: bool = True) -> dict:
    """Produce the four way stratified split and optionally save to CSV."""
    y = df[LABEL_COL]

    # Step 1: test set out first
    dev, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Step 2: calibration out of the remaining pool
    cal_fraction_of_dev = CAL_SIZE / (1.0 - TEST_SIZE)
    dev2, cal = train_test_split(
        dev, test_size=cal_fraction_of_dev, stratify=dev[LABEL_COL], random_state=RANDOM_SEED
    )

    # Step 3: train and validation from what is left
    train, val = train_test_split(
        dev2, test_size=VAL_FRACTION, stratify=dev2[LABEL_COL], random_state=RANDOM_SEED
    )

    splits = {"train": train, "val": val, "calibration": cal, "test": test}

    print("\nSplit summary")
    total = len(df)
    for name, part in splits.items():
        pct = 100 * len(part) / total
        pos = int(part[LABEL_COL].sum())
        print(f"  {name:<12} {len(part):>7,}  ({pct:4.1f}%)  suicide={pos:,}")

    # Safety check: test must not overlap with anything
    assert set(test.index).isdisjoint(set(dev.index)), "test overlaps dev pool"
    assert set(cal.index).isdisjoint(set(val.index)), "calibration overlaps val"

    if save:
        for name, part in splits.items():
            out = SPLITS_DIR / f"{name}.csv"
            part.reset_index(drop=True).to_csv(out, index=False)
            print(f"  saved {out}")
    return splits


def load_split(name: str) -> pd.DataFrame:
    """Load a previously saved split by name."""
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run make_splits first.")
    return pd.read_csv(path)


if __name__ == "__main__":
    frame = load_and_filter()
    make_splits(frame, save=True)
