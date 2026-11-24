"""PHASE 1 - Data acquisition & cleaning. Produces a leak-free train/test split.

Raw (unscaled) features are saved here; scaling / SMOTE happen in phase 2 and
are fitted on the TRAIN split only, otherwise test information leaks in.
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import BENIGN, ID_COLS, family


def load_raw(raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()  # headers have leading spaces
        frames.append(df)
        print(f"loaded {os.path.basename(f)}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    # Non-numeric feature columns: few distinct values -> categorical (one-hot
    # later); many distinct values -> junk-typed numeric, coerce.
    for c in df.columns.drop("Label"):
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 20:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Exact duplicate flows in both train and test inflate scores: drop first.
    df = df.drop_duplicates()
    print(f"cleaning: {n0:,} -> {len(df):,} rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = clean(load_raw(a.raw))
    y = df["Label"].map(family)
    X = df.drop(columns=["Label"])
    num = X.select_dtypes(include="number").columns
    X[num] = X[num].astype("float32")

    const = [c for c in num if X[c].nunique() <= 1]
    X = X.drop(columns=const)
    print(f"dropped {len(const)} constant columns; {X.shape[1]} features left")
    print("class distribution:\n", y.value_counts())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=a.test_size, stratify=y, random_state=a.seed)
    os.makedirs(a.out, exist_ok=True)
    joblib.dump((X_tr, X_te, y_tr, y_te), os.path.join(a.out, "split.joblib"))
    print(f"saved split to {a.out} (benign share train: {(y_tr == BENIGN).mean():.1%})")


if __name__ == "__main__":
    main()
