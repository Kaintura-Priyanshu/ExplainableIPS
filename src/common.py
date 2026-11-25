import re

import numpy as np
import pandas as pd

# Identifiers: a model that memorises IPs/timestamps learns the *lab setup*,
# not attack behaviour, and fails on any other network.
ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
BENIGN = "BENIGN"
ZERO_DAY = "ZERO_DAY_SUSPECT"


def family(label: str) -> str:
    """Collapse the 15 raw labels into attack families (some raw classes have
    <20 samples, e.g. Heartbleed, which is hard to learn/evaluate alone)."""
    s = re.sub(r"[^A-Za-z0-9 ]", " ", str(label)).lower()
    if "benign" in s:
        return BENIGN
    if "ddos" in s or "dos" in s:
        return "DoS_DDoS"
    if "portscan" in s:
        return "PortScan"
    if "patator" in s or "ftp" in s or "ssh" in s:
        return "BruteForce"
    if "web attack" in s:
        return "WebAttack"
    if "bot" in s:
        return "Bot"
    if "infiltration" in s:
        return "Infiltration"
    if "heartbleed" in s:
        return "Heartbleed"
    return "Other"


def apply_transform(df: pd.DataFrame, tf: dict) -> pd.DataFrame:
    """Raw flow rows -> model-ready matrix (one-hot, MinMax, selected features)."""
    X = df.copy()
    X.columns = X.columns.str.strip()
    X = X.drop(columns=[c for c in ID_COLS + ["Label"] if c in X.columns])
    cat = [c for c in tf["cat_cols"] if c in X.columns]
    if cat:
        X = pd.get_dummies(X, columns=cat, dtype=float)
    # columns never seen in training are dropped; missing ones become 0
    X = X.reindex(columns=tf["all_cols"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    Xs = pd.DataFrame(tf["scaler"].transform(X), columns=tf["all_cols"], index=X.index)
    # live traffic can exceed the training min/max; clip so one huge value
    # cannot dominate distance-based parts (SMOTE/IsolationForest)
    Xs = Xs.clip(0.0, 1.0)
    return Xs[tf["selected"]].astype("float32")


def ensemble_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Soft-voting: weighted average of RF and XGBoost class probabilities."""
    w = bundle["rf_weight"]
    return w * bundle["rf"].predict_proba(X) + (1 - w) * bundle["xgb"].predict_proba(X)


def anomaly_flags(bundle: dict, X: pd.DataFrame):
    """Isolation Forest score (lower = more abnormal) and boolean flag."""
    score = bundle["iforest"].score_samples(X)
    return score, score < bundle["if_threshold"]
