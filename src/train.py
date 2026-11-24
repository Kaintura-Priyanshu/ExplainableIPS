"""PHASE 4 - Ensemble training: RandomForest + XGBoost (supervised, known attacks)
and Isolation Forest (unsupervised, unknown attacks).

Why this combination:
  * RF + XGBoost soft-vote: two different tree learners (bagging vs boosting)
    make different mistakes; averaging probabilities is more stable than either.
  * Isolation Forest is trained on BENIGN ONLY. It never sees an attack, so it
    flags "does not look like normal" - the only way to have any chance on an
    attack type absent from the training labels (zero-day).
  * Its threshold is calibrated so that only --if-fpr (default 1%) of held-out
    benign flows get flagged. Without calibration it would alert constantly.

--holdout <Class> removes one attack family from training to MEASURE zero-day
ability honestly: supervised-only recall vs supervised+IsolationForest recall
on a class the models never saw.
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common import BENIGN, anomaly_flags, ensemble_proba
from explain import Explainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/features.joblib")
    ap.add_argument("--models", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--rf-weight", type=float, default=0.5)
    ap.add_argument("--if-fpr", type=float, default=0.01)
    ap.add_argument("--holdout", default=None, help="attack family to hide (zero-day test)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)
    os.makedirs(a.reports, exist_ok=True)

    d = joblib.load(a.data)
    tf, X_res, y_res, X_te, y_te, ben = (d["tf"], d["X_res"], d["y_res"],
                                         d["X_te"], d["y_te"], d["benign_ref"])
    if a.holdout:
        keep = (y_res != a.holdout).to_numpy()
        X_res, y_res = X_res[keep], y_res[keep]
        print(f"[zero-day test] '{a.holdout}' removed from training")

    le = LabelEncoder().fit(y_res)
    classes = list(le.classes_)
    y_enc = le.transform(y_res)
    benign_i = classes.index(BENIGN)

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced", random_state=a.seed).fit(X_res, y_enc)
    xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.9,
                        tree_method="hist", n_jobs=-1, random_state=a.seed,
                        eval_metric="mlogloss").fit(X_res, y_enc)
    print(f"supervised models trained in {time.time() - t0:.1f}s on {len(X_res):,} rows")

    # ---- Isolation Forest on benign only, calibrated on held-out benign ----
    cut = int(len(ben) * 0.8)
    iforest = IsolationForest(n_estimators=200, contamination="auto", n_jobs=-1,
                              random_state=a.seed).fit(ben.iloc[:cut])
    calib = iforest.score_samples(ben.iloc[cut:])
    if_thr = float(np.quantile(calib, a.if_fpr))

    bundle = {"rf": rf, "xgb": xgb, "iforest": iforest, "if_threshold": if_thr,
              "rf_weight": a.rf_weight, "classes": classes, "tf": tf,
              "benign_mean": ben.mean().to_numpy(), "benign_std": ben.std().to_numpy() + 1e-6}

    # ---- evaluation (test set touched once) ----
    y_true = y_te.to_numpy()
    known = np.isin(y_true, classes)
    rf_p, xgb_p = rf.predict_proba(X_te), xgb.predict_proba(X_te)
    ens_p = ensemble_proba(bundle, X_te)
    pred = {n: np.array(classes)[p.argmax(1)] for n, p in
            (("RandomForest", rf_p), ("XGBoost", xgb_p), ("Ensemble", ens_p))}
    print("\n== Macro-F1 on known classes (test set) ==")
    for n, p in pred.items():
        print(f"{n:13s} {f1_score(y_true[known], p[known], average='macro', labels=classes):.4f}")
    print("\n== Ensemble per-class report ==")
    print(classification_report(y_true[known], pred["Ensemble"][known], digits=4, zero_division=0))

    p_attack = 1 - ens_p[:, benign_i]
    is_att = y_true != BENIGN
    print("== IPS confidence thresholds (binary attack vs benign) ==")
    for t in (0.5, 0.7, 0.9):
        fl = p_attack >= t
        tp, fp, fn = (fl & is_att).sum(), (fl & ~is_att).sum(), (~fl & is_att).sum()
        print(f"thr {t}: precision={tp / max(tp + fp, 1):.4f} recall={tp / max(tp + fn, 1):.4f} "
              f"false_positives={fp}")

    _, anom = anomaly_flags(bundle, X_te)
    print(f"\n== Isolation Forest (benign-only, calibrated {a.if_fpr:.0%} FPR) ==")
    print(f"benign flagged: {anom[~is_att].mean():.2%}")
    for c in sorted(set(y_true[is_att])):
        m = y_true == c
        sup = (p_attack[m] >= 0.5).mean()
        comb = ((p_attack[m] >= 0.5) | anom[m]).mean()
        tag = "  <-- UNSEEN in training" if c not in classes else ""
        print(f"{c:12s} supervised={sup:.3f}  IF-alone={anom[m].mean():.3f}  "
              f"supervised+IF={comb:.3f}{tag}")
    fp_comb = ((p_attack >= 0.5) | anom)[~is_att].mean()
    print(f"benign false-alarm rate: supervised={(p_attack[~is_att] >= .5).mean():.3%} "
          f"combined={fp_comb:.3%}  (the price of catching zero-days)")

    # ---- latency: honest numbers instead of assumed ones ----
    rf.n_jobs = 1  # per-flow calls: thread start-up would dominate
    n = min(len(X_te), 3000)
    t = time.perf_counter(); ensemble_proba(bundle, X_te.iloc[:n]); batch_ms = (time.perf_counter() - t) / n * 1e3
    t = time.perf_counter()
    for i in range(100):
        ensemble_proba(bundle, X_te.iloc[i:i + 1])
    single_ms = (time.perf_counter() - t) / 100 * 1e3
    rf.n_jobs = -1
    print(f"\nlatency: {batch_ms:.3f} ms/flow in batches of {n}, {single_ms:.2f} ms/flow one-by-one")

    # ---- explainability: global SHAP ----
    ex = Explainer(xgb, tf["selected"], classes)
    gi = ex.global_importance(X_te.sample(min(len(X_te), 1000), random_state=a.seed))[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in gi][::-1], [v for _, v in gi][::-1])
    ax.set_title("Global TreeSHAP importance (mean |SHAP|)")
    plt.tight_layout(); fig.savefig(os.path.join(a.reports, "shap_global.png"), dpi=130); plt.close(fig)

    cm = confusion_matrix(y_true[known], pred["Ensemble"][known], labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45,
                                                            colorbar=False, values_format="d")
    plt.tight_layout(); fig.savefig(os.path.join(a.reports, "confusion_matrix.png"), dpi=130); plt.close(fig)

    joblib.dump(bundle, os.path.join(a.models, "ids_ensemble.joblib"), compress=3)
    with open(os.path.join(a.reports, "metrics.json"), "w") as f:
        json.dump({"holdout": a.holdout, "if_threshold": if_thr,
                   "ensemble_macro_f1": float(f1_score(y_true[known], pred["Ensemble"][known],
                                                       average="macro", labels=classes)),
                   "latency_ms_batch": batch_ms, "latency_ms_single": single_ms,
                   "shap_top": [(k, float(v)) for k, v in gi]}, f, indent=2)
    print("saved models/ids_ensemble.joblib + reports/")


if __name__ == "__main__":
    main()
