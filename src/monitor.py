import argparse
import ipaddress
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from common import BENIGN, ZERO_DAY, anomaly_flags, apply_transform, ensemble_proba
from explain import Explainer

DEFAULT_ALLOW = {"127.0.0.1", "8.8.8.8"}


def block_ip(ip, enforce):
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return "SKIPPED: not a valid IP"
    cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
    if not enforce:
        return "DRY-RUN: " + " ".join(cmd)
    try:
        subprocess.run(cmd, check=True, timeout=5)
        return "BLOCKED"
    except Exception as e:  # noqa: BLE001
        return f"BLOCK-FAILED: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--model", default="models/ids_ensemble.joblib")
    ap.add_argument("--alert-threshold", type=float, default=0.5)
    ap.add_argument("--block-threshold", type=float, default=0.9)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--block-zero-day", action="store_true")
    ap.add_argument("--allowlist", help="file with one IP per line")
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--explain-max", type=int, default=200, help="SHAP rows per chunk")
    ap.add_argument("--log", default="reports/alerts.jsonl")
    a = ap.parse_args()

    b = joblib.load(a.model)
    tf, classes = b["tf"], b["classes"]
    benign_i = classes.index(BENIGN)
    b["rf"].n_jobs = 1
    explainer = Explainer(b["xgb"], tf["selected"], classes)
    allow = set(DEFAULT_ALLOW)
    if a.allowlist:
        allow |= {l.strip() for l in open(a.allowlist) if l.strip()}
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)

    hits, blocked = defaultdict(int), set()
    n_flows = n_known = n_zd = n_blocks = 0
    t_infer = 0.0

    with open(a.log, "a") as logf:
        for chunk in pd.read_csv(a.flows, encoding="latin-1", chunksize=a.chunk, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            src = (chunk["Source IP"].astype(str).to_numpy() if "Source IP" in chunk
                   else np.array(["unknown"] * len(chunk)))
            t0 = time.perf_counter()
            X = apply_transform(chunk, tf)
            proba = ensemble_proba(b, X)
            _, anom = anomaly_flags(b, X)
            t_infer += time.perf_counter() - t0
            n_flows += len(X)

            p_attack = 1 - proba[:, benign_i]
            attack_proba = proba.copy(); attack_proba[:, benign_i] = -1
            cls_idx = attack_proba.argmax(1)  # most likely ATTACK class
            known = p_attack >= a.alert_threshold
            zd = ~known & anom
            flagged = np.flatnonzero(known | zd)

            # explanations only for flagged rows (SHAP is the costly step)
            expl = {}
            ex_rows = flagged[:a.explain_max]
            kn_rows = [i for i in ex_rows if known[i]]
            if kn_rows:
                for i, e in zip(kn_rows, explainer.top_features(X.iloc[kn_rows], cls_idx[kn_rows])):
                    expl[i] = e
            for i in (i for i in ex_rows if zd[i]):
                z = (X.iloc[i].to_numpy() - b["benign_mean"]) / b["benign_std"]
                top = np.argsort(-np.abs(z))[:3]
                expl[i] = [{"feature": tf["selected"][j], "value": round(float(X.iloc[i, j]), 3),
                            "z_vs_benign": round(float(z[j]), 1)} for j in top]

            for i in flagged:
                ip, p = src[i], float(p_attack[i])
                label = classes[cls_idx[i]] if known[i] else ZERO_DAY
                action = "ALERT"
                blockable = (known[i] and p >= a.block_threshold) or (zd[i] and a.block_zero_day)
                if blockable and ip not in allow and ip != "unknown" and ip not in blocked:
                    hits[ip] += 1
                    if hits[ip] >= a.min_hits:
                        action = block_ip(ip, a.enforce)
                        blocked.add(ip); n_blocks += 1
                n_known += bool(known[i]); n_zd += bool(zd[i])
                rec = {"time": datetime.now().isoformat(timespec="seconds"), "src": ip,
                       "type": label, "p_attack": round(p, 3), "action": action,
                       "why": expl.get(i)}
                logf.write(json.dumps(rec) + "\n")
                if action != "ALERT":
                    print(json.dumps(rec))

    print(f"\nflows={n_flows:,}  known-attack alerts={n_known:,}  zero-day suspects={n_zd:,}  "
          f"blocks={n_blocks} ({'ENFORCED' if a.enforce else 'dry-run'})")
    print(f"inference (transform+RF+XGB+IF): {t_infer / max(n_flows, 1) * 1e3:.3f} ms/flow; log -> {a.log}")


if __name__ == "__main__":
    main()
