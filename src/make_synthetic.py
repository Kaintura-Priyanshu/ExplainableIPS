import argparse
import os

import numpy as np
import pandas as pd

COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Fwd Packet Length Mean",
        "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "SYN Flag Count", "ACK Flag Count",
        "Packet Length Variance", "Init_Win_bytes_forward"]


def make(n, label, rng, ip_pool):
    d = {}
    if label == "BENIGN":
        d["Destination Port"] = rng.choice([80, 443, 53, 22, 8080], n)
        d["Flow Duration"] = rng.exponential(2e6, n)
        d["Total Fwd Packets"] = rng.poisson(12, n)
        d["Total Backward Packets"] = rng.poisson(10, n)
        d["Fwd Packet Length Mean"] = rng.normal(300, 120, n).clip(0)
        d["Bwd Packet Length Mean"] = rng.normal(500, 200, n).clip(0)
        d["Flow IAT Mean"] = rng.exponential(1e5, n)
        d["SYN Flag Count"] = rng.binomial(1, .3, n)
        d["ACK Flag Count"] = rng.binomial(8, .6, n)
        d["Packet Length Variance"] = rng.exponential(5e4, n)
        d["Init_Win_bytes_forward"] = rng.choice([8192, 29200, 65535], n)
    elif label == "DoS":
        d["Destination Port"] = np.full(n, 80)
        d["Flow Duration"] = rng.exponential(2e4, n)
        d["Total Fwd Packets"] = rng.poisson(150, n)
        d["Total Backward Packets"] = rng.poisson(1, n)
        d["Fwd Packet Length Mean"] = rng.normal(40, 10, n).clip(0)
        d["Bwd Packet Length Mean"] = rng.normal(5, 3, n).clip(0)
        d["Flow IAT Mean"] = rng.exponential(500, n)
        d["SYN Flag Count"] = rng.binomial(3, .8, n)
        d["ACK Flag Count"] = rng.binomial(1, .2, n)
        d["Packet Length Variance"] = rng.exponential(200, n)
        d["Init_Win_bytes_forward"] = rng.choice([256, 512], n)
    elif label == "BruteForce":
        d["Destination Port"] = rng.choice([21, 22], n)
        d["Flow Duration"] = rng.exponential(5e5, n)
        d["Total Fwd Packets"] = rng.poisson(6, n)
        d["Total Backward Packets"] = rng.poisson(5, n)
        d["Fwd Packet Length Mean"] = rng.normal(60, 10, n).clip(0)
        d["Bwd Packet Length Mean"] = rng.normal(70, 15, n).clip(0)
        d["Flow IAT Mean"] = rng.exponential(3e4, n)
        d["SYN Flag Count"] = rng.binomial(1, .5, n)
        d["ACK Flag Count"] = rng.binomial(5, .7, n)
        d["Packet Length Variance"] = rng.exponential(800, n)
        d["Init_Win_bytes_forward"] = rng.choice([229, 256], n)
    else:  # PortScan
        d["Destination Port"] = rng.randint(1, 65535, n)
        d["Flow Duration"] = rng.exponential(300, n)
        d["Total Fwd Packets"] = rng.poisson(1.2, n) + 1
        d["Total Backward Packets"] = rng.binomial(1, .3, n)
        d["Fwd Packet Length Mean"] = rng.normal(0, 1, n).clip(0)
        d["Bwd Packet Length Mean"] = np.zeros(n)
        d["Flow IAT Mean"] = rng.exponential(100, n)
        d["SYN Flag Count"] = np.ones(n)
        d["ACK Flag Count"] = np.zeros(n)
        d["Packet Length Variance"] = np.zeros(n)
        d["Init_Win_bytes_forward"] = rng.choice([1024, 2048], n)
    df = pd.DataFrame(d)
    dur = df["Flow Duration"].replace(0, np.nan) / 1e6
    df["Flow Bytes/s"] = (df["Fwd Packet Length Mean"] * df["Total Fwd Packets"]) / dur
    df["Flow Packets/s"] = (df["Total Fwd Packets"] + df["Total Backward Packets"]) / dur
    df.insert(0, "Source IP", rng.choice(ip_pool, n))
    # categorical column to exercise the one-hot path of the pipeline
    df["Protocol"] = rng.choice(["TCP", "UDP"], n, p={"BENIGN": [.8, .2]}.get(label, [.97, .03]))
    df["Label"] = {"BENIGN": "BENIGN", "DoS": "DoS Hulk", "PortScan": "PortScan",
                   "BruteForce": "FTP-Patator"}[label]
    return df[["Source IP"] + COLS + ["Protocol", "Label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic.csv")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    rng = np.random.RandomState(7)
    benign_ips = [f"192.168.10.{i}" for i in range(10, 60)]
    dos_ips = ["172.16.0.1", "172.16.0.2"]
    scan_ips = ["172.16.0.9"]
    df = pd.concat([make(a.n, "BENIGN", rng, benign_ips),
                    make(a.n // 5, "DoS", rng, dos_ips),
                    make(a.n // 5, "PortScan", rng, scan_ips),
                    make(a.n // 40, "BruteForce", rng, ["172.16.0.7"])]).sample(frac=1, random_state=1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} synthetic flows -> {a.out}")


if __name__ == "__main__":
    main()
