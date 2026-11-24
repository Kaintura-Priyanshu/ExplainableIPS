# Explainable ML IDS/IPS on CICIDS2017 

Six-phase architecture: ingestion, preprocessing and class rebalancing, feature selection,
an ensemble of supervised and unsupervised models, explainability, and active IPS mitigation.

```
 Data Acquisition        preprocess.py   load CSVs, drop IDs/inf/dupes, stratified split
        |
 Preprocessing & SMOTE   features.py     one-hot + MinMax (train-fit) ... SMOTE (train only)
        |
 Feature Selection       features.py     Pearson filter -> RFE (RandomForest)
        |
 Ensemble Inference      train.py        RandomForest + XGBoost (soft vote)
        |                                + Isolation Forest (benign-only, zero-day)
 Explainable AI          explain.py      TreeSHAP per alert (+ z-score for zero-day)
        |
 Active IPS Mitigation   monitor.py      iptables DROP + JSONL telemetry log
```

## Run

```bash
pip install -r requirements.txt
export PYTHONPATH=src                      # Windows PowerShell: $env:PYTHONPATH="src"

python src/preprocess.py                   # phase 1
python src/features.py                     # phases 2-3  (--k-features 30 default)
python src/train.py                        # phase 4 + SHAP plots + latency
python src/monitor.py --flows data/raw/<one-day-file>.csv     # phases 5-6, dry-run
```

Real data: put the CICIDS2017 CSVs in `data/raw/`
Use the **TrafficLabelling** version if you
want blocking, since it has `Source IP`. No dataset yet? `python src/make_synthetic.py` makes
a tiny fake one for smoke tests only. Its numbers mean nothing.

Honest zero-day test (hide one attack from training, then see whether it is still caught):

```bash
python src/train.py --holdout PortScan     # or DoS_DDoS, BruteForce, Bot ...
```

Safe blocking: `monitor.py` is dry-run unless you pass `--enforce` (Linux + root). Try it in a lab first.

## Why each phase is built this way

| Phase | Decision | Reasoning |
|---|---|---|
| 1 | Drop Flow ID, IPs, ports, timestamp from features | Otherwise the model memorises the lab network rather than attack behaviour. |
| 1 | De-duplicate before splitting | Identical flows in train and test inflate scores. |
| 2 | Scaler and one-hot fitted on **train only** | Fitting on all data leaks test statistics. |
| 2 | **SMOTE only on training rows, after the split** | Synthetic points made before splitting end up as near-copies of test points, which gives fake accuracy. |
| 2 | Undersample benign first (`--max-benign`) | Benign is ~80% and highly redundant, so this saves training time. |
| 3 | Pearson filter (\|r\|>0.95) then RFE | The filter removes twin features cheaply, so RFE spends its budget on useful ones. Fewer features means faster inference. |
| 4 | RF + XGBoost soft vote | Bagging and boosting make different errors, and averaged probabilities are more stable. |
| 4 | Isolation Forest on **real benign only** | It must learn normal traffic and never see SMOTE-made points or attacks. |
| 4 | IF threshold calibrated to a 1% benign false-positive rate | Uncalibrated anomaly detectors alert constantly. |
| 5 | TreeSHAP on XGBoost | Each alert says which features drove it, so analysts can confirm or dismiss it quickly. |
| 6 | Block only at P≥0.9 on ≥3 flows, with an allow-list | A false positive blocks a legitimate host, which is an outage. |
| 6 | Zero-day suspects are **alert-only** by default | Anomaly alerts have a higher false-positive rate, so blocking needs an opt-in (`--block-zero-day`). |

## What the output means

Each line of `reports/alerts.jsonl`:

```json
{"src":"172.16.0.1","type":"DoS_DDoS","p_attack":1.0,"action":"DRY-RUN: iptables -I INPUT -s 172.16.0.1 -j DROP",
 "why":[{"feature":"Total Fwd Packets","value":0.742,"shap":5.43}, ...]}
```

`type` is a known class or `ZERO_DAY_SUSPECT`. `why` holds the SHAP contributions for known
attacks, or the z-score against benign traffic for zero-day suspects.

## Measured (synthetic smoke test, not real results)

* All models scored 1.00 because the fake attacks are trivially separable. Run on the real data for real numbers.
* With PortScan hidden from training, Isolation Forest alone flagged 100% of it at a 1.06% benign false-alarm rate.
* Latency measured on the synthetic data: 0.005 ms/flow in batches, 4–5 ms/flow one flow at a time. Sub-millisecond latency holds
  only when flows are scored in micro-batches (`--chunk`). Single-flow calls are slower because of per-call overhead.

## Limitations

1. CICIDS2017 is lab traffic from 2017, so published ~99% scores are optimistic. Expect a drop on real networks.
2. Live use needs a flow extractor (CICFlowMeter) that produces the same feature schema as training.
3. Encrypted traffic (TLS) hides payloads. Flow statistics still work, but web attacks inside HTTPS are hard.
4. Adversarial traffic crafted to look benign can evade tree models. Add adversarial training as a next step.
5. Anomaly detection means extra false alerts (the 1% calibration setting) and needs analyst triage.
6. Feature selection saw the labels of a held-out class in `--holdout` runs, so zero-day numbers are slightly optimistic.
7. Source-IP blocking is blunt: spoofed or NATed addresses can cause collateral damage. Add rule expiry for production.

## Layout

```
src/common.py       shared labels, feature transform, ensemble + anomaly helpers
src/preprocess.py   phase 1       src/features.py   phases 2-3
src/train.py        phase 4       src/explain.py    phase 5
src/monitor.py      phase 6       src/make_synthetic.py  smoke-test data
```
