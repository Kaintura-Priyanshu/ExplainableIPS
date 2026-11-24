#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python src/make_synthetic.py
python src/preprocess.py
python src/features.py --k-features 10
python src/train.py --trees 60 --holdout PortScan   # zero-day test
python src/monitor.py --flows data/raw/synthetic.csv
echo "Alerts with explanations: reports/alerts.jsonl"
