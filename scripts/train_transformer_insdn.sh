#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
python -m sdn_nids_realtime.train_v2.train_classifier   --dataset-dir datasets/InSDN   --prepared-dir artifacts_v4/prepared_classifier   --output-dir artifacts_v4/models/transformer   --bundle-dir artifacts_v4/bundles   --model transformer   --seq-len 4   --feature-scheme insdn_runtime10_v2   --threshold-strategy target_fpr   --device cpu
