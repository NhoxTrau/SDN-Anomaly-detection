#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
python -m sdn_nids_realtime.train_v2.prepare_data   --dataset-dir datasets/InSDN   --save-dir artifacts_v4/prepared_classifier   --seq-len 4   --feature-scheme insdn_ml_core_v1
