#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
python -m sdn_nids_realtime.train_v2.prepare_ae_data_v2   --dataset-dir datasets/InSDN   --save-dir artifacts_v4/prepared_autoencoder   --seq-len 4   --feature-scheme insdn_runtime10_v2
python -m sdn_nids_realtime.train_v2.run_train_autoencoder   --prepared-dir artifacts_v4/prepared_autoencoder   --output-dir artifacts_v4/models/lstm_autoencoder_insdn   --bundle-dir artifacts_v4/bundles   --seq-len 4   --feature-scheme insdn_runtime10_v2   --device cpu
