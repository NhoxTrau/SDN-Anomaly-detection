#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
bash scripts/prepare_insdn.sh
bash scripts/train_lstm_insdn.sh
bash scripts/train_transformer_insdn.sh
bash scripts/train_lstm_ae_insdn.sh
