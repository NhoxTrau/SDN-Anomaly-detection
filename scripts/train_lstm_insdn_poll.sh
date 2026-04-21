#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$(cd "$PROJECT_ROOT/.." && pwd):${PYTHONPATH}"
POLL_INTERVAL="${SDN_NIDS_POLL_INTERVAL:-1.5}"
SEQ_LEN="${SDN_NIDS_SEQ_LEN:-4}"
KEY_MODE="${SDN_NIDS_KEY_MODE:-flow}"
PREP_DIR="${SDN_NIDS_PREPARED_POLL_DIR:-artifacts_v4/prepared_poll_classifier_t${POLL_INTERVAL//./p}_${KEY_MODE}}"
pybin="${PYTHON_BIN:-python3}"
if [ ! -f "$PREP_DIR/train.npz" ] || [ "${SDN_NIDS_FORCE_PREPARE:-0}" = "1" ]; then
  $pybin -m sdn_nids_realtime.train_v2.poll_sequence_builder \
    --dataset-dir datasets/InSDN \
    --save-dir "$PREP_DIR" \
    --poll-interval-s "$POLL_INTERVAL" \
    --max-polls "${SDN_NIDS_MAX_POLLS:-6}" \
    --seq-len "$SEQ_LEN" \
    --feature-scheme insdn_runtime10_v2 \
    --key-mode "$KEY_MODE" \
    --window-label-mode "${SDN_NIDS_WINDOW_LABEL_MODE:-any_attack_priority}" \
    --test-holdout-mode "${SDN_NIDS_TEST_HOLDOUT_MODE:-temporal}" \
    --min-val-groups-per-label "${SDN_NIDS_MIN_VAL_GROUPS_PER_LABEL:-1}" \
    --min-test-groups-per-label "${SDN_NIDS_MIN_TEST_GROUPS_PER_LABEL:-1}"
fi
$pybin -m sdn_nids_realtime.train_v2.train_classifier \
  --prepared-dir "$PREP_DIR" \
  --output-dir artifacts_v4/models/lstm_poll \
  --bundle-dir artifacts_v4/bundles \
  --model lstm \
  --seq-len "$SEQ_LEN" \
  --feature-scheme insdn_runtime10_v2 \
  --threshold-strategy target_fpr \
  --target-fpr "${SDN_NIDS_TARGET_FPR:-0.01}" \
  --balance-mode "${SDN_NIDS_BALANCE_MODE:-auto}" \
  --device "${SDN_NIDS_DEVICE:-cpu}"
