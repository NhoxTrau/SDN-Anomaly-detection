#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export SDN_NIDS_RUNTIME_ROOT="${SDN_NIDS_RUNTIME_ROOT:-$PROJECT_ROOT/runtime_logs}"
export SDN_NIDS_BUNDLE_PATH="${SDN_NIDS_BUNDLE_PATH:-$PROJECT_ROOT/artifacts_v4/bundles/lstm/runtime_bundle.json}"
export SDN_NIDS_RUN_ID="${SDN_NIDS_RUN_ID:-realtime_insdn}"

# Train/deploy contract already fixed by the latest bundle.
export SDN_NIDS_RUNTIME_KEY_MODE="${SDN_NIDS_RUNTIME_KEY_MODE:-flow}"
export SDN_NIDS_FLOW_MATCH_MODE="${SDN_NIDS_FLOW_MATCH_MODE:-conversation}"

# Realtime evaluation should stay temporally aligned with the trained sequence spacing.
export SDN_NIDS_POLL_INTERVAL="${SDN_NIDS_POLL_INTERVAL:-1.5}"
export SDN_NIDS_DASHBOARD_TIMELINE_INTERVAL_S="${SDN_NIDS_DASHBOARD_TIMELINE_INTERVAL_S:-$SDN_NIDS_POLL_INTERVAL}"
export SDN_NIDS_MIN_POLL_INTERVAL="${SDN_NIDS_MIN_POLL_INTERVAL:-1.0}"
export SDN_NIDS_MAX_POLL_INTERVAL="${SDN_NIDS_MAX_POLL_INTERVAL:-2.5}"
export SDN_NIDS_ADAPTIVE_POLLING="${SDN_NIDS_ADAPTIVE_POLLING:-0}"
export SDN_NIDS_POLL_REPLY_TIMEOUT="${SDN_NIDS_POLL_REPLY_TIMEOUT:-4.0}"
export SDN_NIDS_POLL_BACKPRESSURE_THRESHOLD="${SDN_NIDS_POLL_BACKPRESSURE_THRESHOLD:-8000}"
export SDN_NIDS_HIGH_PRESSURE_THRESHOLD="${SDN_NIDS_HIGH_PRESSURE_THRESHOLD:-5000}"
export SDN_NIDS_RAW_STATS_HIGH_PRESSURE_THRESHOLD="${SDN_NIDS_RAW_STATS_HIGH_PRESSURE_THRESHOLD:-2000}"
export SDN_NIDS_MAX_OBS_PER_REPLY_HIGH_PRESSURE="${SDN_NIDS_MAX_OBS_PER_REPLY_HIGH_PRESSURE:-4000}"
export SDN_NIDS_MAX_OBS_PER_REPLY_EXTREME_PRESSURE="${SDN_NIDS_MAX_OBS_PER_REPLY_EXTREME_PRESSURE:-2000}"
export SDN_NIDS_TELEMETRY_SKIP_QUEUE_THRESHOLD="${SDN_NIDS_TELEMETRY_SKIP_QUEUE_THRESHOLD:-2500}"
export SDN_NIDS_QUEUE_BACKPRESSURE_DROP_THRESHOLD="${SDN_NIDS_QUEUE_BACKPRESSURE_DROP_THRESHOLD:-7000}"

export SDN_NIDS_ENABLE_BLOCKING="${SDN_NIDS_ENABLE_BLOCKING:-0}"
export SDN_NIDS_INFERENCE_BATCH_MAX="${SDN_NIDS_INFERENCE_BATCH_MAX:-512}"
export SDN_NIDS_INFERENCE_BATCH_WAIT_MS="${SDN_NIDS_INFERENCE_BATCH_WAIT_MS:-1.5}"
export SDN_NIDS_FLOW_IDLE_TIMEOUT="${SDN_NIDS_FLOW_IDLE_TIMEOUT:-5}"
export SDN_NIDS_EMIT_ONLY_ON_CHANGE="${SDN_NIDS_EMIT_ONLY_ON_CHANGE:-1}"
export SDN_NIDS_MIN_EMIT_PACKET_DELTA="${SDN_NIDS_MIN_EMIT_PACKET_DELTA:-1}"
export SDN_NIDS_MIN_EMIT_BYTE_DELTA="${SDN_NIDS_MIN_EMIT_BYTE_DELTA:-1}"
export SDN_NIDS_FORCE_EMIT_AFTER="${SDN_NIDS_FORCE_EMIT_AFTER:-2}"

# Dashboard ingest: keep backward compatibility with old UDP variable names.
export SDN_NIDS_DASHBOARD_STREAM_HOST="${SDN_NIDS_DASHBOARD_STREAM_HOST:-127.0.0.1}"
export SDN_NIDS_DASHBOARD_STREAM_PORT="${SDN_NIDS_DASHBOARD_STREAM_PORT:-8765}"
export SDN_NIDS_DASHBOARD_UDP_HOST="$SDN_NIDS_DASHBOARD_STREAM_HOST"
export SDN_NIDS_DASHBOARD_UDP_PORT="$SDN_NIDS_DASHBOARD_STREAM_PORT"

if [[ -z "${SDN_NIDS_RULE_POLICY_JSON:-}" ]]; then
  read -r -d '' SDN_NIDS_RULE_POLICY_JSON <<'JSON' || true
{
  "min_alert_hits": 2,
  "alert_cooldown_seconds": 2.0,
  "alert_hold_seconds": 4.0,
  "rule_assist_ratio": 0.92,
  "attack_display_floor": 0.50,
  "scan_window_s": 6.0,
  "scan_unique_ports": 10,
  "scan_attack_unique_ports": 24,
  "bfa_window_s": 8.0,
  "bfa_attempt_threshold": 10,
  "bfa_attack_threshold": 20,
  "bfa_ports": [21, 22, 23],
  "hard_packet_rate": 5000.0,
  "hard_byte_rate": 5000000.0,
  "hard_packet_delta": 3000.0,
  "hard_byte_delta": 2500000.0,
  "volumetric_min_hits": 1,
  "baseline_window_s": 20.0,
  "baseline_min_samples": 4,
  "rate_multiplier_warn": 2.0,
  "rate_multiplier_attack": 3.5,
  "rules_only_attack_enabled": false,
  "emergency_rules_enabled": false,
  "udp_flood_packet_rate": 3500.0,
  "udp_flood_byte_rate": 2500000.0,
  "icmp_flood_packet_rate": 2500.0,
  "tcp_small_pkt_attack_rate": 2500.0,
  "small_pkt_avg_size_max": 120.0,
  "allowlisted_services": {}
}
JSON
  export SDN_NIDS_RULE_POLICY_JSON
fi

ryu-manager sdn_nids_realtime.controller.ryu_telemetry_controller
