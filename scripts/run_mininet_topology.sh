#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH}"
export SDN_NIDS_CONTROLLER_IP="${SDN_NIDS_CONTROLLER_IP:-127.0.0.1}"
export SDN_NIDS_CONTROLLER_PORT="${SDN_NIDS_CONTROLLER_PORT:-6653}"
export SDN_NIDS_TOPO_HOSTS="${SDN_NIDS_TOPO_HOSTS:-7}"

sudo -E python -m sdn_nids_realtime.demo.topology \
  --hosts "$SDN_NIDS_TOPO_HOSTS" \
  --controller-ip "$SDN_NIDS_CONTROLLER_IP" \
  --controller-port "$SDN_NIDS_CONTROLLER_PORT"
