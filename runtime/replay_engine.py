from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..train_v2.common import FlowObservation, feature_names_for_scheme, resolve_project_path
from .telemetry_runtime import TelemetryRuntime


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "parsed_timestamp" in df.columns:
        df["parsed_timestamp"] = pd.to_datetime(df["parsed_timestamp"], errors="coerce", format="mixed")
    return df


def replay_csv(runtime: TelemetryRuntime, csv_path: str | Path, feature_scheme: str | None = None, speed: float = 1.0) -> None:
    df = _load_csv(resolve_project_path(csv_path))
    feature_cols = runtime.feature_names if feature_scheme in (None, "", "bundle") else feature_names_for_scheme(feature_scheme)
    if df.empty:
        return
    sort_cols = [c for c in ["parsed_timestamp", "row_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    prev_ts = None
    for _, row in df.iterrows():
        feature_vec = row.reindex(feature_cols, fill_value=0.0).to_numpy(dtype=np.float32)
        ts = row["parsed_timestamp"].timestamp() if "parsed_timestamp" in df.columns and pd.notna(row["parsed_timestamp"]) else time.time()
        obs = FlowObservation(
            key=str(row.get("flow_key", row.get("conversation_key", f"{row.get('src_ip', '0.0.0.0')}->{row.get('dst_ip', '0.0.0.0')}:{row.get('dst_port', -1)}"))),
            dpid=int(row.get("switch_id", row.get("dpid", 0)) or 0),
            timestamp=float(ts),
            src_ip=str(row.get("src_ip", "0.0.0.0")),
            dst_ip=str(row.get("dst_ip", "0.0.0.0")),
            src_port=int(row.get("src_port", -1) or -1),
            dst_port=int(row.get("dst_port", -1) or -1),
            protocol=int(row.get("protocol", -1) or -1),
            packet_count=float(row.get("packet_count", 0.0) or 0.0),
            byte_count=float(row.get("byte_count", 0.0) or 0.0),
            duration_s=float(row.get("flow_duration_s", row.get("duration_s", 0.0)) or 0.0),
            packet_rate=float(row.get("packet_rate", 0.0) or 0.0),
            byte_rate=float(row.get("byte_rate", 0.0) or 0.0),
            avg_packet_size=float(row.get("avg_packet_size", 0.0) or 0.0),
            packet_delta=float(row.get("packet_delta", 0.0) or 0.0),
            byte_delta=float(row.get("byte_delta", 0.0) or 0.0),
            packet_rate_delta=float(row.get("packet_rate_delta", 0.0) or 0.0),
            byte_rate_delta=float(row.get("byte_rate_delta", 0.0) or 0.0),
            has_history=bool(int(row.get("has_history", 1) or 0)),
            feature_vector=feature_vec,
            label=str(row.get("label", "Normal")),
            source_file=str(row.get("source_file", "unknown")),
        )
        runtime.process_observation(obs)
        if prev_ts is not None and speed > 0 and "parsed_timestamp" in df.columns and pd.notna(row.get("parsed_timestamp")):
            gap = max(0.0, (row["parsed_timestamp"].timestamp() - prev_ts) / speed)
            if gap > 0:
                time.sleep(min(gap, 0.2))
        prev_ts = row["parsed_timestamp"].timestamp() if "parsed_timestamp" in df.columns and pd.notna(row.get("parsed_timestamp")) else prev_ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay prepared telemetry CSV into the realtime inference engine.")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--feature-scheme", default="bundle", choices=["bundle", "insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--run-id", default="replay")
    parser.add_argument("--drain-seconds", type=float, default=1.5)
    args = parser.parse_args()
    runtime = TelemetryRuntime(runtime_root=args.runtime_root, bundle_path=args.bundle_path, run_id=args.run_id)
    replay_csv(runtime, args.csv_path, feature_scheme=args.feature_scheme, speed=args.speed)
    time.sleep(max(0.0, float(args.drain_seconds)))
    runtime._flush_state(force=True)
    runtime.wait_for_csv_drain(timeout_s=max(10.0, float(args.drain_seconds) + 10.0))


if __name__ == "__main__":
    main()
