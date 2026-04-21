#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

import numpy as np
import pandas as pd

try:
    from sdn_nids_realtime.train_v2.common import DEFAULT_INSDN_DIR, resolve_project_path
    from sdn_nids_realtime.train_v2.insdn_loader import load_insdn_dataframe
except ModuleNotFoundError:
    from train_v2.common import DEFAULT_INSDN_DIR, resolve_project_path
    from train_v2.insdn_loader import load_insdn_dataframe


def _select_tail_ratio(df: pd.DataFrame, take_last_ratio: float) -> pd.DataFrame:
    take_last_ratio = float(min(max(take_last_ratio, 0.0), 1.0))
    if take_last_ratio <= 0.0 or take_last_ratio >= 1.0 or df.empty:
        return df.copy()
    parts = []
    for source_file, grp in df.sort_values(["source_file", "parsed_timestamp", "row_id"]).groupby("source_file", sort=False):
        del source_file
        keep_n = max(1, int(np.ceil(len(grp) * take_last_ratio)))
        parts.append(grp.tail(keep_n))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def build_runtime_replay_dataframe(
    dataset_dir: str | Path = DEFAULT_INSDN_DIR,
    poll_interval_s: float = 1.0,
    max_polls: int = 6,
    take_last_ratio: float = 0.20,
    normal_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = load_insdn_dataframe(resolve_project_path(dataset_dir))
    base = base.sort_values(["source_file", "parsed_timestamp", "row_id"]).reset_index(drop=True)
    base = _select_tail_ratio(base, take_last_ratio)
    if normal_only:
        base = base.loc[base["label"].astype(str).str.upper().eq("NORMAL")].copy()
    frames: list[pd.DataFrame] = []
    for row in base.itertuples(index=False):
        duration_s = max(float(getattr(row, "flow_duration_s", 0.0) or 0.0), max(poll_interval_s, 1e-3))
        total_packets = max(float(getattr(row, "packet_count", 0.0) or 0.0), 0.0)
        total_bytes = max(float(getattr(row, "byte_count", 0.0) or 0.0), 0.0)
        steps = int(max(2, min(max_polls, np.ceil(duration_s / max(poll_interval_s, 1e-3)))))
        fracs = np.clip(np.arange(1, steps + 1, dtype=np.float64) / float(steps), 0.0, 1.0)
        pkt_counts = total_packets * fracs
        byte_counts = total_bytes * fracs
        durations = np.maximum(duration_s * fracs, 1e-6)
        pkt_rates = pkt_counts / durations
        byte_rates = byte_counts / durations
        avg_pkt = np.where(pkt_counts > 0.0, byte_counts / np.maximum(pkt_counts, 1.0), 0.0)
        pkt_delta = np.diff(np.concatenate([[0.0], pkt_counts]))
        byte_delta = np.diff(np.concatenate([[0.0], byte_counts]))
        pkt_rate_delta = np.diff(np.concatenate([[0.0], pkt_rates]))
        byte_rate_delta = np.diff(np.concatenate([[0.0], byte_rates]))
        base_ts = pd.Timestamp(getattr(row, "parsed_timestamp"))
        frames.append(pd.DataFrame({
            "flow_key": [f"replay::{getattr(row, 'source_file')}::{int(getattr(row, 'row_id'))}"] * steps,
            "conversation_key": [str(getattr(row, "conversation_key", ""))] * steps,
            "parsed_timestamp": [base_ts + pd.to_timedelta((i + 1) * poll_interval_s, unit="s") for i in range(steps)],
            "source_file": [str(getattr(row, "source_file"))] * steps,
            "label": [str(getattr(row, "label"))] * steps,
            "src_ip": [str(getattr(row, "src_ip"))] * steps,
            "dst_ip": [str(getattr(row, "dst_ip"))] * steps,
            "src_port": [int(getattr(row, "src_port"))] * steps,
            "dst_port": [int(getattr(row, "dst_port"))] * steps,
            "protocol": [int(getattr(row, "protocol"))] * steps,
            "packet_count": pkt_counts,
            "byte_count": byte_counts,
            "flow_duration_s": durations,
            "packet_rate": pkt_rates,
            "byte_rate": byte_rates,
            "avg_packet_size": avg_pkt,
            "packet_delta": pkt_delta,
            "byte_delta": byte_delta,
            "packet_rate_delta": pkt_rate_delta,
            "byte_rate_delta": byte_rate_delta,
            "has_history": [0] + [1] * (steps - 1),
            "packet_count_total": [total_packets] * steps,
            "byte_count_total": [total_bytes] * steps,
            "poll_step": np.arange(1, steps + 1, dtype=np.int64),
            "n_polls": [steps] * steps,
            "src_port_norm": [float(getattr(row, "src_port_norm", 0.0))] * steps,
            "dst_port_norm": [float(getattr(row, "dst_port_norm", 0.0))] * steps,
            "protocol_tcp": [float(getattr(row, "protocol_tcp", 0.0))] * steps,
            "protocol_udp": [float(getattr(row, "protocol_udp", 0.0))] * steps,
            "protocol_icmp": [float(getattr(row, "protocol_icmp", 0.0))] * steps,
        }))
    replay_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    replay_df = replay_df.sort_values(["parsed_timestamp", "source_file", "flow_key", "poll_step"]).reset_index(drop=True)
    summary = {
        "dataset_dir": str(resolve_project_path(dataset_dir)),
        "poll_interval_s": float(poll_interval_s),
        "max_polls": int(max_polls),
        "take_last_ratio": float(take_last_ratio),
        "normal_only": bool(normal_only),
        "n_source_rows": int(len(base)),
        "n_replay_rows": int(len(replay_df)),
        "labels": {str(k): int(v) for k, v in replay_df["label"].value_counts().to_dict().items()} if not replay_df.empty else {},
    }
    return replay_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime-aligned labeled replay CSV from InSDN completed-flow CSVs.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=6)
    parser.add_argument("--take-last-ratio", type=float, default=0.20)
    parser.add_argument("--normal-only", action="store_true")
    args = parser.parse_args()

    df, summary = build_runtime_replay_dataframe(
        dataset_dir=args.dataset_dir,
        poll_interval_s=args.poll_interval,
        max_polls=args.max_polls,
        take_last_ratio=args.take_last_ratio,
        normal_only=args.normal_only,
    )
    out_path = resolve_project_path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary["output_csv"] = str(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
