from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CONVERTED_DIR,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_RAW_CICIDS_DIR,
    LABEL_TO_INDEX,
    clean_numeric_features,
    feature_names_for_scheme,
    normalise_label,
    resolve_project_path,
)

# ---------------------------------------------------------------------------
# Column alias map — maps canonical key → list of possible CICIDS column names
# Extended to cover fwd/bwd split columns needed for telemetry_v2.
# ---------------------------------------------------------------------------
RAW_ALIAS_MAP = {
    "src port":            ["Source Port", "Src Port", "source port"],
    "dst port":            ["Destination Port", "Dst Port", "destination port"],
    "protocol":            ["Protocol", "protocol"],
    "flow duration":       ["Flow Duration"],
    "tot fwd pkts":        ["Total Fwd Packets", "Tot Fwd Pkts"],
    "tot bwd pkts":        ["Total Backward Packets", "Tot Bwd Pkts"],
    "totlen fwd pkts":     ["Total Length of Fwd Packets", "TotLen Fwd Pkts"],
    "totlen bwd pkts":     ["Total Length of Bwd Packets", "TotLen Bwd Pkts"],
    "flow byts/s":         ["Flow Bytes/s", "Flow Byts/s"],
    "flow pkts/s":         ["Flow Packets/s", "Flow Pkts/s"],
    "fwd pkts/s":          ["Fwd Packets/s", "Fwd Pkts/s"],
    "bwd pkts/s":          ["Bwd Packets/s"],
    "label":               ["Label", "label", "Class"],
}


def discover_csv_files(dataset_dir: str | Path) -> list[Path]:
    dataset_path = resolve_project_path(dataset_dir)
    if dataset_path.is_file():
        return [dataset_path]
    files = sorted(p for p in dataset_path.glob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    return files


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [str(c).strip() for c in columns]


def _normalize_col(name: str) -> str:
    low = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(low.split())


def _canonical_map(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_normalize_col(c): c for c in columns}
    out: dict[str, str] = {}
    for key, aliases in RAW_ALIAS_MAP.items():
        for alias in aliases:
            alias_key = _normalize_col(alias)
            if alias_key in normalized:
                out[key] = normalized[alias_key]
                break
    return out


def _get_series(df: pd.DataFrame, mapping: dict[str, str], key: str, default=None) -> pd.Series:
    col = mapping.get(key)
    if col is None or col not in df.columns:
        if default is None:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _parse_timestamps(series: pd.Series, fallback_index: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    fallback = pd.Timestamp("1970-01-01") + pd.to_timedelta(fallback_index, unit="ms")
    return parsed.where(parsed.notna(), fallback)


def _coarse_label(label: object) -> str:
    return normalise_label(label)


def _read_csv_safely(file_path: Path, chunksize: int | None):
    if chunksize and chunksize > 0:
        return pd.read_csv(file_path, low_memory=False, chunksize=chunksize)
    return [pd.read_csv(file_path, low_memory=False)]


def _safe_duration_s(flow_duration_us: pd.Series) -> np.ndarray:
    duration_s = (flow_duration_us.clip(lower=0.0) / 1_000_000.0).astype(np.float64)
    return duration_s.where(duration_s > 0.0, 0.001).to_numpy(dtype=np.float64)


def derive_telemetry_features(
    df: pd.DataFrame,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    source_file: str | None = None,
    feature_scheme: str = "telemetry_v2",
) -> pd.DataFrame:
    """Convert CICIDS flow rows into controller-aligned telemetry features.

    Supports both telemetry_v1 (legacy) and telemetry_v2 (recommended).

    IMPORTANT — CICIDS vs real-time delta features:
    ------------------------------------------------
    MachineLearningCSV contains *completed flow summaries*, not per-poll snapshots.
    For telemetry_v2, delta features (packet_delta, byte_delta, etc.) are initialized
    here as 0.0 placeholders. They are then recomputed as proper within-group
    row-to-row differences in split_grouped_sequences_pre_window(), AFTER the data
    has been sorted chronologically within each conversation group.

    This faithfully simulates what a stateful SDN controller would see: consecutive
    poll responses for flows of the same type, where each Δ measures the change
    from the previous observation.

    fwd_pkt_fraction / fwd_byte_fraction:
        Derived from CICIDS fwd/bwd split columns. In real-time SDN, these are
        available when bidirectional flow table entries are installed.
    inter_poll_gap_s:
        In real-time: actual seconds between controller polls of a specific flow.
        In CICIDS training: proxied by flow_duration_s (each completed flow record
        acts as one polling interval), which is much more informative than a
        hard-coded constant of 1.0.
    """
    data = df.copy()
    data.columns = _normalize_columns(data.columns)
    data["source_file"] = source_file or "unknown"

    mapping = _canonical_map(data.columns)

    data["src_ip"] = ""
    data["dst_ip"] = ""
    data["src_port"] = _num(_get_series(data, mapping, "src port", -1), -1).astype(int)
    data["protocol"] = _num(_get_series(data, mapping, "protocol", -1), -1).astype(int)
    data["dst_port"] = _num(_get_series(data, mapping, "dst port", -1), -1).astype(int)

    data["label"] = _get_series(data, mapping, "label", "Normal").astype(str).apply(_coarse_label)
    data["is_attack"] = (data["label"] != "Normal").astype(np.int64)
    data["y_multi"] = data["label"].map(LABEL_TO_INDEX).fillna(0).astype(np.int64)

    data["row_id"] = np.arange(len(data), dtype=np.int64)
    data["parsed_timestamp"] = _parse_timestamps(
        pd.Series([pd.NaT] * len(data), index=data.index),
        data["row_id"],
    )

    # --- Core flow stats (available from OF counters in real-time) ---
    flow_duration_us = _num(_get_series(data, mapping, "flow duration", 0.0), 0.0)
    fwd_pkts = _num(_get_series(data, mapping, "tot fwd pkts", 0.0), 0.0)
    bwd_pkts = _num(_get_series(data, mapping, "tot bwd pkts", 0.0), 0.0)
    fwd_len = _num(_get_series(data, mapping, "totlen fwd pkts", 0.0), 0.0)
    bwd_len = _num(_get_series(data, mapping, "totlen bwd pkts", 0.0), 0.0)
    flow_byts_s = _num(_get_series(data, mapping, "flow byts/s", 0.0), 0.0)
    flow_pkts_s = _num(_get_series(data, mapping, "flow pkts/s", 0.0), 0.0)

    duration_s = _safe_duration_s(flow_duration_us)
    packet_count = (fwd_pkts + bwd_pkts).to_numpy(dtype=np.float64)
    byte_count = (fwd_len + bwd_len).to_numpy(dtype=np.float64)
    avg_packet_size = byte_count / np.maximum(packet_count, 1.0)

    # Prefer CICFlowMeter rate columns; fall back to counts / duration
    packet_rate = flow_pkts_s.to_numpy(dtype=np.float64)
    byte_rate = flow_byts_s.to_numpy(dtype=np.float64)
    missing_pkt_rate = ~np.isfinite(packet_rate) | (packet_rate <= 0.0)
    missing_byte_rate = ~np.isfinite(byte_rate) | (byte_rate <= 0.0)
    packet_rate[missing_pkt_rate] = packet_count[missing_pkt_rate] / np.maximum(duration_s[missing_pkt_rate], 1e-6)
    byte_rate[missing_byte_rate] = byte_count[missing_byte_rate] / np.maximum(duration_s[missing_byte_rate], 1e-6)

    # Store base features
    data["flow_duration_s"] = duration_s
    data["packet_count"] = packet_count
    data["byte_count"] = byte_count
    data["packet_rate"] = packet_rate
    data["byte_rate"] = byte_rate
    data["avg_packet_size"] = avg_packet_size

    # --- Delta features: initialized to 0.0 here ---
    # IMPORTANT: These will be overwritten by within-group row-to-row differences
    # in split_grouped_sequences_pre_window() after chronological sorting.
    # Do NOT set these equal to packet_count/byte_count (that was the old bug).
    data["packet_delta"] = 0.0
    data["byte_delta"] = 0.0
    data["packet_rate_delta"] = 0.0
    data["byte_rate_delta"] = 0.0

    data["dst_port_norm"] = np.clip(
        data["dst_port"].astype(np.float64), 0.0, 65535.0
    ) / 65535.0

    if feature_scheme == "telemetry_v2":
        # --- Directional asymmetry features (telemetry_v2 only) ---
        # Discriminates scan/probe (high fwd, low bwd) from normal bidirectional flows.
        # In real-time SDN: computable when bidirectional match rules are installed,
        # each direction tracked by a separate flow entry.
        total_pkts_arr = packet_count  # already np.float64
        total_bytes_arr = byte_count
        fwd_pkts_arr = fwd_pkts.to_numpy(dtype=np.float64)
        fwd_len_arr = fwd_len.to_numpy(dtype=np.float64)
        data["fwd_pkt_fraction"] = np.clip(
            fwd_pkts_arr / np.maximum(total_pkts_arr, 1.0), 0.0, 1.0
        )
        data["fwd_byte_fraction"] = np.clip(
            fwd_len_arr / np.maximum(total_bytes_arr, 1.0), 0.0, 1.0
        )
        # Protocol encoding [0, 1] — TCP≈0.247, UDP≈0.067, ICMP≈0.004
        data["protocol_norm"] = np.clip(
            data["protocol"].astype(np.float64) / 255.0, 0.0, 1.0
        )
        # inter_poll_gap_s proxy: use flow_duration_s (each CICIDS row ≈ one poll interval)
        # This is far more informative than the hardcoded constant 1.0 used previously.
        data["inter_poll_gap_s"] = np.maximum(duration_s, 0.001)

    elif feature_scheme == "telemetry_v1":
        # Legacy: keep old features for backward compatibility
        # These have the known duplication issues but are retained for existing models.
        data["flow_age_s"] = duration_s.copy()       # == flow_duration_s (duplicate)
        data["inter_poll_gap_s"] = np.full(len(data), float(poll_interval_s), dtype=np.float64)

    proto_text = data["protocol"].astype(int).astype(str)
    port_text = data["dst_port"].astype(int).astype(str)
    data["conversation_key"] = (
        data["source_file"].astype(str) + "__proto_" + proto_text + "__dstport_" + port_text
    )

    feature_cols = feature_names_for_scheme(feature_scheme)
    out_cols = [
        "source_file", "row_id", "parsed_timestamp", "conversation_key",
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "label", "is_attack", "y_multi",
        *feature_cols,
    ]
    for col in out_cols:
        if col not in data.columns:
            data[col] = 0.0
    return clean_numeric_features(data[out_cols], feature_cols)


def convert_cicids_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    chunksize: int | None = 200_000,
    limit_files: int | None = None,
    feature_scheme: str = "telemetry_v2",
) -> dict[str, object]:
    input_path = resolve_project_path(input_dir)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    converted_dir = output_path / "telemetry_rows"
    converted_dir.mkdir(parents=True, exist_ok=True)

    files = discover_csv_files(input_path)
    if limit_files:
        files = files[:limit_files]

    combined_csv = output_path / "telemetry_dataset.csv"
    if combined_csv.exists():
        combined_csv.unlink()

    manifest: dict[str, object] = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "feature_scheme": feature_scheme,
        "poll_interval_s": float(poll_interval_s),
        # NOTE: delta_mode is now "within_group_diff" — deltas are computed
        # as row-to-row differences within each conversation group after sorting,
        # NOT as copies of cumulative counts.
        "delta_mode": "within_group_diff_placeholder",
        "delta_finalization": "split_grouped_sequences_pre_window",
        "conversation_key": "source_file_protocol_dstport",
        "files": [],
        "n_rows": 0,
        "n_attack_rows": 0,
        "n_normal_rows": 0,
    }

    header_written = False
    for file_path in files:
        per_file_rows: list[pd.DataFrame] = []
        file_rows = 0
        file_attack = 0
        for chunk in _read_csv_safely(file_path, chunksize):
            if isinstance(chunk, pd.DataFrame):
                chunk.columns = _normalize_columns(chunk.columns)
                telemetry_df = derive_telemetry_features(
                    chunk,
                    poll_interval_s=poll_interval_s,
                    source_file=file_path.stem,
                    feature_scheme=feature_scheme,
                )
                per_file_rows.append(telemetry_df)
                file_rows += len(telemetry_df)
                file_attack += int(telemetry_df["is_attack"].sum())
        if not per_file_rows:
            continue

        combined = pd.concat(per_file_rows, ignore_index=True)
        combined.to_csv(converted_dir / f"{file_path.stem}.telemetry.csv", index=False)
        # Append to combined CSV — write header only on first file
        combined.to_csv(combined_csv, index=False, mode="a", header=not header_written)
        header_written = True

        manifest["files"].append({
            "file": file_path.name,
            "rows": int(file_rows),
            "attack_rows": int(file_attack),
            "normal_rows": int(file_rows - file_attack),
        })
        manifest["n_rows"] += int(file_rows)
        manifest["n_attack_rows"] += int(file_attack)
        manifest["n_normal_rows"] += int(file_rows - file_attack)

    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CICIDS MachineLearningCSV files into controller-compatible telemetry rows."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_RAW_CICIDS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_CONVERTED_DIR))
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument(
        "--feature-scheme", default="telemetry_v2",
        choices=["telemetry_v1", "telemetry_v2"],
        help="Feature scheme to use. telemetry_v2 is recommended (fixes duplication bugs in v1).",
    )
    args = parser.parse_args()

    manifest = convert_cicids_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poll_interval_s=args.poll_interval,
        chunksize=args.chunksize,
        limit_files=args.limit_files,
        feature_scheme=args.feature_scheme,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
