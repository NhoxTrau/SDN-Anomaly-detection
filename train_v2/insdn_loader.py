from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import LABEL_TO_INDEX, clean_numeric_features, feature_names_for_scheme, is_normal_label, normalise_label, normalize_port

REQUIRED_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
    "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Label",
]
DEFAULT_FILE_NAMES = ["Normal_data.csv", "OVS.csv", "metasploitable-2.csv"]


def _safe_read_csv(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_timestamps(series: pd.Series, fallback_index: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    fallback = pd.Timestamp("1970-01-01") + pd.to_timedelta(fallback_index, unit="ms")
    return parsed.where(parsed.notna(), fallback)


def _derive_ml_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    src_port = pd.to_numeric(df.get("Src Port"), errors="coerce").fillna(0.0)
    dst_port = pd.to_numeric(df.get("Dst Port"), errors="coerce").fillna(0.0)
    protocol = pd.to_numeric(df.get("Protocol"), errors="coerce").fillna(0.0)
    fwd_pkts = pd.to_numeric(df.get("Tot Fwd Pkts"), errors="coerce").fillna(0.0)
    bwd_pkts = pd.to_numeric(df.get("Tot Bwd Pkts"), errors="coerce").fillna(0.0)
    fwd_bytes = pd.to_numeric(df.get("TotLen Fwd Pkts"), errors="coerce").fillna(0.0)
    bwd_bytes = pd.to_numeric(df.get("TotLen Bwd Pkts"), errors="coerce").fillna(0.0)
    duration_us = pd.to_numeric(df.get("Flow Duration"), errors="coerce").fillna(0.0)

    packet_count = (fwd_pkts + bwd_pkts).clip(lower=0.0)
    byte_count = (fwd_bytes + bwd_bytes).clip(lower=0.0)
    flow_duration_s = (duration_us / 1e6).clip(lower=0.0)
    duration_safe = flow_duration_s.replace(0.0, 1e-6)

    out["packet_count"] = packet_count
    out["byte_count"] = byte_count
    out["flow_duration_s"] = flow_duration_s
    out["packet_rate"] = packet_count / duration_safe
    out["byte_rate"] = byte_count / duration_safe
    out["avg_packet_size"] = np.where(packet_count > 0, byte_count / np.maximum(packet_count, 1.0), 0.0)
    out["src_port_norm"] = src_port.apply(normalize_port)
    out["dst_port_norm"] = dst_port.apply(normalize_port)
    out["protocol_tcp"] = (protocol == 6).astype(np.float32)
    out["protocol_udp"] = (protocol == 17).astype(np.float32)
    out["protocol_icmp"] = (protocol == 1).astype(np.float32)
    out["flow_id"] = df.get("Flow ID", "").astype(str).str.strip()
    out["src_ip"] = df.get("Src IP", "").astype(str).str.strip()
    out["dst_ip"] = df.get("Dst IP", "").astype(str).str.strip()
    out["src_port"] = src_port.astype(np.int64)
    out["dst_port"] = dst_port.astype(np.int64)
    out["protocol"] = protocol.astype(np.int64)
    out["timestamp"] = df.get("Timestamp", "")
    out["label"] = df.get("Label", "Normal").apply(normalise_label)
    out["is_attack"] = (~out["label"].apply(is_normal_label)).astype(np.int64)
    out["label_index"] = out["label"].map(LABEL_TO_INDEX).fillna(0).astype(np.int64)
    return out


def _default_conversation_key(df: pd.DataFrame) -> pd.Series:
    src_ip = df.get("src_ip", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    dst_ip = df.get("dst_ip", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    dst_port = pd.to_numeric(df.get("dst_port", -1), errors="coerce").fillna(-1).astype(int).astype(str)
    protocol = pd.to_numeric(df.get("protocol", -1), errors="coerce").fillna(-1).astype(int).astype(str)
    return src_ip + "__" + dst_ip + "__" + dst_port + "__" + protocol


def load_insdn_dataframe(dataset_dir: str | Path, file_names: Iterable[str] | None = None) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    file_names = list(file_names or DEFAULT_FILE_NAMES)
    frames = []
    for file_name in file_names:
        path = dataset_dir / file_name
        if not path.exists():
            continue
        raw = _safe_read_csv(path)
        missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        mapped = _derive_ml_core_features(raw)
        mapped["source_file"] = path.name
        mapped["row_id"] = np.arange(len(mapped), dtype=np.int64)
        mapped["parsed_timestamp"] = _parse_timestamps(mapped["timestamp"], mapped["row_id"])
        mapped["conversation_key"] = _default_conversation_key(mapped)
        frames.append(mapped)
    if not frames:
        raise FileNotFoundError(f"No InSDN CSV files found in {dataset_dir}")
    full = pd.concat(frames, ignore_index=True)
    feature_cols = feature_names_for_scheme("insdn_ml_core_v1")
    full = clean_numeric_features(full, feature_names=feature_cols)
    full = full.replace([np.inf, -np.inf], 0.0)
    return full.dropna(subset=["conversation_key", "parsed_timestamp"]).reset_index(drop=True)
