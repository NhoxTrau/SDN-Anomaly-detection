from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
DEFAULT_INSDN_DIR = DEFAULT_DATASET_DIR / "InSDN"
DEFAULT_RAW_CICIDS_DIR = DEFAULT_INSDN_DIR
DEFAULT_CONVERTED_DIR = PROJECT_ROOT / "artifacts_v4" / "converted"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "artifacts_v4" / "prepared_classifier"
DEFAULT_AE_PREPARED_DIR = PROJECT_ROOT / "artifacts_v4" / "prepared_autoencoder"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts_v4" / "models"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "runtime_logs"
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "artifacts_v4" / "bundles"
DEFAULT_RUN_ID = "realtime_demo"
RANDOM_SEED = 42
DEFAULT_POLL_INTERVAL_S = 2.0

ATTACK_LABELS = ["Normal", "DDoS", "DoS", "Probe", "BFA", "Web-Attack", "BOTNET", "U2R"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(ATTACK_LABELS)}

# Legacy 11-feature contract kept for backward compatibility with older bundles.
INSDN_ML_CORE_FEATURE_NAMES = [
    "packet_count",
    "byte_count",
    "flow_duration_s",
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "src_port_norm",
    "dst_port_norm",
    "protocol_tcp",
    "protocol_udp",
    "protocol_icmp",
]

# Runtime-safe 10-feature contract: drops src_port_norm because runtime aggregation keys merge
# multiple concurrent subflows by (src_ip, dst_ip, dst_port, proto). Once src_port is merged,
# training on it becomes semantically inconsistent with realtime inference.
INSDN_RUNTIME10_FEATURE_NAMES = [
    "packet_count",
    "byte_count",
    "flow_duration_s",
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "dst_port_norm",
    "protocol_tcp",
    "protocol_udp",
    "protocol_icmp",
]

TELEMETRY_FEATURE_NAMES = INSDN_RUNTIME10_FEATURE_NAMES
TELEMETRY_FEATURE_NAMES_V2 = INSDN_RUNTIME10_FEATURE_NAMES
DEFAULT_FEATURE_SCHEME = "insdn_runtime10_v2"
FEATURE_SCHEMES = {
    "insdn_runtime10_v2": INSDN_RUNTIME10_FEATURE_NAMES,
    "insdn_ml_core_v1": INSDN_ML_CORE_FEATURE_NAMES,
    "insdn_openflow_v1": INSDN_RUNTIME10_FEATURE_NAMES,
    "telemetry_v2": INSDN_RUNTIME10_FEATURE_NAMES,
    "telemetry_v1": INSDN_ML_CORE_FEATURE_NAMES,
}


def feature_names_for_scheme(feature_scheme: str) -> list[str]:
    if feature_scheme not in FEATURE_SCHEMES:
        raise ValueError(f"Unknown feature scheme: {feature_scheme}. Available: {list(FEATURE_SCHEMES)}")
    return list(FEATURE_SCHEMES[feature_scheme])


def resolve_project_path(pathlike: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(pathlike)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalise_label(raw_label: object) -> str:
    text = str(raw_label).strip()
    low = text.lower().replace("_", " ").replace("-", " ")
    low = " ".join(low.split())
    if low in {"", "nan", "none"}:
        return "Normal"
    if low in {"normal", "benign", "background"}:
        return "Normal"
    if "ddos" in low:
        return "DDoS"
    if low.startswith("dos"):
        return "DoS"
    if "probe" in low or "scan" in low:
        return "Probe"
    if "brute" in low or low == "bfa" or "patator" in low:
        return "BFA"
    if "web attack" in low or "xss" in low or "sql" in low:
        return "Web-Attack"
    if "bot" in low:
        return "BOTNET"
    if low == "u2r" or "infiltration" in low:
        return "U2R"
    return text if text else "Normal"


def is_normal_label(label: object) -> bool:
    return normalise_label(label) == "Normal"


def clean_numeric_features(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    feature_names = feature_names or []
    data = df.copy()
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if feature_names:
        data[feature_names] = data[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def signed_log1p_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))


def apply_transform_array(X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    del feature_names
    return signed_log1p_array(X)


def load_scaler_stats(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_port(port: float | int) -> float:
    try:
        port_value = float(port)
    except Exception:
        port_value = 0.0
    return float(min(max(port_value, 0.0), 65535.0) / 65535.0)


@dataclass
class FlowTelemetryState:
    key: str
    dpid: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    first_seen_ts: float
    last_seen_ts: float
    last_poll_ts: float
    last_packet_count: float = 0.0
    last_byte_count: float = 0.0
    last_packet_rate: float = 0.0
    last_byte_rate: float = 0.0
    seen_polls: int = 0

    @property
    def flow_age_s(self) -> float:
        return max(0.0, self.last_seen_ts - self.first_seen_ts)

    def compute_runtime_stats(
        self,
        packet_count: float,
        byte_count: float,
        duration_s: float,
        poll_interval_s: float,
    ) -> dict[str, float | bool]:
        duration_safe = max(float(duration_s), 1e-6)
        packet_count = float(max(packet_count, 0.0))
        byte_count = float(max(byte_count, 0.0))
        packet_rate = packet_count / duration_safe
        byte_rate = byte_count / duration_safe
        avg_packet_size = byte_count / max(packet_count, 1.0)
        has_history = self.seen_polls > 0
        packet_delta = max(0.0, packet_count - self.last_packet_count) if has_history else 0.0
        byte_delta = max(0.0, byte_count - self.last_byte_count) if has_history else 0.0
        packet_rate_delta = (packet_rate - self.last_packet_rate) if has_history else 0.0
        byte_rate_delta = (byte_rate - self.last_byte_rate) if has_history else 0.0
        return {
            "packet_count": packet_count,
            "byte_count": byte_count,
            "flow_duration_s": float(max(duration_s, 0.0)),
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "avg_packet_size": avg_packet_size,
            "src_port_norm": float(normalize_port(self.src_port)),
            "dst_port_norm": float(normalize_port(self.dst_port)),
            "protocol_tcp": 1.0 if int(self.protocol) == 6 else 0.0,
            "protocol_udp": 1.0 if int(self.protocol) == 17 else 0.0,
            "protocol_icmp": 1.0 if int(self.protocol) == 1 else 0.0,
            "packet_delta": packet_delta,
            "byte_delta": byte_delta,
            "packet_rate_delta": packet_rate_delta,
            "byte_rate_delta": byte_rate_delta,
            "poll_interval_s": float(max(poll_interval_s, 1e-6)),
            "has_history": bool(has_history),
        }

    def to_feature_vector(self, stats: dict[str, float | bool], feature_names: Iterable[str]) -> np.ndarray:
        return np.asarray([float(stats.get(name, 0.0)) for name in feature_names], dtype=np.float32)

    def update_from_stats(
        self,
        packet_count: float,
        byte_count: float,
        duration_s: float,
        poll_ts: float,
        poll_interval_s: float,
        feature_names: Iterable[str],
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        stats = self.compute_runtime_stats(packet_count, byte_count, duration_s, poll_interval_s)
        self.last_packet_count = float(stats["packet_count"])
        self.last_byte_count = float(stats["byte_count"])
        self.last_poll_ts = float(poll_ts)
        self.last_seen_ts = float(poll_ts)
        self.last_packet_rate = float(stats["packet_rate"])
        self.last_byte_rate = float(stats["byte_rate"])
        self.seen_polls += 1
        return self.to_feature_vector(stats, feature_names), stats


@dataclass
class FlowObservation:
    key: str
    dpid: int
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    packet_count: float
    byte_count: float
    duration_s: float
    packet_rate: float
    byte_rate: float
    avg_packet_size: float
    packet_delta: float
    byte_delta: float
    packet_rate_delta: float
    byte_rate_delta: float
    has_history: bool
    feature_vector: np.ndarray
    poll_request_ts: float | None = None
    poll_reply_ts: float | None = None
    feature_enqueue_ts: float | None = None
    feature_dequeue_ts: float | None = None
    inference_enqueue_ts: float | None = None
    inference_start_ts: float | None = None
    inference_end_ts: float | None = None
    alert_emit_ts: float | None = None
    poll_cycle_id: int = 0
    reply_part_count: int = 0
    label: str | None = None
    source_file: str | None = None
