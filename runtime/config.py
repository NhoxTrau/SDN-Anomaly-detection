from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


@dataclass
class RuntimeConfig:
    poll_interval_s: float = 1.0
    dashboard_timeline_interval_s: float = 1.0
    inference_batch_max: int = 256
    inference_batch_wait_ms: float = 4.0
    queue_maxsize: int = 50000
    csv_queue_maxsize: int = 20000
    adaptive_polling: bool = False
    raw_stats_queue_maxsize: int = 4096

    def validated(self) -> "RuntimeConfig":
        self.poll_interval_s = max(0.10, float(self.poll_interval_s))
        self.dashboard_timeline_interval_s = max(0.10, float(self.dashboard_timeline_interval_s))
        self.inference_batch_max = max(1, int(self.inference_batch_max))
        self.inference_batch_wait_ms = max(0.0, float(self.inference_batch_wait_ms))
        self.queue_maxsize = max(256, int(self.queue_maxsize))
        self.csv_queue_maxsize = max(256, int(self.csv_queue_maxsize))
        self.raw_stats_queue_maxsize = max(128, int(self.raw_stats_queue_maxsize))
        self.adaptive_polling = bool(self.adaptive_polling)
        return self

    @classmethod
    def from_environment(cls, manifest: dict[str, Any] | None = None) -> "RuntimeConfig":
        manifest = manifest or {}
        return cls(
            poll_interval_s=_env_float("SDN_NIDS_POLL_INTERVAL", float(manifest.get("poll_interval_s", 1.0))),
            dashboard_timeline_interval_s=_env_float("SDN_NIDS_DASHBOARD_TIMELINE_INTERVAL_S", float(manifest.get("poll_interval_s", 1.0))),
            inference_batch_max=_env_int("SDN_NIDS_INFERENCE_BATCH_MAX", int(manifest.get("inference_batch_max", 256))),
            inference_batch_wait_ms=_env_float("SDN_NIDS_INFERENCE_BATCH_WAIT_MS", float(manifest.get("inference_batch_wait_ms", 4.0))),
            queue_maxsize=_env_int("SDN_NIDS_QUEUE_MAXSIZE", int(manifest.get("queue_maxsize", 50000))),
            csv_queue_maxsize=_env_int("SDN_NIDS_CSV_QUEUE_MAXSIZE", int(manifest.get("csv_queue_maxsize", 20000))),
            adaptive_polling=_truthy(os.environ.get("SDN_NIDS_ADAPTIVE_POLLING"), default=bool(manifest.get("adaptive_polling", False))),
            raw_stats_queue_maxsize=_env_int("SDN_NIDS_RAW_STATS_QUEUE_MAXSIZE", int(manifest.get("raw_stats_queue_maxsize", 4096))),
        ).validated()
