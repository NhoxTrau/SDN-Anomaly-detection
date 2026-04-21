from __future__ import annotations

from statistics import mean
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def pressure_state(*, runtime_queue_depth: int, raw_queue_depth: int, timeouts: int, avg_reply_delay_ms: float, queue_threshold: int, raw_threshold: int, reply_delay_high_ms: float) -> str:
    if timeouts > 0 or runtime_queue_depth >= queue_threshold or raw_queue_depth >= raw_threshold or avg_reply_delay_ms >= reply_delay_high_ms:
        return "high"
    if runtime_queue_depth >= max(1, queue_threshold // 3) or raw_queue_depth >= max(1, raw_threshold // 3) or avg_reply_delay_ms >= (reply_delay_high_ms * 0.6):
        return "medium"
    return "normal"


def recommended_poll_interval(
    *,
    current_interval_s: float,
    min_interval_s: float,
    max_interval_s: float,
    runtime_queue_depth: int,
    raw_queue_depth: int,
    avg_reply_delay_ms: float,
    timeout_total: int,
    throughput_obs_s: float,
    queue_threshold: int,
    raw_threshold: int,
    reply_delay_high_ms: float,
) -> float:
    """Heuristic interval recommendation for thesis experiments.

    Goal: stabilize polling under bursty flood traffic without making benign runs
    feel sluggish. We bias toward increasing the interval under pressure and
    gently decreasing it only when the pipeline is healthy.
    """
    interval = float(current_interval_s)
    if timeout_total > 0:
        interval = max(interval, min(max_interval_s, interval * 1.20 + 0.10))
    if avg_reply_delay_ms >= reply_delay_high_ms:
        interval = max(interval, min(max_interval_s, interval * 1.10 + 0.05))
    if runtime_queue_depth >= queue_threshold:
        interval = max(interval, min(max_interval_s, interval * 1.15 + 0.10))
    elif runtime_queue_depth <= max(16, queue_threshold // 12) and raw_queue_depth <= max(8, raw_threshold // 12) and avg_reply_delay_ms <= max(10.0, reply_delay_high_ms * 0.4) and timeout_total == 0 and throughput_obs_s > 0:
        interval = min(interval, max(min_interval_s, interval * 0.95))
    if raw_queue_depth >= raw_threshold:
        interval = max(interval, min(max_interval_s, interval * 1.10 + 0.05))
    return max(min_interval_s, min(max_interval_s, interval))


def summarize_polling_metrics(polling_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list((polling_stats or {}).values())
    if not rows:
        return {
            "switch_count": 0,
            "timeout_total": 0,
            "avg_reply_delay_ms": 0.0,
            "max_reply_delay_ms": 0.0,
            "avg_flows_per_reply": 0.0,
            "trimmed_flows_total": 0,
        }
    return {
        "switch_count": len(rows),
        "timeout_total": sum(_safe_int(r.get("timeouts", 0)) for r in rows),
        "avg_reply_delay_ms": mean(_safe_float(r.get("avg_reply_delay_ms", 0.0)) for r in rows),
        "max_reply_delay_ms": max(_safe_float(r.get("max_reply_delay_ms", 0.0)) for r in rows),
        "avg_flows_per_reply": mean(_safe_float(r.get("avg_flows_per_reply", 0.0)) for r in rows),
        "trimmed_flows_total": sum(_safe_int(r.get("trimmed_flows", 0)) for r in rows),
    }
