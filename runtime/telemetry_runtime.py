from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..train_v2.common import DEFAULT_RUN_ID, FlowObservation
from .config import RuntimeConfig
from .dashboard_stream import RuntimeDashboardPublisher
from .inference_engine import InferenceEngine
from .rules import RuleDecision, RuleEngine
from .scalability import recommended_poll_interval, summarize_polling_metrics
from .state_store import append_csv_row, atomic_write_json, cleanup_orphan_temp_files
from .telemetry_logger import get_logger, log_json

_ALLOWED_STATUSES = {"NORMAL", "SUSPECT", "ATTACK"}
_STATUS_RANK = {"NORMAL": 0, "SUSPECT": 1, "ATTACK": 2}


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _normalize_status(status: str | None, default: str = "NORMAL") -> str:
    text = str(status or default).strip().upper()
    return text if text in _ALLOWED_STATUSES else default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _decision_sort_key(decision: RuleDecision, score: float | None) -> tuple[int, float]:
    return (_STATUS_RANK.get(_normalize_status(decision.status), 0), float(score if score is not None else -1.0))


def _load_rule_policy_overrides(base_policy: dict[str, Any]) -> dict[str, Any]:
    policy = dict(base_policy or {})

    override_json = os.environ.get("SDN_NIDS_RULE_POLICY_JSON", "").strip()
    if override_json:
        try:
            loaded = json.loads(override_json)
            if isinstance(loaded, dict):
                policy.update(loaded)
        except Exception:
            pass

    allowlist_json = os.environ.get("SDN_NIDS_ALLOWLIST_JSON", "").strip()
    allowlist_path = os.environ.get("SDN_NIDS_ALLOWLIST_PATH", "").strip()
    allowlist_override = None
    if allowlist_json:
        try:
            loaded = json.loads(allowlist_json)
            if isinstance(loaded, dict):
                allowlist_override = loaded
        except Exception:
            allowlist_override = None
    elif allowlist_path:
        try:
            loaded = json.loads(Path(allowlist_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                allowlist_override = loaded
        except Exception:
            allowlist_override = None

    if allowlist_override is not None:
        policy["allowlisted_services"] = {
            str(ip): sorted({int(port) for port in ports})
            for ip, ports in allowlist_override.items()
            if isinstance(ports, (list, tuple, set))
        }

    if "SDN_NIDS_DISABLE_SCAN_RULE" in os.environ:
        policy["disable_scan_rule"] = _truthy_env("SDN_NIDS_DISABLE_SCAN_RULE")

    return policy


class TelemetryRuntime:
    def __init__(
        self,
        runtime_root: str | Path,
        bundle_path: str | Path,
        run_id: str = DEFAULT_RUN_ID,
        queue_maxsize: int = 50000,
        execution_provider: str = "cpu",
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.run_id = run_id
        self.runtime_dir = self.runtime_root / run_id
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_state = self.runtime_dir / "dashboard_state.json"
        self.controller_metrics_state = self.runtime_dir / "controller_metrics.json"
        self.model_info_state = self.runtime_dir / "model_info.json"
        self.alerts_csv = self.runtime_dir / "alerts.csv"
        self.telemetry_csv = self.runtime_dir / "telemetry_stream.csv"
        self.scores_csv = self.runtime_dir / "scores.csv"
        self.queue: Queue[FlowObservation] = Queue(maxsize=queue_maxsize)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.telemetry_logger = get_logger("sdn_nids.runtime")
        self.start_time = time.time()
        self.total_rows_read = 0
        self.total_inferences = 0
        self.total_batches = 0
        self.total_scored_rows = 0
        self.total_batch_latency_ms = 0.0
        self.total_runtime_seconds = 0.0
        self.dropped_rows = 0
        self.last_score: float | None = None
        self.last_latency_ms: float = 0.0
        self.last_batch_size: int = 0
        self.last_inference_batch_size: int = 0
        self.state_payload: dict[str, object] = {}
        self.recent_alerts: deque[dict[str, object]] = deque()
        self.recent_scores: deque[tuple[float, float, str, str]] = deque()
        self.sequence_buffers: dict[str, deque[np.ndarray]] = {}
        self.sequence_last_seen: dict[str, float] = {}
        self.last_observation_ts: float | None = None
        self.last_decision_status: str = "NORMAL"
        self.last_decision_reason: str = ""
        self.last_decision_source: str = "runtime"
        self.last_runtime_phase: str = "idle"
        self.last_decision_ts: float | None = None
        self.last_state_write_ts: float = 0.0
        self.has_scored_observation: bool = False
        self.last_scored_ts: float | None = None
        self.last_latency_breakdown: dict[str, float] = {}
        self._latency_breakdown_totals: dict[str, float] = {}
        self._latency_breakdown_counts: dict[str, int] = {}
        self._external_controller_metrics: dict[str, object] = {}
        self._last_controller_metrics_write_ts = 0.0
        self.controller_metrics_write_interval_s = max(0.25, _safe_float_env("SDN_NIDS_CONTROLLER_METRICS_WRITE_INTERVAL_S", 0.75))

        self.bundle_path = Path(bundle_path)
        manifest_path = self.bundle_path / "runtime_bundle.json" if self.bundle_path.is_dir() else self.bundle_path
        self.bundle_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        self.metrics_payload: dict[str, object] = {}
        self.feature_names = list(manifest.get("feature_names", []))
        self.n_features = len(self.feature_names)
        self.seq_len = int(manifest.get("seq_len", 4))
        self.task_type = str(manifest.get("task_type", "classifier"))
        self.model_name = str(manifest.get("model_name", "model"))
        self.feature_scheme = str(manifest.get("feature_scheme", "insdn_ml_core_v1"))
        self.score_direction = str(manifest.get("score_direction", manifest.get("thresholds", {}).get("score_direction", "higher_is_attack")))
        thresholds = manifest.get("thresholds", {})
        runtime_key = str(manifest.get("runtime_threshold_key", thresholds.get("runtime_threshold_key", "threshold")))
        self.threshold = float(thresholds.get(runtime_key, thresholds.get("threshold", 0.0)))
        self.config = RuntimeConfig.from_environment(manifest)
        self.poll_interval_s = float(self.config.poll_interval_s)
        self.dashboard_timeline_interval_s = float(self.config.dashboard_timeline_interval_s)

        rule_policy = _load_rule_policy_overrides(manifest.get("rule_policy", {}))
        self.rule_policy = rule_policy
        self.rule_engine = RuleEngine(rule_policy)
        self.alert_hold_seconds = float(rule_policy.get("alert_hold_seconds", 10.0))
        self.sequence_idle_timeout_s = float(manifest.get("sequence_idle_timeout_s", 120.0))
        self.idle_reset_seconds = float(manifest.get("runtime_idle_reset_seconds", max(2.0 * float(manifest.get("poll_interval_s", 2.0)), 6.0)))
        self.state_write_interval_s = float(manifest.get("state_write_interval_s", 0.75))
        self.telemetry_log_every_n = int(_safe_int_env("SDN_NIDS_TELEMETRY_LOG_EVERY_N", int(manifest.get("telemetry_log_every_n", 5))))
        self.telemetry_skip_queue_threshold = _safe_int_env("SDN_NIDS_TELEMETRY_SKIP_QUEUE_THRESHOLD", int(manifest.get("telemetry_skip_queue_threshold", 1500)))
        self.queue_backpressure_drop_threshold = _safe_int_env("SDN_NIDS_QUEUE_BACKPRESSURE_DROP_THRESHOLD", int(manifest.get("queue_backpressure_drop_threshold", 4000)))
        self.csv_queue_maxsize = int(self.config.csv_queue_maxsize)
        self.inference_batch_max = int(self.config.inference_batch_max)
        self.inference_batch_wait_ms = float(self.config.inference_batch_wait_ms)
        self.dashboard_stream_enabled = _truthy_env("SDN_NIDS_DASHBOARD_STREAM_ENABLED", default=True)
        self.dashboard_stream_host = os.environ.get("SDN_NIDS_DASHBOARD_STREAM_HOST") or os.environ.get("SDN_NIDS_DASHBOARD_UDP_HOST", "127.0.0.1")
        self.dashboard_stream_port = _safe_int_env("SDN_NIDS_DASHBOARD_STREAM_PORT", _safe_int_env("SDN_NIDS_DASHBOARD_UDP_PORT", 8765))
        self.dashboard_stream_state_interval_s = _safe_float_env("SDN_NIDS_DASHBOARD_STREAM_STATE_INTERVAL_S", 0.20)
        self._last_stream_state_ts = 0.0
        self._last_stream_state_sig = ""
        self.stream_drop_count = 0
        self.dashboard_publisher = RuntimeDashboardPublisher(
            enabled=self.dashboard_stream_enabled,
            host=self.dashboard_stream_host,
            port=self.dashboard_stream_port,
        )
        self.csv_drop_count = 0
        self._csv_queue: Queue[tuple[Path, tuple[str, ...], dict[str, object]]] = Queue(maxsize=self.csv_queue_maxsize)
        self._csv_thread = threading.Thread(target=self._csv_writer_loop, daemon=True)
        self._csv_thread.start()
        cleanup_orphan_temp_files(self.runtime_dir, max_age_s=60.0)
        metrics_filename = str(manifest.get("metrics_filename", "")).strip()
        if metrics_filename:
            metrics_path = self.bundle_dir / metrics_filename
            if metrics_path.exists():
                try:
                    self.metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    self.metrics_payload = {}

        self._write_model_info_state()

        self.inference_engine = InferenceEngine(
            bundle_dir=self.bundle_dir,
            manifest=manifest,
            feature_names=self.feature_names,
            task_type=self.task_type,
            execution_provider=execution_provider,
        )


    @staticmethod
    def _compact_metrics_payload(metrics: dict[str, object]) -> dict[str, object]:
        data = dict(metrics or {})
        if "history" in data:
            data["history_len"] = int(len(data.get("history", []) or []))
            data.pop("history", None)
        return data

    def _write_model_info_state(self) -> None:
        payload = {
            "run_id": self.run_id,
            "bundle_info": {
                "model_name": self.model_name,
                "task_type": self.task_type,
                "seq_len": int(self.seq_len),
                "feature_scheme": self.feature_scheme,
                "feature_count": int(len(self.feature_names)),
                "bundle_dir": str(self.bundle_dir),
            },
            "feature_names": list(self.feature_names),
        }
        try:
            atomic_write_json(self.model_info_state, payload)
        except Exception:
            pass

    def _csv_writer_loop(self) -> None:
        while not self._stop.is_set() or self._csv_queue.qsize() > 0:
            try:
                path, fieldnames, row = self._csv_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                append_csv_row(path, fieldnames, row)
            except Exception:
                self.csv_drop_count += 1
            finally:
                try:
                    self._csv_queue.task_done()
                except Exception:
                    pass

    def wait_for_csv_drain(self, timeout_s: float = 30.0, poll_interval_s: float = 0.05) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() <= deadline:
            try:
                if self._csv_queue.unfinished_tasks == 0 and self._csv_queue.qsize() == 0:
                    return True
            except Exception:
                if self._csv_queue.qsize() == 0:
                    return True
            time.sleep(max(0.005, float(poll_interval_s)))
        return False

    def _enqueue_csv_row(self, path: str | Path, fieldnames: Iterable[str], row: dict[str, object]) -> None:
        try:
            self._csv_queue.put_nowait((Path(path), tuple(fieldnames), dict(row)))
        except Full:
            self.csv_drop_count += 1

    def enqueue(self, observation: FlowObservation) -> None:
        try:
            if observation.inference_enqueue_ts is None:
                observation.inference_enqueue_ts = time.time()
            if self.queue.qsize() >= self.queue_backpressure_drop_threshold:
                low_information = (not bool(observation.has_history)) or (float(observation.packet_delta) <= 0.0 and float(observation.byte_delta) <= 0.0)
                if low_information:
                    self.dropped_rows += 1
                    log_json(self.telemetry_logger, "runtime_drop", reason="queue_backpressure_low_information", queue_depth=self.queue.qsize(), key=observation.key)
                    return
            self.queue.put_nowait(observation)
        except Exception:
            self.dropped_rows += 1
            log_json(self.telemetry_logger, "runtime_drop", reason="queue_put_failed", queue_depth=self.queue.qsize(), key=getattr(observation, "key", ""))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.inference_engine.transform(X)

    def _extract_scores(self, batch: np.ndarray) -> np.ndarray:
        return self.inference_engine.extract_scores(batch)

    def evict_entity(self, entity_key: str) -> None:
        self.sequence_buffers.pop(entity_key, None)
        self.sequence_last_seen.pop(entity_key, None)
        self.rule_engine.reset_entity(entity_key)

    def _evict_idle_sequences(self, now: float) -> None:
        to_delete = [key for key, ts in self.sequence_last_seen.items() if (now - ts) > self.sequence_idle_timeout_s]
        for key in to_delete:
            self.sequence_last_seen.pop(key, None)
            self.sequence_buffers.pop(key, None)
            self.rule_engine.reset_entity(key)

    def _accumulate_latency_breakdown(self, breakdown: dict[str, float]) -> None:
        self.last_latency_breakdown = dict(breakdown)
        for key, value in breakdown.items():
            if not np.isfinite(value):
                continue
            self._latency_breakdown_totals[key] = float(self._latency_breakdown_totals.get(key, 0.0) + float(value))
            self._latency_breakdown_counts[key] = int(self._latency_breakdown_counts.get(key, 0) + 1)

    def _average_latency_breakdown(self) -> dict[str, float]:
        averages: dict[str, float] = {}
        for key, total in self._latency_breakdown_totals.items():
            count = int(self._latency_breakdown_counts.get(key, 0))
            if count > 0:
                averages[key] = float(total / count)
        return averages

    def _compute_latency_breakdown(self, obs: FlowObservation, end_ts: float) -> dict[str, float]:
        breakdown: dict[str, float] = {}
        if obs.poll_request_ts is not None and obs.poll_reply_ts is not None:
            breakdown["switch_latency_ms"] = max(0.0, (float(obs.poll_reply_ts) - float(obs.poll_request_ts)) * 1000.0)
        if obs.poll_reply_ts is not None and obs.feature_enqueue_ts is not None:
            breakdown["feature_build_ms"] = max(0.0, (float(obs.feature_enqueue_ts) - float(obs.poll_reply_ts)) * 1000.0)
        if obs.inference_enqueue_ts is not None and obs.inference_start_ts is not None:
            breakdown["inference_queue_wait_ms"] = max(0.0, (float(obs.inference_start_ts) - float(obs.inference_enqueue_ts)) * 1000.0)
        if obs.inference_start_ts is not None and obs.inference_end_ts is not None:
            breakdown["inference_ms"] = max(0.0, (float(obs.inference_end_ts) - float(obs.inference_start_ts)) * 1000.0)
        if obs.inference_end_ts is not None:
            breakdown["postprocess_ms"] = max(0.0, (float(end_ts) - float(obs.inference_end_ts)) * 1000.0)
        if obs.poll_request_ts is not None:
            breakdown["total_e2e_ms"] = max(0.0, (float(end_ts) - float(obs.poll_request_ts)) * 1000.0)
        elif obs.feature_enqueue_ts is not None:
            breakdown["total_e2e_ms"] = max(0.0, (float(end_ts) - float(obs.feature_enqueue_ts)) * 1000.0)
        return breakdown

    def _compact_controller_metrics(self, payload: dict[str, object]) -> dict[str, object]:
        data = dict(payload or {})
        active_flows = list(data.get("active_flows", []) or [])
        topology = dict(data.get("topology", {}) or {})
        links = list(topology.get("links", []) or [])
        hosts = list(topology.get("hosts", []) or [])
        data["active_flow_count"] = int(data.get("active_flow_count", len(active_flows)))
        data["active_flow_preview_count"] = int(len(active_flows))
        data.pop("active_flows", None)
        topology["host_count"] = int(topology.get("host_count", len(hosts)))
        topology["link_count"] = int(topology.get("link_count", len(links)))
        if len(hosts) > 48:
            topology["hosts_preview"] = hosts[:48]
            topology.pop("hosts", None)
        if len(links) > 60:
            topology["links_preview"] = links[:60]
            topology.pop("links", None)
        data["topology"] = topology
        return data

    def set_controller_metrics(self, payload: dict[str, object]) -> None:
        metrics = dict(payload or {})
        self._external_controller_metrics = metrics
        now = time.time()
        if (now - float(self._last_controller_metrics_write_ts)) >= float(self.controller_metrics_write_interval_s):
            try:
                atomic_write_json(self.controller_metrics_state, metrics)
                self._last_controller_metrics_write_ts = now
            except Exception:
                pass

    def model_info(self) -> dict[str, object]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "manifest": dict(self.manifest),
            "metrics": dict(self.metrics_payload),
            "feature_names": list(self.feature_names),
            "model_name": self.model_name,
            "task_type": self.task_type,
            "seq_len": int(self.seq_len),
            "threshold": float(self.threshold),
            "score_direction": self.score_direction,
        }

    def _purge_old_alerts(self, now: float) -> None:
        while self.recent_alerts and (now - float(self.recent_alerts[0].get("timestamp", 0.0))) > self.alert_hold_seconds:
            self.recent_alerts.popleft()
        while self.recent_scores and (now - float(self.recent_scores[0][0])) > self.alert_hold_seconds:
            self.recent_scores.popleft()

    def _is_runtime_idle(self, now: float) -> bool:
        if self.last_observation_ts is None:
            return True
        return (now - self.last_observation_ts) > self.idle_reset_seconds

    def _flush_state(self, status: str | None = None, reason: str = "", force: bool = False, phase: str | None = None) -> None:
        now = time.time()
        self._purge_old_alerts(now)
        self._evict_idle_sequences(now)
        self.rule_engine.on_idle(now)
        runtime_idle = self._is_runtime_idle(now)

        if runtime_idle:
            self.recent_alerts.clear()
            self.last_decision_status = "NORMAL"
            self.last_decision_reason = ""
            self.last_decision_source = "runtime"
            self.last_runtime_phase = "idle"
            display_score = 0.0
        else:
            if status is not None:
                self.last_decision_status = _normalize_status(status)
                self.last_decision_reason = reason
                self.last_decision_ts = now
            if phase is not None:
                self.last_runtime_phase = phase
            recent_peak_score = max((entry[1] for entry in self.recent_scores), default=float(self.last_score or 0.0))
            display_score = float(recent_peak_score)

        recent_peak_score = max((entry[1] for entry in self.recent_scores), default=float(self.last_score or 0.0))
        recent_status = "NORMAL"
        recent_peak_category = ""
        for _ts, _score, _status, _category in sorted(self.recent_scores, key=lambda row: (_STATUS_RANK.get(row[2], 0), row[1])):
            if _STATUS_RANK.get(_status, 0) >= _STATUS_RANK.get(recent_status, 0):
                recent_status = _status
                if _category:
                    recent_peak_category = _category
        recent_signal_count = sum(1 for _ts, _score, _status, _category in self.recent_scores if _normalize_status(_status) in {"ATTACK", "SUSPECT"})
        model_recent_hit = False
        if self.score_direction == "lower_is_attack":
            model_recent_hit = bool(recent_peak_score and recent_peak_score <= self.threshold)
        else:
            model_recent_hit = bool(recent_peak_score and recent_peak_score >= self.threshold)
        warming_up = (not runtime_idle) and (not self.has_scored_observation)

        effective_status = _normalize_status(self.last_decision_status)
        effective_reason = self.last_decision_reason
        if runtime_idle:
            effective_status = "NORMAL"
            effective_reason = ""
            warming_up = False
        else:
            if warming_up and recent_status == "NORMAL":
                effective_status = "NORMAL"
                effective_reason = ""
            else:
                if _STATUS_RANK.get(recent_status, 0) > _STATUS_RANK.get(effective_status, 0):
                    effective_status = recent_status
                    if recent_peak_category:
                        effective_reason = f"holding recent {recent_peak_category} evidence"
                    elif model_recent_hit:
                        effective_reason = "holding recent model evidence above threshold"
                elif effective_status == "NORMAL" and model_recent_hit:
                    effective_status = "SUSPECT"
                    effective_reason = "recent model score above threshold; confirming"

        if effective_status in {"ATTACK", "SUSPECT"} and self.last_decision_ts is not None:
            if (now - float(self.last_decision_ts)) > self.alert_hold_seconds and recent_signal_count == 0 and not self.recent_alerts:
                effective_status = "NORMAL"
                effective_reason = "telemetry stabilized below alert hold window"

        avg_latency_ms = (self.total_batch_latency_ms / max(self.total_rows_read, 1)) if self.total_rows_read else 0.0
        throughput = self.total_rows_read / max(time.time() - self.start_time, 1e-6)
        active_signal_count = max(len(self.recent_alerts), recent_signal_count)
        payload = {
            "run_id": self.run_id,
            "server_timestamp": _utc_now_iso(),
            "model_name": self.model_name,
            "task_type": self.task_type,
            "feature_scheme": self.feature_scheme,
            "seq_len": self.seq_len,
            "poll_interval_s": float(self.poll_interval_s),
            "dashboard_timeline_interval_s": float(self.dashboard_timeline_interval_s),
            "poll_timestamp": pd.Timestamp.utcnow().isoformat(),
            "status": effective_status,
            "phase": self.last_runtime_phase,
            "reason": effective_reason,
            "decision_source": self.last_decision_source,
            "warming_up": bool(warming_up),
            "has_scored_observation": bool(self.has_scored_observation),
            "threshold": float(self.threshold),
            "score_direction": self.score_direction,
            "max_score": float(self.last_score) if self.last_score is not None else 0.0,
            "recent_peak_score": float(recent_peak_score),
            "display_score": float(display_score),
            "recent_peak_category": recent_peak_category,
            "recent_alert_count": len(self.recent_alerts),
            "active_signal_count": int(active_signal_count),
            "recent_alerts": list(self.recent_alerts)[-10:],
            "total_rows_read": self.total_rows_read,
            "total_inferences": self.total_inferences,
            "total_scored_rows": self.total_scored_rows,
            "total_batches": self.total_batches,
            "dropped_rows": self.dropped_rows,
            "csv_drop_count": self.csv_drop_count,
            "stream_drop_count": int(self.stream_drop_count),
            "queue_depth": self.queue.qsize(),
            "csv_queue_depth": self._csv_queue.qsize(),
            "sequence_buffer_count": len(self.sequence_buffers),
            "feature_names": self.feature_names,
            "latency_ms": float(self.last_latency_ms),
            "avg_latency_ms": float(avg_latency_ms),
            "throughput_obs_s": float(throughput),
            "last_batch_size": int(self.last_batch_size),
            "last_inference_batch_size": int(self.last_inference_batch_size),
            "inference_batch_max": int(self.inference_batch_max),
            "uptime_s": round(time.time() - self.start_time, 2),
            "latency_breakdown": dict(self.last_latency_breakdown),
            "avg_latency_breakdown": self._average_latency_breakdown(),
            "polling_summary": summarize_polling_metrics((self._external_controller_metrics or {}).get("polling_stats", {})),
            "recommended_poll_interval_s": recommended_poll_interval(
                current_interval_s=float(self.poll_interval_s),
                min_interval_s=max(0.25, float(self.poll_interval_s) * 0.5),
                max_interval_s=max(float(self.poll_interval_s), float(self.poll_interval_s) * 2.5),
                runtime_queue_depth=int(self.queue.qsize()),
                raw_queue_depth=int((self._external_controller_metrics or {}).get("raw_stats_queue_depth", 0)),
                avg_reply_delay_ms=float(summarize_polling_metrics((self._external_controller_metrics or {}).get("polling_stats", {})).get("avg_reply_delay_ms", 0.0)),
                timeout_total=int(summarize_polling_metrics((self._external_controller_metrics or {}).get("polling_stats", {})).get("timeout_total", 0)),
                throughput_obs_s=float(throughput),
                queue_threshold=int(self.queue_backpressure_drop_threshold),
                raw_threshold=int((self._external_controller_metrics or {}).get("raw_stats_queue_maxsize", 4096)),
                reply_delay_high_ms=max(500.0, float(self.poll_interval_s) * 1200.0),
            ),
            "bundle_info": {
                "model_name": self.model_name,
                "task_type": self.task_type,
                "seq_len": int(self.seq_len),
                "feature_scheme": self.feature_scheme,
                "feature_count": int(len(self.feature_names)),
                "bundle_dir": str(self.bundle_dir),
            },
        }
        if self._external_controller_metrics:
            compact_controller_metrics = self._compact_controller_metrics(self._external_controller_metrics)
            payload["controller_summary"] = {
                "raw_stats_queue_depth": int(compact_controller_metrics.get("raw_stats_queue_depth", 0)),
                "feature_state_count": int(compact_controller_metrics.get("feature_state_count", 0)),
                "pressure_state": str(compact_controller_metrics.get("pressure_state", "normal")),
                "recommended_poll_interval_s": float(compact_controller_metrics.get("recommended_poll_interval_s", self.poll_interval_s)),
                "active_flow_count": int(compact_controller_metrics.get("active_flow_count", 0)),
                "active_flow_preview_count": int(compact_controller_metrics.get("active_flow_preview_count", 0)),
                "polling_summary": dict(compact_controller_metrics.get("polling_summary", {})),
            }
        self.state_payload = payload
        if force or effective_status in {"ATTACK", "SUSPECT"}:
            log_json(self.telemetry_logger, "runtime_state", status=effective_status, phase=self.last_runtime_phase, queue_depth=int(self.queue.qsize()), display_score=round(float(display_score), 6), active_signals=int(active_signal_count))
        stream_sig = f"{effective_status}|{self.last_runtime_phase}|{round(display_score, 5)}|{active_signal_count}|{int(self.queue.qsize())}|{effective_reason}"
        if self.dashboard_stream_enabled and ((now - self._last_stream_state_ts) >= self.dashboard_stream_state_interval_s or stream_sig != self._last_stream_state_sig):
            self.dashboard_publisher.publish_state(self.run_id, payload)
            self.stream_drop_count = self.dashboard_publisher.drop_count
            self._last_stream_state_ts = now
            self._last_stream_state_sig = stream_sig
        if force or (now - self.last_state_write_ts) >= self.state_write_interval_s:
            atomic_write_json(self.dashboard_state, payload)
            self.last_state_write_ts = now

    def _append_alert(self, obs: FlowObservation, score: float, decision: RuleDecision) -> None:
        decision_status = _normalize_status(decision.status)
        alert = {
            "timestamp": time.time(),
            "poll_timestamp": _utc_now_iso(),
            "entity_key": obs.key,
            "emit_key": decision.emit_key or obs.key,
            "anomaly_score": float(score),
            "src_ip": obs.src_ip,
            "dst_ip": obs.dst_ip,
            "src_port": int(obs.src_port),
            "dst_port": int(obs.dst_port),
            "protocol": int(obs.protocol),
            "packet_rate": float(obs.packet_rate),
            "packet_delta": float(obs.packet_delta),
            "byte_rate": float(obs.byte_rate),
            "byte_delta": float(obs.byte_delta),
            "reason": decision.reason,
            "status": decision_status,
            "severity": decision.severity,
            "block": bool(decision.block),
            "hit_count": int(decision.hit_count),
            "decision_source": decision.source,
            "category": decision.category,
            "support_sources": ",".join(decision.support_sources),
            "support_reasons": " | ".join(decision.support_reasons),
            "threshold": float(self.threshold),
            "model_hit": bool(score >= self.threshold) if self.score_direction != "lower_is_attack" else bool(score <= self.threshold),
        }
        self.recent_alerts.append(alert)
        if self.dashboard_stream_enabled:
            self.dashboard_publisher.publish_alert(self.run_id, alert)
            self.stream_drop_count = self.dashboard_publisher.drop_count
        keys = ["poll_timestamp", "entity_key", "emit_key", "anomaly_score", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_rate", "packet_delta", "byte_rate", "byte_delta", "status", "reason", "severity", "block", "hit_count", "decision_source", "category", "support_sources", "support_reasons", "threshold", "model_hit"]
        self._enqueue_csv_row(
            self.alerts_csv,
            fieldnames=keys,
            row={k: alert.get(k, "") for k in keys},
        )

    def _log_score(self, obs: FlowObservation, score: float, decision: RuleDecision, phase: str) -> None:
        keys = [
            "timestamp", "entity_key", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "score", "threshold", "status", "phase", "reason", "decision_source", "severity",
            "packet_rate", "byte_rate", "packet_delta", "byte_delta",
            "label", "source_file", "support_sources", "support_reasons",
        ]
        row = {
            "timestamp": _utc_now_iso(),
            "entity_key": obs.key,
            "src_ip": obs.src_ip,
            "dst_ip": obs.dst_ip,
            "src_port": int(obs.src_port),
            "dst_port": int(obs.dst_port),
            "protocol": int(obs.protocol),
            "score": float(score),
            "threshold": float(self.threshold),
            "status": _normalize_status(decision.status),
            "phase": phase,
            "reason": decision.reason,
            "decision_source": decision.source,
            "category": decision.category,
            "severity": decision.severity,
            "packet_rate": float(obs.packet_rate),
            "byte_rate": float(obs.byte_rate),
            "packet_delta": float(obs.packet_delta),
            "byte_delta": float(obs.byte_delta),
            "label": obs.label or "",
            "source_file": obs.source_file or "",
            "support_sources": ",".join(decision.support_sources),
            "support_reasons": " | ".join(decision.support_reasons),
        }
        self._enqueue_csv_row(self.scores_csv, fieldnames=keys, row=row)

    def _log_telemetry_if_needed(self, obs: FlowObservation) -> None:
        if self.telemetry_log_every_n <= 0:
            return
        if (self.total_rows_read % self.telemetry_log_every_n) != 0:
            return
        if self.queue.qsize() >= self.telemetry_skip_queue_threshold:
            return
        telemetry_keys = [
            "timestamp", "entity_key", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "packet_rate", "byte_rate", "packet_delta", "byte_delta", "packet_rate_delta", "byte_rate_delta",
            "has_history", *self.feature_names,
        ]
        self._enqueue_csv_row(
            self.telemetry_csv,
            fieldnames=telemetry_keys,
            row={
                "timestamp": _utc_now_iso(),
                "entity_key": obs.key,
                "src_ip": obs.src_ip,
                "dst_ip": obs.dst_ip,
                "src_port": obs.src_port,
                "dst_port": obs.dst_port,
                "protocol": obs.protocol,
                "packet_rate": obs.packet_rate,
                "byte_rate": obs.byte_rate,
                "packet_delta": obs.packet_delta,
                "byte_delta": obs.byte_delta,
                "packet_rate_delta": obs.packet_rate_delta,
                "byte_rate_delta": obs.byte_rate_delta,
                "has_history": int(obs.has_history),
                **{name: float(val) for name, val in zip(self.feature_names, obs.feature_vector)},
            },
        )

    def process_observation(self, obs: FlowObservation) -> dict[str, object]:
        results = self.process_observations([obs])
        return results[0] if results else {"score": None, "status": "NORMAL", "reason": "no observation processed", "latency_ms": 0.0, "block": False}

    def process_observations(self, observations: list[FlowObservation]) -> list[dict[str, object]]:
        if not observations:
            return []
        t0 = time.perf_counter()
        inference_start_wall = time.time()
        self.last_observation_ts = float(observations[-1].timestamp)
        self.total_rows_read += len(observations)
        self.total_batches += 1
        self.last_batch_size = len(observations)

        valid_items: list[tuple[FlowObservation, np.ndarray]] = []
        results: list[dict[str, object]] = []
        decisions_by_obs_idx: dict[int, tuple[RuleDecision, float | None, str]] = {}
        phase_by_obs_idx: dict[int, str] = {}

        for idx, obs in enumerate(observations):
            raw = np.asarray(obs.feature_vector, dtype=np.float32).reshape(-1)
            if raw.shape[-1] != self.n_features:
                decision = RuleDecision(status="NORMAL", reason="")
                decisions_by_obs_idx[idx] = (decision, None, "normalizing")
                phase_by_obs_idx[idx] = "feature_mismatch"
                continue
            valid_items.append((obs, raw))
            self._log_telemetry_if_needed(obs)

        if valid_items:
            transformed_batch = self._transform(np.asarray([raw for _, raw in valid_items], dtype=np.float32))
            ready_obs: list[FlowObservation] = []
            ready_seqs: list[np.ndarray] = []
            ready_indices: list[int] = []
            valid_iter_idx = 0
            for idx, obs in enumerate(observations):
                if idx in decisions_by_obs_idx:
                    continue
                transformed = transformed_batch[valid_iter_idx]
                valid_iter_idx += 1
                buf = self.sequence_buffers.setdefault(obs.key, deque(maxlen=self.seq_len))
                buf.append(transformed)
                self.sequence_last_seen[obs.key] = float(obs.timestamp)
                self._evict_idle_sequences(float(obs.timestamp))
                if len(buf) >= self.seq_len:
                    ready_obs.append(obs)
                    ready_indices.append(idx)
                    ready_seqs.append(np.asarray(list(buf)[-self.seq_len:], dtype=np.float32))
                else:
                    decision = self.rule_engine.evaluate(obs, None, self.threshold, self.score_direction)
                    phase = f"warmup_{len(buf)}/{self.seq_len}"
                    decisions_by_obs_idx[idx] = (decision, None, phase)
                    phase_by_obs_idx[idx] = phase

            if ready_seqs:
                seq_batch = np.asarray(ready_seqs, dtype=np.float32)
                for obs in ready_obs:
                    obs.inference_start_ts = inference_start_wall
                scores = self._extract_scores(seq_batch)
                inference_end_wall = time.time()
                for obs in ready_obs:
                    obs.inference_end_ts = inference_end_wall
                self.last_inference_batch_size = int(len(scores))
                self.total_inferences += 1
                self.total_scored_rows += int(len(scores))
                self.last_score = float(np.max(scores)) if len(scores) else self.last_score
                for idx, obs, score in zip(ready_indices, ready_obs, scores):
                    decision = self.rule_engine.evaluate(obs, float(score), self.threshold, self.score_direction)
                    phase = "scored"
                    decisions_by_obs_idx[idx] = (decision, float(score), phase)
                    phase_by_obs_idx[idx] = phase
                    self.has_scored_observation = True
                    self.last_scored_ts = time.time()
                    self.recent_scores.append((time.time(), float(score), _normalize_status(decision.status), decision.category))
                    self._log_score(obs, float(score), decision, phase)
            else:
                self.last_inference_batch_size = 0
        else:
            self.last_inference_batch_size = 0

        # support-only decisions should still emit alerts during warmup, otherwise short-lived
        # Probe/BFA flows never reach the dashboard even when rules are firing.
        for idx, obs in enumerate(observations):
            decision, score, phase = decisions_by_obs_idx.get(idx, (RuleDecision(status="NORMAL", reason=""), None, "idle"))
            if score is None and _normalize_status(decision.status) == "ATTACK":
                self.recent_scores.append((time.time(), float(self.threshold * 0.65), _normalize_status(decision.status), decision.category))
            emit_key = decision.emit_key or obs.key
            if _normalize_status(decision.status) == "ATTACK" and self.rule_engine.should_emit_alert(emit_key):
                self._append_alert(obs, float(score) if score is not None else 0.0, decision)

        dominant_idx = 0
        dominant_decision = RuleDecision(status="NORMAL", reason="")
        dominant_score: float | None = None
        dominant_phase = "idle"
        for idx in range(len(observations)):
            decision, score, phase = decisions_by_obs_idx.get(idx, (RuleDecision(status="NORMAL", reason=""), None, "idle"))
            status = _normalize_status(decision.status)
            decision = RuleDecision(
                status=status,
                reason=decision.reason,
                block=decision.block,
                severity=decision.severity,
                hit_increment=decision.hit_increment,
                hit_count=decision.hit_count,
                emit_key=decision.emit_key,
                source=decision.source,
                category=decision.category,
                support_reasons=decision.support_reasons,
                support_sources=decision.support_sources,
            )
            decisions_by_obs_idx[idx] = (decision, score, phase)
            if _decision_sort_key(decision, score) >= _decision_sort_key(dominant_decision, dominant_score):
                dominant_idx = idx
                dominant_decision = decision
                dominant_score = score
                dominant_phase = phase

        self.last_decision_source = dominant_decision.source
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_latency_ms = batch_latency_ms / max(len(observations), 1)
        self.total_batch_latency_ms += batch_latency_ms
        end_ts = time.time()
        aggregate_breakdown: dict[str, list[float]] = {}
        for obs in observations:
            obs.alert_emit_ts = end_ts
            breakdown = self._compute_latency_breakdown(obs, end_ts)
            for key, value in breakdown.items():
                aggregate_breakdown.setdefault(key, []).append(float(value))
        averaged_breakdown = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in aggregate_breakdown.items()
            if values
        }
        if averaged_breakdown:
            self._accumulate_latency_breakdown(averaged_breakdown)
        self._flush_state(status=dominant_decision.status, reason=dominant_decision.reason, phase=dominant_phase)

        for idx, obs in enumerate(observations):
            decision, score, _phase = decisions_by_obs_idx[idx]
            breakdown = self._compute_latency_breakdown(obs, end_ts)
            results.append({
                "score": score,
                "status": _normalize_status(decision.status),
                "reason": decision.reason,
                "latency_ms": self.last_latency_ms,
                "latency_breakdown": breakdown,
                "block": decision.block,
                "decision_source": decision.source,
            })
        return results

    def stop(self, timeout_s: float = 5.0) -> bool:
        self._stop.set()
        drained = self.wait_for_csv_drain(timeout_s=max(0.25, timeout_s))
        return bool(drained)

    def run_forever(self) -> None:
        batch_wait_s = max(0.0, self.inference_batch_wait_ms / 1000.0)
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=1.0)
            except Empty:
                self._flush_state(status="NORMAL", reason="", force=True, phase="idle")
                if self._stop.is_set():
                    break
                continue
            observations = [first]
            deadline = time.perf_counter() + batch_wait_s
            while len(observations) < self.inference_batch_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 and (self.queue.empty() or len(observations) >= self.inference_batch_max):
                    break
                timeout = max(0.0, remaining)
                try:
                    observations.append(self.queue.get(timeout=timeout if timeout > 0 else 0.0))
                except Empty:
                    break
                if self.queue.empty() and time.perf_counter() >= deadline:
                    break
            self.process_observations(observations)

    def queue_depth_now(self) -> int:
        return self.queue.qsize()

    def current_state(self) -> dict[str, object]:
        return dict(self.state_payload)
