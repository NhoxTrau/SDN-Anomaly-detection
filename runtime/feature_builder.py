from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..train_v2.common import (
    DEFAULT_FEATURE_SCHEME,
    DEFAULT_POLL_INTERVAL_S,
    FlowObservation,
    FlowTelemetryState,
    feature_names_for_scheme,
)


@dataclass
class FlowMatchInfo:
    dpid: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    in_port: int = -1
    key_mode: str = "flow"

    def key(self) -> str:
        base = f"dp{self.dpid}|src={self.src_ip}|dst={self.dst_ip}|proto={self.protocol}"
        if self.key_mode == "service":
            return f"{base}|dport={self.dst_port}"
        return f"{base}|sport={self.src_port}|dport={self.dst_port}"


class OpenFlowFeatureBuilder:
    def __init__(
        self,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        feature_names: Iterable[str] | None = None,
        key_mode: str = "flow",
    ) -> None:
        self.poll_interval_s = float(poll_interval_s)
        self.feature_names = list(feature_names or feature_names_for_scheme(DEFAULT_FEATURE_SCHEME))
        self.key_mode = str(key_mode or "flow").strip().lower()
        if self.key_mode not in {"flow", "service"}:
            raise ValueError(f"Unsupported key_mode={self.key_mode!r}. Use 'flow' or 'service'.")
        self.states: dict[str, FlowTelemetryState] = {}

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _lookup_match_value(self, match: Any, *field_names: str) -> Any:
        if hasattr(match, "get"):
            for field_name in field_names:
                value = match.get(field_name)
                if value not in (None, ""):
                    return value
        if not hasattr(match, "to_jsondict"):
            return None
        try:
            fields = match.to_jsondict().get("OFPMatch", {}).get("oxm_fields", [])
        except Exception:
            return None
        lookup: dict[str, Any] = {}
        for item in fields:
            if not isinstance(item, dict):
                continue
            candidate = item
            if len(item) == 1:
                nested = next(iter(item.values()))
                if isinstance(nested, dict):
                    candidate = nested
            field_name = candidate.get("field")
            if field_name is None:
                continue
            lookup[str(field_name).strip().lower()] = candidate.get("value")
        for field_name in field_names:
            value = lookup.get(str(field_name).strip().lower())
            if value not in (None, ""):
                return value
        return None

    def build_key_from_match(self, dpid: int, match: Any) -> FlowMatchInfo:
        src_ip = self._normalize_text(self._lookup_match_value(match, "ipv4_src"))
        dst_ip = self._normalize_text(self._lookup_match_value(match, "ipv4_dst"))
        src_port = self._safe_int(self._lookup_match_value(match, "tcp_src", "udp_src"))
        dst_port = self._safe_int(self._lookup_match_value(match, "tcp_dst", "udp_dst"))
        protocol = self._safe_int(self._lookup_match_value(match, "ip_proto"))
        in_port = self._safe_int(self._lookup_match_value(match, "in_port"))
        return FlowMatchInfo(
            dpid=dpid,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            in_port=in_port,
            key_mode=self.key_mode,
        )

    def get_or_create_state(self, info: FlowMatchInfo, first_seen_ts: float, current_ts: float) -> FlowTelemetryState:
        key = info.key()
        state = self.states.get(key)
        if state is None:
            state = FlowTelemetryState(
                key=key,
                dpid=info.dpid,
                src_ip=info.src_ip,
                dst_ip=info.dst_ip,
                src_port=info.src_port,
                dst_port=info.dst_port,
                protocol=info.protocol,
                first_seen_ts=first_seen_ts,
                last_seen_ts=current_ts,
                last_poll_ts=current_ts,
            )
            self.states[key] = state
        return state

    def state_count(self) -> int:
        return len(self.states)

    def get_state(self, key: str) -> FlowTelemetryState | None:
        return self.states.get(key)

    def should_emit_observation(
        self,
        info: FlowMatchInfo,
        packet_count: float,
        byte_count: float,
        *,
        min_packet_delta: float = 1.0,
        min_byte_delta: float = 1.0,
        force_emit_after_s: float = 15.0,
        now_ts: float | None = None,
    ) -> bool:
        state = self.states.get(info.key())
        if state is None:
            return float(packet_count) > 0.0 or float(byte_count) > 0.0
        packet_delta = float(packet_count) - float(state.last_packet_count)
        byte_delta = float(byte_count) - float(state.last_byte_count)
        if packet_delta < 0.0 or byte_delta < 0.0:
            return True
        if packet_delta >= float(min_packet_delta) or byte_delta >= float(min_byte_delta):
            return True
        if now_ts is not None and (float(now_ts) - float(state.last_seen_ts)) >= float(force_emit_after_s):
            return True
        return False

    def evict_state_by_key(self, key: str) -> None:
        self.states.pop(key, None)

    def evict_state_by_match(self, dpid: int, match: Any) -> str | None:
        try:
            key = self.build_key_from_match(dpid, match).key()
        except Exception:
            return None
        self.evict_state_by_key(key)
        return key

    def evict_idle_states(self, current_ts: float, idle_timeout_s: float = 120.0) -> int:
        to_delete = [key for key, state in self.states.items() if (current_ts - state.last_seen_ts) > idle_timeout_s]
        for key in to_delete:
            del self.states[key]
        return len(to_delete)

    def make_observation_from_info(
        self,
        info: FlowMatchInfo,
        packet_count: float,
        byte_count: float,
        duration_s: float,
        now_ts: float,
        poll_request_ts: float | None = None,
        poll_reply_ts: float | None = None,
        feature_dequeue_ts: float | None = None,
        poll_cycle_id: int = 0,
        reply_part_count: int = 0,
        fwd_packet_count: float = -1.0,
        fwd_byte_count: float = -1.0,
    ) -> FlowObservation:
        del fwd_packet_count, fwd_byte_count
        if not info.src_ip or not info.dst_ip:
            raise ValueError("missing ipv4_src/ipv4_dst in match")
        state = self.get_or_create_state(info, first_seen_ts=now_ts, current_ts=now_ts)
        feature_vector, stats = state.update_from_stats(
            packet_count,
            byte_count,
            duration_s,
            poll_ts=now_ts,
            poll_interval_s=self.poll_interval_s,
            feature_names=self.feature_names,
        )
        return FlowObservation(
            key=info.key(),
            dpid=info.dpid,
            timestamp=now_ts,
            src_ip=info.src_ip,
            dst_ip=info.dst_ip,
            src_port=info.src_port,
            dst_port=info.dst_port,
            protocol=info.protocol,
            packet_count=float(stats["packet_count"]),
            byte_count=float(stats["byte_count"]),
            duration_s=float(stats["flow_duration_s"]),
            packet_rate=float(stats["packet_rate"]),
            byte_rate=float(stats["byte_rate"]),
            avg_packet_size=float(stats["avg_packet_size"]),
            packet_delta=float(stats["packet_delta"]),
            byte_delta=float(stats["byte_delta"]),
            packet_rate_delta=float(stats["packet_rate_delta"]),
            byte_rate_delta=float(stats["byte_rate_delta"]),
            has_history=bool(stats["has_history"]),
            feature_vector=feature_vector,
            poll_request_ts=poll_request_ts,
            poll_reply_ts=poll_reply_ts,
            feature_enqueue_ts=float(now_ts),
            feature_dequeue_ts=feature_dequeue_ts,
            poll_cycle_id=int(poll_cycle_id),
            reply_part_count=int(reply_part_count),
        )

    def make_observation(
        self,
        dpid: int,
        match: Any,
        packet_count: float,
        byte_count: float,
        duration_s: float,
        now_ts: float,
        poll_request_ts: float | None = None,
        poll_reply_ts: float | None = None,
        feature_dequeue_ts: float | None = None,
        poll_cycle_id: int = 0,
        reply_part_count: int = 0,
        fwd_packet_count: float = -1.0,
        fwd_byte_count: float = -1.0,
    ) -> FlowObservation:
        info = self.build_key_from_match(dpid, match)
        return self.make_observation_from_info(
            info=info,
            packet_count=packet_count,
            byte_count=byte_count,
            duration_s=duration_s,
            now_ts=now_ts,
            poll_request_ts=poll_request_ts,
            poll_reply_ts=poll_reply_ts,
            feature_dequeue_ts=feature_dequeue_ts,
            poll_cycle_id=poll_cycle_id,
            reply_part_count=reply_part_count,
            fwd_packet_count=fwd_packet_count,
            fwd_byte_count=fwd_byte_count,
        )
