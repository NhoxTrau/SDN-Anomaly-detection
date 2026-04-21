from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..train_v2.common import FlowObservation

_NON_PORT_PROTOCOLS = {1}


@dataclass
class RuleDecision:
    status: str
    reason: str
    block: bool = False
    severity: str = "low"
    hit_increment: int = 0
    hit_count: int = 0
    emit_key: str = ""
    source: str = "runtime"
    category: str = "Normal"
    support_reasons: tuple[str, ...] = field(default_factory=tuple)
    support_sources: tuple[str, ...] = field(default_factory=tuple)


class RuleEngine:
    def __init__(self, policy: dict | None = None) -> None:
        policy = policy or {}
        self.min_alert_hits = int(policy.get("min_alert_hits", 2))
        self.alert_cooldown_seconds = float(policy.get("alert_cooldown_seconds", 5.0))
        self.alert_hold_seconds = float(policy.get("alert_hold_seconds", 8.0))
        self.score_margin = float(policy.get("score_margin", 0.5))
        self.rule_assist_ratio = float(policy.get("rule_assist_ratio", 0.92))
        self.attack_display_floor = float(policy.get("attack_display_floor", 0.20))
        self.scan_window_s = float(policy.get("scan_window_s", 6.0))
        self.scan_unique_ports = int(policy.get("scan_unique_ports", 12))
        self.scan_attack_unique_ports = int(policy.get("scan_attack_unique_ports", 40))
        self.scan_emit_cooldown_s = float(policy.get("scan_emit_cooldown_s", max(self.scan_window_s, self.alert_cooldown_seconds)))
        self.hard_packet_rate = float(policy.get("hard_packet_rate", 4000.0))
        self.hard_byte_rate = float(policy.get("hard_byte_rate", 4_000_000.0))
        self.hard_packet_delta = float(policy.get("hard_packet_delta", 2500.0))
        self.hard_byte_delta = float(policy.get("hard_byte_delta", 2_000_000.0))
        self.volumetric_min_hits = int(policy.get("volumetric_min_hits", 2))
        self.baseline_window_s = float(policy.get("baseline_window_s", 30.0))
        self.baseline_min_samples = int(policy.get("baseline_min_samples", 4))
        self.rate_multiplier_warn = float(policy.get("rate_multiplier_warn", 2.5))
        self.rate_multiplier_attack = float(policy.get("rate_multiplier_attack", 4.0))
        self.entity_idle_timeout_s = float(policy.get("entity_idle_timeout_s", max(self.alert_hold_seconds, 20.0)))
        self.disable_scan_rule = bool(policy.get("disable_scan_rule", False))
        self.rules_only_attack_enabled = bool(policy.get("rules_only_attack_enabled", False))
        self.emergency_rules_enabled = bool(policy.get("emergency_rules_enabled", False))
        self.udp_flood_packet_rate = float(policy.get("udp_flood_packet_rate", 3000.0))
        self.udp_flood_byte_rate = float(policy.get("udp_flood_byte_rate", 2_000_000.0))
        self.icmp_flood_packet_rate = float(policy.get("icmp_flood_packet_rate", 2000.0))
        self.tcp_small_pkt_attack_rate = float(policy.get("tcp_small_pkt_attack_rate", 2500.0))
        self.small_pkt_avg_size_max = float(policy.get("small_pkt_avg_size_max", 120.0))
        self.allowlisted_services = {
            str(ip): {int(p) for p in ports}
            for ip, ports in (policy.get("allowlisted_services") or {}).items()
        }
        self.bfa_window_s = float(policy.get("bfa_window_s", 8.0))
        self.bfa_attempt_threshold = int(policy.get("bfa_attempt_threshold", 10))
        self.bfa_attack_threshold = int(policy.get("bfa_attack_threshold", 20))
        self.bfa_ports = {int(p) for p in policy.get("bfa_ports", [21, 22, 23])}

        self.alert_hits: dict[str, int] = defaultdict(int)
        self.last_alert_ts_by_key: dict[str, float] = {}
        self.entity_last_seen: dict[str, float] = {}
        self.scan_memory: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self.scan_last_alert_ts: dict[str, float] = {}
        self.bfa_memory: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self.service_rate_history: dict[str, deque[tuple[float, float, float]]] = defaultdict(deque)
        self.service_violation_hits: dict[str, int] = defaultdict(int)
        self.volumetric_hits: dict[str, int] = defaultdict(int)

    def _service_key(self, obs: FlowObservation) -> str:
        return f"{obs.dst_ip}:{obs.dst_port}:{obs.protocol}"

    def _entity_scope(self, obs: FlowObservation) -> str:
        return f"{obs.src_ip}->{obs.dst_ip}:{obs.dst_port}:{obs.protocol}"

    def _scan_scope(self, obs: FlowObservation) -> str:
        return f"{obs.src_ip}->{obs.dst_ip}:{obs.protocol}"

    @staticmethod
    def _is_ephemeral(port: int) -> bool:
        return port >= 1024

    def _purge_scan(self, scope_key: str, now: float) -> None:
        dq = self.scan_memory[scope_key]
        while dq and now - dq[0][0] > self.scan_window_s:
            dq.popleft()
        if not dq:
            self.scan_memory.pop(scope_key, None)

    def _purge_service_history(self, service_key: str, now: float) -> None:
        dq = self.service_rate_history[service_key]
        while dq and now - dq[0][0] > self.baseline_window_s:
            dq.popleft()
        if not dq:
            self.service_rate_history.pop(service_key, None)

    def _baseline(self, service_key: str, now: float) -> tuple[float, float] | None:
        self._purge_service_history(service_key, now)
        dq = self.service_rate_history.get(service_key)
        if not dq or len(dq) < self.baseline_min_samples:
            return None
        pkt_rates = [pkt for _, pkt, _ in dq]
        byte_rates = [byte for _, _, byte in dq]
        return max(sum(pkt_rates) / len(pkt_rates), 1.0), max(sum(byte_rates) / len(byte_rates), 1.0)

    def _update_baseline(self, obs: FlowObservation, now: float) -> None:
        if not obs.has_history:
            return
        service_key = self._service_key(obs)
        dq = self.service_rate_history[service_key]
        dq.append((now, float(obs.packet_rate), float(obs.byte_rate)))
        self._purge_service_history(service_key, now)

    def _is_probable_response_flow(self, obs: FlowObservation) -> bool:
        return obs.src_port > 0 and obs.src_port <= 1024 and obs.dst_port >= 1024

    def _has_activity(self, obs: FlowObservation) -> bool:
        return bool(obs.has_history) and (float(obs.packet_delta) > 0.0 or float(obs.byte_delta) > 0.0)

    def _has_observed_packets(self, obs: FlowObservation) -> bool:
        if self._has_activity(obs):
            return True
        return float(obs.packet_count) > 0.0 or float(obs.byte_count) > 0.0

    def _near_threshold(self, score: float, threshold: float, score_direction: str) -> bool:
        if score_direction == "lower_is_attack":
            return score <= threshold / max(self.rule_assist_ratio, 1e-6)
        return score >= threshold * self.rule_assist_ratio

    def cleanup(self, now: float) -> None:
        stale_keys = [key for key, ts in self.entity_last_seen.items() if (now - ts) > self.entity_idle_timeout_s]
        for key in stale_keys:
            self.entity_last_seen.pop(key, None)
            self.alert_hits.pop(key, None)
            self.service_violation_hits.pop(key, None)
            self.volumetric_hits.pop(key, None)
            self.last_alert_ts_by_key.pop(key, None)
        for service_key in list(self.service_rate_history.keys()):
            self._purge_service_history(service_key, now)
        for scope_key in list(self.scan_memory.keys()):
            self._purge_scan(scope_key, now)
            if not self.scan_memory.get(scope_key):
                self.scan_last_alert_ts.pop(scope_key, None)
        for scope_key, dq in list(self.bfa_memory.items()):
            while dq and now - dq[0][0] > self.bfa_window_s:
                dq.popleft()
            if not dq:
                self.bfa_memory.pop(scope_key, None)

    def on_idle(self, now: float) -> None:
        self.cleanup(now)

    def reset_entity(self, entity_key: str) -> None:
        self.entity_last_seen.pop(entity_key, None)
        self.alert_hits.pop(entity_key, None)
        self.service_violation_hits.pop(entity_key, None)
        self.volumetric_hits.pop(entity_key, None)
        self.last_alert_ts_by_key.pop(entity_key, None)

    def _scan_support(self, obs: FlowObservation, now: float) -> RuleDecision | None:
        if self.disable_scan_rule or obs.protocol in _NON_PORT_PROTOCOLS:
            return None
        if not obs.src_ip or not obs.dst_ip or obs.src_ip == "0.0.0.0" or obs.dst_port < 0:
            return None
        if not self._has_observed_packets(obs):
            return None
        if self._is_probable_response_flow(obs):
            return None
        if self._is_ephemeral(int(obs.dst_port)):
            return None
        scope_key = self._scan_scope(obs)
        dq = self.scan_memory[scope_key]
        dq.append((now, int(obs.dst_port)))
        self._purge_scan(scope_key, now)
        unique_ports = {port for _, port in dq}
        if len(unique_ports) >= self.scan_attack_unique_ports:
            severity = "high"
        elif len(unique_ports) >= self.scan_unique_ports:
            severity = "medium"
        else:
            return None
        return RuleDecision(
            status="CONTEXT",
            reason=f"scan support: {len(unique_ports)} low dst ports to {obs.dst_ip} in {self.scan_window_s:.0f}s",
            severity=severity,
            hit_increment=1,
            hit_count=len(unique_ports),
            emit_key=f"scan:{scope_key}",
            source="scan_rule",
            category="Probe",
        )

    def _service_support(self, obs: FlowObservation) -> RuleDecision | None:
        if obs.protocol in _NON_PORT_PROTOCOLS or obs.dst_port < 0:
            return None
        if not self._has_observed_packets(obs):
            return None
        if self._is_probable_response_flow(obs):
            return None
        if self._is_ephemeral(int(obs.dst_port)):
            return None
        allowed_ports = self.allowlisted_services.get(obs.dst_ip)
        if not allowed_ports:
            return None
        scope_key = self._entity_scope(obs)
        if obs.dst_port not in allowed_ports:
            self.service_violation_hits[scope_key] += 1
            hits = self.service_violation_hits[scope_key]
            return RuleDecision(
                status="CONTEXT",
                reason=f"service support: unexpected privileged dst_port {obs.dst_port} for {obs.dst_ip}",
                severity="medium" if hits < 3 else "high",
                hit_increment=1,
                hit_count=hits,
                emit_key=f"service:{scope_key}",
                source="service_rule",
                category="Service-Anomaly",
            )
        self.service_violation_hits[scope_key] = 0
        return None

    def _protocol_flood_support(self, obs: FlowObservation) -> RuleDecision | None:
        if not self._has_activity(obs):
            return None
        pkt_rate = float(obs.packet_rate)
        byte_rate = float(obs.byte_rate)
        avg_size = max(float(obs.avg_packet_size), 1.0)
        if obs.protocol == 17 and (pkt_rate >= self.udp_flood_packet_rate or byte_rate >= self.udp_flood_byte_rate):
            return RuleDecision(
                status="CONTEXT",
                reason=f"udp flood support: pkt_rate={pkt_rate:.1f}, byte_rate={byte_rate:.1f}",
                severity="high",
                hit_increment=1,
                hit_count=1,
                emit_key=f"udp:{self._entity_scope(obs)}",
                source="udp_rule",
                category="UDP-Flood",
            )
        if obs.protocol == 1 and pkt_rate >= self.icmp_flood_packet_rate:
            return RuleDecision(
                status="CONTEXT",
                reason=f"icmp flood support: pkt_rate={pkt_rate:.1f}",
                severity="high",
                hit_increment=1,
                hit_count=1,
                emit_key=f"icmp:{self._entity_scope(obs)}",
                source="icmp_rule",
                category="ICMP-Flood",
            )
        if obs.protocol == 6 and pkt_rate >= self.tcp_small_pkt_attack_rate and avg_size <= self.small_pkt_avg_size_max:
            return RuleDecision(
                status="CONTEXT",
                reason=f"tcp small-packet flood support: pkt_rate={pkt_rate:.1f}, avg_size={avg_size:.1f}",
                severity="high",
                hit_increment=1,
                hit_count=1,
                emit_key=f"tcpflood:{self._entity_scope(obs)}",
                source="tcp_flood_rule",
                category="TCP-Flood",
            )
        return None

    def _volumetric_support(self, obs: FlowObservation, now: float) -> RuleDecision | None:
        if not self._has_activity(obs):
            return None
        baseline = self._baseline(self._service_key(obs), now)
        pkt_rate = float(obs.packet_rate)
        byte_rate = float(obs.byte_rate)
        pkt_delta = float(obs.packet_delta)
        byte_delta = float(obs.byte_delta)
        hard = (
            pkt_rate >= self.hard_packet_rate
            or byte_rate >= self.hard_byte_rate
            or pkt_delta >= self.hard_packet_delta
            or byte_delta >= self.hard_byte_delta
        )
        scope_key = self._entity_scope(obs)
        if hard:
            self.volumetric_hits[scope_key] += 1
            return RuleDecision(
                status="CONTEXT",
                reason=(
                    f"volumetric support: pkt_rate={pkt_rate:.1f}, byte_rate={byte_rate:.1f}, "
                    f"pkt_delta={pkt_delta:.1f}, byte_delta={byte_delta:.1f}"
                ),
                severity="high",
                hit_increment=1,
                hit_count=self.volumetric_hits[scope_key],
                emit_key=f"vol:{scope_key}",
                source="volumetric_rule",
                category="Traffic-Flood",
            )
        if baseline is None:
            return None
        base_pkt, base_byte = baseline
        warn = pkt_rate >= base_pkt * self.rate_multiplier_warn or byte_rate >= base_byte * self.rate_multiplier_warn
        attackish = pkt_rate >= base_pkt * self.rate_multiplier_attack or byte_rate >= base_byte * self.rate_multiplier_attack
        if warn or attackish:
            self.volumetric_hits[scope_key] += 1
            severity = "high" if attackish else "medium"
            return RuleDecision(
                status="CONTEXT",
                reason=(
                    f"baseline volumetric support: pkt_rate={pkt_rate:.1f} ({pkt_rate/max(base_pkt,1e-6):.2f}x), "
                    f"byte_rate={byte_rate:.1f} ({byte_rate/max(base_byte,1e-6):.2f}x)"
                ),
                severity=severity,
                hit_increment=1,
                hit_count=self.volumetric_hits[scope_key],
                emit_key=f"vol:{scope_key}",
                source="volumetric_rule",
                category="Traffic-Flood",
            )
        self.volumetric_hits[scope_key] = 0
        return None

    def _bruteforce_support(self, obs: FlowObservation, now: float) -> RuleDecision | None:
        if int(obs.dst_port) not in self.bfa_ports or obs.protocol != 6:
            return None
        if not self._has_observed_packets(obs):
            return None
        if self._is_probable_response_flow(obs):
            return None
        scope_key = f"{obs.src_ip}->{obs.dst_ip}:{obs.dst_port}"
        dq = self.bfa_memory[scope_key]
        dq.append((now, int(obs.dst_port)))
        while dq and now - dq[0][0] > self.bfa_window_s:
            dq.popleft()
        attempts = len(dq)
        if attempts < self.bfa_attempt_threshold:
            return None
        severity = "high" if attempts >= self.bfa_attack_threshold else "medium"
        return RuleDecision(
            status="CONTEXT",
            reason=f"bfa support: {attempts} short TCP attempts to {obs.dst_ip}:{obs.dst_port} in {self.bfa_window_s:.0f}s",
            severity=severity,
            hit_increment=1,
            hit_count=attempts,
            emit_key=f"bfa:{scope_key}",
            source="bfa_rule",
            category="BFA",
        )

    @staticmethod
    def _max_support_severity(supports: list[RuleDecision]) -> str:
        if any(s.severity == "high" for s in supports):
            return "high"
        if any(s.severity == "medium" for s in supports):
            return "medium"
        return "low"

    @staticmethod
    def _support_summary(supports: list[RuleDecision]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        if not supports:
            return (), (), ""
        reasons = tuple(s.reason for s in supports)
        sources = tuple(s.source for s in supports)
        headline = supports[0].reason if len(supports) == 1 else f"{supports[0].reason} (+{len(supports)-1} supports)"
        return reasons, sources, headline

    @staticmethod
    def _support_only_allowed(supports: list[RuleDecision]) -> bool:
        if not supports:
            return False
        # Runtime display policy: only Brute Force may promote to ATTACK via rules-only.
        return any(s.category == "BFA" for s in supports)

    @staticmethod
    def _dominant_category(supports: list[RuleDecision], fallback: str = "Anomaly") -> str:
        if not supports:
            return fallback
        return supports[0].category or fallback

    def _emergency_attack(self, supports: list[RuleDecision]) -> RuleDecision | None:
        if not self.emergency_rules_enabled or not self.rules_only_attack_enabled:
            return None
        if not supports:
            return None
        severe = [s for s in supports if s.severity == "high"]
        if len(severe) < 2:
            return None
        reasons, sources, headline = self._support_summary(severe)
        return RuleDecision(
            status="ATTACK",
            reason=f"emergency rules-only attack | {headline}",
            severity="high",
            emit_key=severe[0].emit_key,
            source=severe[0].source,
            category=self._dominant_category(severe, "Emergency"),
            support_reasons=reasons,
            support_sources=sources,
        )

    def evaluate(self, obs: FlowObservation, score: float | None, threshold: float, score_direction: str) -> RuleDecision:
        now = time.time()
        self.entity_last_seen[obs.key] = now
        self.cleanup(now)

        supports = [
            d for d in (
                self._scan_support(obs, now),
                self._bruteforce_support(obs, now),
                self._service_support(obs),
                self._protocol_flood_support(obs),
                self._volumetric_support(obs, now),
            ) if d is not None
        ]
        support_reasons, support_sources, support_headline = self._support_summary(supports)
        support_severity = self._max_support_severity(supports)

        if score is None:
            self._update_baseline(obs, now)
            emergency = self._emergency_attack(supports)
            if emergency is not None:
                return emergency
            if supports and support_severity in {"high", "medium"} and self._support_only_allowed(supports):
                dominant_support = supports[0]
                return RuleDecision(
                    status="ATTACK",
                    reason=f"rule-confirmed attack | {support_headline}",
                    severity=support_severity,
                    hit_count=max((s.hit_count for s in supports), default=0),
                    emit_key=next((s.emit_key for s in supports if s.emit_key), obs.key),
                    source=dominant_support.source,
                    category=self._dominant_category(supports),
                    support_reasons=support_reasons,
                    support_sources=support_sources,
                )
            if supports:
                return RuleDecision(
                    status="SUSPECT",
                    reason=support_headline or "rule support during warm-up",
                    severity=support_severity,
                    hit_count=max((s.hit_count for s in supports), default=0),
                    emit_key=next((s.emit_key for s in supports if s.emit_key), obs.key),
                    source=supports[0].source,
                    category=self._dominant_category(supports),
                    support_reasons=support_reasons,
                    support_sources=support_sources,
                )
            return RuleDecision(
                status="NORMAL",
                reason="",
                source="runtime",
                category="Normal",
                support_reasons=support_reasons,
                support_sources=support_sources,
            )

        is_attack = score <= threshold if score_direction == "lower_is_attack" else score >= threshold
        self._update_baseline(obs, now)
        if is_attack:
            self.alert_hits[obs.key] += 1
            hits = self.alert_hits[obs.key]
            margin = (threshold - score) if score_direction == "lower_is_attack" else (score - threshold)
            attack_floor_ok = score <= self.attack_display_floor if score_direction == "lower_is_attack" else score >= self.attack_display_floor
            if support_severity == "high" or (attack_floor_ok and hits >= self.min_alert_hits) or (hits >= (self.min_alert_hits + 1)):
                status = "ATTACK"
                severity = "high" if support_severity == "high" or attack_floor_ok else "medium"
            else:
                status = "SUSPECT"
                severity = "medium"
            reason = f"model threshold hit {hits} times (margin={margin:.5f})"
            if support_headline:
                reason += f" | {support_headline}"
            return RuleDecision(
                status=status,
                reason=reason,
                severity=severity,
                hit_increment=1,
                hit_count=hits,
                emit_key=f"model:{obs.key}",
                source="model",
                category=self._dominant_category(supports, "Model-Attack"),
                support_reasons=support_reasons,
                support_sources=support_sources,
            )

        self.alert_hits[obs.key] = 0
        borderline = self._near_threshold(float(score), threshold, score_direction)
        emergency = self._emergency_attack(supports)
        if emergency is not None:
            return emergency
        if supports and (borderline or support_severity in {"medium", "high"}):
            dominant_support = supports[0]
            promoted = self._support_only_allowed(supports) and not borderline
            return RuleDecision(
                status="ATTACK" if promoted else "SUSPECT",
                reason=(
                    f"model borderline near threshold | {support_headline}"
                    if borderline else f"rule support without model hit | {support_headline}"
                ),
                severity="high" if support_severity == "high" else "medium",
                hit_count=max((s.hit_count for s in supports), default=0),
                emit_key=f"assist:{obs.key}",
                source="model+rules" if borderline else dominant_support.source,
                category=self._dominant_category(supports),
                support_reasons=support_reasons,
                support_sources=support_sources,
            )
        return RuleDecision(
            status="NORMAL",
            reason="score below threshold",
            severity="low",
            hit_count=0,
            source="model",
            category="Normal",
            support_reasons=support_reasons,
            support_sources=support_sources,
        )

    def should_emit_alert(self, entity_key: str, now: float | None = None) -> bool:
        now = now or time.time()
        last_ts = self.last_alert_ts_by_key.get(entity_key, 0.0)
        if (now - last_ts) >= self.alert_cooldown_seconds:
            self.last_alert_ts_by_key[entity_key] = now
            return True
        return False
