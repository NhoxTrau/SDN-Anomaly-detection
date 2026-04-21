from __future__ import annotations

import argparse
import heapq
import ipaddress
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3

from sdn_nids_realtime.runtime.feature_builder import OpenFlowFeatureBuilder
from sdn_nids_realtime.runtime.scalability import pressure_state, recommended_poll_interval, summarize_polling_metrics
from sdn_nids_realtime.runtime.telemetry_runtime import TelemetryRuntime
from sdn_nids_realtime.train_v2.common import DEFAULT_BUNDLE_DIR, DEFAULT_RUN_ID, DEFAULT_RUNTIME_ROOT


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RawFlowItem:
    info: object
    packet_count: float
    byte_count: float
    duration_s: float


@dataclass
class RawFlowBatch:
    dpid: int
    poll_cycle_id: int
    poll_request_ts: float
    poll_reply_ts: float
    reply_part_count: int
    items: list[RawFlowItem] = field(default_factory=list)


class TelemetryAwareRyuController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(
        self,
        *args,
        runtime_root: str | None = None,
        bundle_path: str | None = None,
        run_id: str | None = None,
        poll_interval_s: float | None = None,
        adaptive_polling: bool | None = None,
        min_poll_interval_s: float | None = None,
        max_poll_interval_s: float | None = None,
        enable_blocking: bool | None = None,
        **kwargs,
    ):
        runtime_root = runtime_root or os.environ.get("SDN_NIDS_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))
        bundle_path = bundle_path or os.environ.get("SDN_NIDS_BUNDLE_PATH", str(DEFAULT_BUNDLE_DIR / "lstm"))
        run_id = run_id or os.environ.get("SDN_NIDS_RUN_ID", DEFAULT_RUN_ID)
        poll_interval_s = float(poll_interval_s if poll_interval_s is not None else os.environ.get("SDN_NIDS_POLL_INTERVAL", 1.0))
        min_poll_interval_s = float(min_poll_interval_s if min_poll_interval_s is not None else os.environ.get("SDN_NIDS_MIN_POLL_INTERVAL", 0.75))
        max_poll_interval_s = float(max_poll_interval_s if max_poll_interval_s is not None else os.environ.get("SDN_NIDS_MAX_POLL_INTERVAL", 3.0))
        if adaptive_polling is None:
            adaptive_polling = _truthy_env("SDN_NIDS_ADAPTIVE_POLLING", default=False)
        if enable_blocking is None:
            enable_blocking = _truthy_env("SDN_NIDS_ENABLE_BLOCKING", default=False)

        super().__init__(*args, **kwargs)
        self.datapaths: dict[int, object] = {}
        self.mac_to_port: dict[int, dict[str, int]] = {}
        self.blocked_ips: set[str] = set()
        self.enable_blocking = bool(enable_blocking)
        self.runtime = TelemetryRuntime(runtime_root=runtime_root, bundle_path=bundle_path, run_id=run_id)
        self.runtime_key_mode = str(os.environ.get("SDN_NIDS_RUNTIME_KEY_MODE", "flow")).strip().lower() or "flow"
        if self.runtime_key_mode not in {"flow", "service"}:
            self.runtime_key_mode = "flow"
        self.feature_builder = OpenFlowFeatureBuilder(
            poll_interval_s=poll_interval_s,
            feature_names=self.runtime.feature_names,
            key_mode=self.runtime_key_mode,
        )
        self.poll_interval_s = float(poll_interval_s)
        self.adaptive_polling = bool(adaptive_polling)
        self.min_poll_interval_s = float(min_poll_interval_s)
        self.max_poll_interval_s = float(max_poll_interval_s)
        self.evict_on_flow_removed = _truthy_env("SDN_NIDS_EVICT_ON_FLOW_REMOVED", default=False)
        self.flow_match_mode = str(os.environ.get("SDN_NIDS_FLOW_MATCH_MODE", "exact" if self.runtime_key_mode == "flow" else "conversation")).strip().lower() or ("exact" if self.runtime_key_mode == "flow" else "conversation")
        self.flow_idle_timeout_s = int(float(os.environ.get("SDN_NIDS_FLOW_IDLE_TIMEOUT", 12.0)))
        self.emit_only_on_change = _truthy_env("SDN_NIDS_EMIT_ONLY_ON_CHANGE", default=True)
        self.min_emit_packet_delta = float(os.environ.get("SDN_NIDS_MIN_EMIT_PACKET_DELTA", 1.0))
        self.min_emit_byte_delta = float(os.environ.get("SDN_NIDS_MIN_EMIT_BYTE_DELTA", 1.0))
        self.force_emit_after_s = float(os.environ.get("SDN_NIDS_FORCE_EMIT_AFTER", max(8.0, 4.0 * self.poll_interval_s)))
        self.poll_reply_timeout_s = float(os.environ.get("SDN_NIDS_POLL_REPLY_TIMEOUT", max(4.0 * self.poll_interval_s, 4.0)))
        self.poll_backpressure_threshold = int(os.environ.get("SDN_NIDS_POLL_BACKPRESSURE_THRESHOLD", 2500))
        self.poll_backpressure_sleep_s = float(os.environ.get("SDN_NIDS_POLL_BACKPRESSURE_SLEEP", min(self.poll_interval_s, 0.20)))
        self.raw_stats_queue_maxsize = int(os.environ.get("SDN_NIDS_RAW_STATS_QUEUE_MAXSIZE", 4096))
        self.high_pressure_threshold = int(os.environ.get("SDN_NIDS_HIGH_PRESSURE_THRESHOLD", max(1000, self.poll_backpressure_threshold // 2)))
        self.raw_stats_high_pressure_threshold = int(os.environ.get("SDN_NIDS_RAW_STATS_HIGH_PRESSURE_THRESHOLD", max(256, self.raw_stats_queue_maxsize // 2)))
        self.reply_delay_high_ms = float(os.environ.get("SDN_NIDS_REPLY_DELAY_HIGH_MS", max(500.0, self.poll_interval_s * 1200.0)))
        self.dashboard_active_flows_limit = max(20, int(os.environ.get("SDN_NIDS_DASHBOARD_ACTIVE_FLOWS_LIMIT", "120")))
        self.dashboard_topology_links_limit = max(20, int(os.environ.get("SDN_NIDS_DASHBOARD_TOPOLOGY_LINKS_LIMIT", "60")))
        self.dashboard_topology_hosts_limit = max(20, int(os.environ.get("SDN_NIDS_DASHBOARD_TOPOLOGY_HOSTS_LIMIT", "60")))
        self.max_obs_per_reply_high_pressure = int(os.environ.get("SDN_NIDS_MAX_OBS_PER_REPLY_HIGH_PRESSURE", 1200))
        self.max_obs_per_reply_extreme_pressure = int(os.environ.get("SDN_NIDS_MAX_OBS_PER_REPLY_EXTREME_PRESSURE", max(200, self.max_obs_per_reply_high_pressure // 2)))
        self._pending_flow_stats: dict[int, dict[str, float | int]] = {}
        self._flow_stats_timeout_count: dict[int, int] = {}
        self._poll_cycle_seq = 0
        self._switch_poll_metrics: dict[int, dict[str, float | int]] = {}
        self._last_poll_log_ts = 0.0
        self.controller_metrics_publish_interval_s = max(0.10, float(os.environ.get("SDN_NIDS_CONTROLLER_METRICS_PUBLISH_INTERVAL_S", 0.75)))
        self._last_controller_metrics_publish_ts = 0.0
        self.raw_stats_queue: Queue[RawFlowBatch] = Queue(maxsize=self.raw_stats_queue_maxsize)
        self.raw_stats_drop_count = 0
        self._stop = threading.Event()
        self._feature_thread = threading.Thread(target=self._feature_worker, daemon=True)
        self._feature_thread.start()
        self._monitor_thread = threading.Thread(target=self._runtime_worker, daemon=True)
        self._monitor_thread.start()
        self._alert_thread = threading.Thread(target=self._react_to_alerts, daemon=True)
        self._alert_thread.start()
        self._last_state_eviction_ts = time.time()
        self._poller_thread = hub.spawn(self._poll_loop)
        self.logger.info(
            "Telemetry controller started | blocking=%s | adaptive_polling=%s | bundle=%s | features=%s | match_mode=%s | key_mode=%s",
            self.enable_blocking,
            self.adaptive_polling,
            bundle_path,
            len(self.runtime.feature_names),
            self.flow_match_mode,
            self.runtime_key_mode,
        )
        self._publish_controller_metrics()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        dpid = datapath.id
        # If this is a reconnect, clear stale pending-stats state so the poll
        # loop doesn't immediately fire a false timeout on the old entry.
        if dpid in self._pending_flow_stats:
            self._pending_flow_stats.pop(dpid, None)
            self._flow_stats_timeout_count.pop(dpid, None)
            self.logger.info("Switch %s reconnected — cleared stale pending stats", dpid)
        self.datapaths[dpid] = datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, priority=0, match=match, actions=actions)
        self.logger.info("Switch %s connected", dpid)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def state_change_handler(self, ev):
        dpid = ev.datapath.id
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dpid] = ev.datapath
            # Clear stale pending entry on every state transition to MAIN
            self._pending_flow_stats.pop(dpid, None)
            self._flow_stats_timeout_count.pop(dpid, None)
        elif ev.state == CONFIG_DISPATCHER and dpid in self.datapaths:
            self.datapaths[dpid] = ev.datapath

    def add_flow(self, datapath, priority, match, actions, idle_timeout: int = 0, hard_timeout: int = 0, send_flow_removed: bool = False):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        flags = ofproto.OFPFF_SEND_FLOW_REM if send_flow_removed else 0
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
            flags=flags,
            match=match,
            instructions=inst,
        )
        datapath.send_msg(mod)

    def _extract_l4_fields(self, pkt) -> tuple[int, int, int]:
        proto = 0
        src_port = -1
        dst_port = -1
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        if tcp_pkt is not None:
            proto = 6
            src_port = int(tcp_pkt.src_port)
            dst_port = int(tcp_pkt.dst_port)
        elif udp_pkt is not None:
            proto = 17
            src_port = int(udp_pkt.src_port)
            dst_port = int(udp_pkt.dst_port)
        else:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt is not None:
                proto = int(ip_pkt.proto)
        return proto, src_port, dst_port

    def _build_flow_match(self, parser, in_port: int, eth_pkt, ip_pkt, proto: int, src_port: int, dst_port: int):
        if ip_pkt is None:
            return parser.OFPMatch(in_port=in_port, eth_src=eth_pkt.src, eth_dst=eth_pkt.dst)

        if self.flow_match_mode == "exact":
            match_kwargs = {
                "in_port": in_port,
                "eth_src": eth_pkt.src,
                "eth_dst": eth_pkt.dst,
                "eth_type": 0x0800,
                "ipv4_src": ip_pkt.src,
                "ipv4_dst": ip_pkt.dst,
            }
            if proto in (6, 17):
                match_kwargs["ip_proto"] = proto
                if src_port >= 0:
                    match_kwargs["tcp_src" if proto == 6 else "udp_src"] = src_port
                if dst_port >= 0:
                    match_kwargs["tcp_dst" if proto == 6 else "udp_dst"] = dst_port
            elif proto > 0:
                match_kwargs["ip_proto"] = proto
            return parser.OFPMatch(**match_kwargs)

        # "conversation" mode: keep forwarding correct but collapse subflows that only differ by src_port/MAC.
        match_kwargs = {
            "in_port": in_port,
            "eth_type": 0x0800,
            "ipv4_src": ip_pkt.src,
            "ipv4_dst": ip_pkt.dst,
        }
        if proto in (6, 17):
            match_kwargs["ip_proto"] = proto
            if dst_port >= 0:
                match_kwargs["tcp_dst" if proto == 6 else "udp_dst"] = dst_port
        elif proto > 0:
            match_kwargs["ip_proto"] = proto
        return parser.OFPMatch(**match_kwargs)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match.get("in_port", 0)
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if eth_pkt is None or eth_pkt.ethertype == 0x88CC:
            return
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth_pkt.src] = in_port
        out_port = self.mac_to_port[dpid].get(eth_pkt.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            proto, src_port, dst_port = self._extract_l4_fields(pkt)
            match = self._build_flow_match(parser, in_port, eth_pkt, ip_pkt, proto, src_port, dst_port)
            self.add_flow(
                datapath,
                priority=10,
                match=match,
                actions=actions,
                idle_timeout=self.flow_idle_timeout_s,
                send_flow_removed=self.evict_on_flow_removed,
            )
        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        if not self.evict_on_flow_removed:
            return
        msg = ev.msg
        match = getattr(msg, "match", None)
        if match is None:
            return
        try:
            entity_key = self.feature_builder.evict_state_by_match(msg.datapath.id, match)
            if entity_key:
                self.runtime.evict_entity(entity_key)
        except Exception as exc:
            self.logger.debug("Failed to evict removed flow state: %s", exc)

    def _adaptive_polling_frozen(self) -> bool:
        state = self.runtime.current_state()
        status = str(state.get("status", "")).upper()
        return bool(state.get("recent_alerts")) or status in {"ATTACK", "SUSPECT"}

    def _ensure_switch_metrics(self, dpid: int) -> dict[str, float | int]:
        metrics = self._switch_poll_metrics.get(dpid)
        if metrics is None:
            metrics = {
                "polls_sent": 0,
                "replies_received": 0,
                "timeouts": 0,
                "total_reply_delay_ms": 0.0,
                "avg_reply_delay_ms": 0.0,
                "max_reply_delay_ms": 0.0,
                "last_reply_delay_ms": 0.0,
                "avg_flows_per_reply": 0.0,
                "total_flows_observed": 0,
                "last_flows_per_reply": 0,
                "last_reply_parts": 0,
                "trimmed_replies": 0,
                "trimmed_flows": 0,
                "pressure_state": "normal",
            }
            self._switch_poll_metrics[dpid] = metrics
        return metrics

    def _record_poll_sent(self, dpid: int) -> None:
        metrics = self._ensure_switch_metrics(dpid)
        metrics["polls_sent"] = int(metrics.get("polls_sent", 0)) + 1

    def _record_poll_timeout(self, dpid: int) -> None:
        metrics = self._ensure_switch_metrics(dpid)
        metrics["timeouts"] = int(metrics.get("timeouts", 0)) + 1

    def _record_poll_reply(self, dpid: int, reply_delay_ms: float, flows_in_reply: int, reply_parts: int) -> None:
        metrics = self._ensure_switch_metrics(dpid)
        metrics["replies_received"] = int(metrics.get("replies_received", 0)) + 1
        metrics["total_reply_delay_ms"] = float(metrics.get("total_reply_delay_ms", 0.0)) + float(reply_delay_ms)
        metrics["avg_reply_delay_ms"] = float(metrics["total_reply_delay_ms"]) / max(int(metrics["replies_received"]), 1)
        metrics["max_reply_delay_ms"] = max(float(metrics.get("max_reply_delay_ms", 0.0)), float(reply_delay_ms))
        metrics["last_reply_delay_ms"] = float(reply_delay_ms)
        metrics["total_flows_observed"] = int(metrics.get("total_flows_observed", 0)) + int(flows_in_reply)
        metrics["last_flows_per_reply"] = int(flows_in_reply)
        metrics["avg_flows_per_reply"] = float(metrics["total_flows_observed"]) / max(int(metrics["replies_received"]), 1)
        metrics["last_reply_parts"] = int(reply_parts)

    def _record_poll_trim(self, dpid: int, dropped_flows: int, state: str) -> None:
        metrics = self._ensure_switch_metrics(dpid)
        if dropped_flows > 0:
            metrics["trimmed_replies"] = int(metrics.get("trimmed_replies", 0)) + 1
            metrics["trimmed_flows"] = int(metrics.get("trimmed_flows", 0)) + int(dropped_flows)
        metrics["pressure_state"] = str(state or "normal")

    def _publish_controller_metrics(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_controller_metrics_publish_ts) < self.controller_metrics_publish_interval_s:
            return
        states = list(self.feature_builder.states.values())
        active_flow_count = len(states)
        preview_limit = max(self.dashboard_active_flows_limit, self.dashboard_topology_links_limit, 1)
        preview_states = heapq.nlargest(
            preview_limit,
            states,
            key=lambda state: (float(state.last_seen_ts), float(state.last_packet_rate), float(state.last_byte_rate)),
        )
        active_flows = [
            {
                "entity_key": state.key,
                "dpid": int(state.dpid),
                "src_ip": state.src_ip,
                "dst_ip": state.dst_ip,
                "src_port": int(state.src_port),
                "dst_port": int(state.dst_port),
                "protocol": int(state.protocol),
                "last_packet_count": float(state.last_packet_count),
                "last_byte_count": float(state.last_byte_count),
                "last_packet_rate": float(state.last_packet_rate),
                "last_byte_rate": float(state.last_byte_rate),
                "flow_age_s": float(state.flow_age_s),
                "seen_polls": int(state.seen_polls),
                "last_seen_ts": float(state.last_seen_ts),
            }
            for state in preview_states
        ]
        topology_hosts = sorted({ip for state in states for ip in (state.src_ip, state.dst_ip) if ip})
        topology = {
            "switches": [{"id": f"s{idx + 1}", "dpid": int(dpid)} for idx, dpid in enumerate(sorted(self.datapaths.keys()))],
            "hosts": topology_hosts[: self.dashboard_topology_hosts_limit],
            "host_count": int(len(topology_hosts)),
            "links": [
                {
                    "dpid": int(flow["dpid"]),
                    "src_ip": flow["src_ip"],
                    "dst_ip": flow["dst_ip"],
                    "dst_port": int(flow["dst_port"]),
                    "protocol": int(flow["protocol"]),
                    "packet_rate": float(flow["last_packet_rate"]),
                    "byte_rate": float(flow["last_byte_rate"]),
                }
                for flow in active_flows[: self.dashboard_topology_links_limit]
            ],
            "link_count": int(min(active_flow_count, self.dashboard_topology_links_limit)),
        }
        polling_stats = {
            str(dpid): {
                key: (float(value) if isinstance(value, float) else value)
                for key, value in metrics.items()
                if key not in {"total_reply_delay_ms"}
            }
            for dpid, metrics in self._switch_poll_metrics.items()
        }
        polling_summary = summarize_polling_metrics(polling_stats)
        recommended_interval = recommended_poll_interval(
            current_interval_s=float(self.poll_interval_s),
            min_interval_s=float(self.min_poll_interval_s),
            max_interval_s=float(self.max_poll_interval_s),
            runtime_queue_depth=int(self.runtime.queue_depth_now()),
            raw_queue_depth=int(self.raw_stats_queue.qsize()),
            avg_reply_delay_ms=float(polling_summary.get("avg_reply_delay_ms", 0.0)),
            timeout_total=int(polling_summary.get("timeout_total", 0)),
            throughput_obs_s=float((self.runtime.current_state() or {}).get("throughput_obs_s", 0.0) or 0.0),
            queue_threshold=int(self.poll_backpressure_threshold),
            raw_threshold=int(self.raw_stats_high_pressure_threshold),
            reply_delay_high_ms=float(self.reply_delay_high_ms),
        )
        current_pressure = pressure_state(
            runtime_queue_depth=int(self.runtime.queue_depth_now()),
            raw_queue_depth=int(self.raw_stats_queue.qsize()),
            timeouts=int(polling_summary.get("timeout_total", 0)),
            avg_reply_delay_ms=float(polling_summary.get("avg_reply_delay_ms", 0.0)),
            queue_threshold=int(self.poll_backpressure_threshold),
            raw_threshold=int(self.raw_stats_high_pressure_threshold),
            reply_delay_high_ms=float(self.reply_delay_high_ms),
        )
        self.runtime.set_controller_metrics({
            "raw_stats_queue_depth": int(self.raw_stats_queue.qsize()),
            "raw_stats_queue_maxsize": int(self.raw_stats_queue_maxsize),
            "raw_stats_drop_count": int(self.raw_stats_drop_count),
            "feature_state_count": int(self.feature_builder.state_count()),
            "poll_interval_s": float(self.poll_interval_s),
            "adaptive_polling": bool(self.adaptive_polling),
            "pressure_state": current_pressure,
            "recommended_poll_interval_s": float(recommended_interval),
            "polling_summary": polling_summary,
            "polling_stats": polling_stats,
            "active_flow_count": int(active_flow_count),
            "active_flows": active_flows[: self.dashboard_active_flows_limit],
            "topology": topology,
        })
        self._last_controller_metrics_publish_ts = now

    def _next_poll_interval(self, queue_depth: int) -> float:
        polling_summary = summarize_polling_metrics({
            str(dpid): metrics for dpid, metrics in self._switch_poll_metrics.items()
        })
        raw_qd = int(self.raw_stats_queue.qsize())
        throughput_obs_s = float((self.runtime.current_state() or {}).get("throughput_obs_s", 0.0) or 0.0)
        return recommended_poll_interval(
            current_interval_s=float(self.poll_interval_s),
            min_interval_s=float(self.min_poll_interval_s),
            max_interval_s=float(self.max_poll_interval_s),
            runtime_queue_depth=int(queue_depth),
            raw_queue_depth=raw_qd,
            avg_reply_delay_ms=float(polling_summary.get("avg_reply_delay_ms", 0.0)),
            timeout_total=int(polling_summary.get("timeout_total", 0)),
            throughput_obs_s=throughput_obs_s,
            queue_threshold=int(self.high_pressure_threshold),
            raw_threshold=int(self.raw_stats_high_pressure_threshold),
            reply_delay_high_ms=float(self.reply_delay_high_ms),
        )

    def _poll_loop(self):
        while not self._stop.is_set():
            cycle_start = time.time()
            qd = self.runtime.queue_depth_now()
            if qd >= self.poll_backpressure_threshold:
                if (cycle_start - self._last_poll_log_ts) >= 5.0:
                    self.logger.warning(
                        "Polling paused due to runtime backpressure | queue_depth=%s | threshold=%s",
                        qd,
                        self.poll_backpressure_threshold,
                    )
                    self._last_poll_log_ts = cycle_start
                hub.sleep(self.poll_backpressure_sleep_s)
                continue

            start = time.perf_counter()
            for dp in list(self.datapaths.values()):
                pending = self._pending_flow_stats.get(dp.id)
                if pending is not None:
                    last_progress_at = float(pending.get("last_progress_at", pending.get("started_at", cycle_start)))
                    age_s = cycle_start - last_progress_at
                    if age_s < self.poll_reply_timeout_s:
                        continue
                    timeout_count = int(self._flow_stats_timeout_count.get(dp.id, 0)) + 1
                    self._flow_stats_timeout_count[dp.id] = timeout_count
                    self._record_poll_timeout(dp.id)
                    if timeout_count == 1 or (cycle_start - self._last_poll_log_ts) >= 5.0:
                        self.logger.warning(
                            "FlowStats reply timeout on datapath %s after %.2fs without reply progress; resetting pending flag (parts=%s, consecutive=%s)",
                            dp.id,
                            age_s,
                            int(pending.get("parts", 0)),
                            timeout_count,
                        )
                        self._last_poll_log_ts = cycle_start
                    self._pending_flow_stats.pop(dp.id, None)
                try:
                    parser = dp.ofproto_parser
                    req = parser.OFPFlowStatsRequest(dp)
                    poll_cycle_id = self._poll_cycle_seq
                    self._poll_cycle_seq += 1
                    self._pending_flow_stats[dp.id] = {
                        "started_at": cycle_start,
                        "last_progress_at": cycle_start,
                        "parts": 0,
                        "poll_cycle_id": poll_cycle_id,
                    }
                    self._record_poll_sent(dp.id)
                    dp.send_msg(req)
                except Exception as exc:
                    self._pending_flow_stats.pop(getattr(dp, "id", None), None)
                    self.logger.warning("Polling error for datapath %s: %s", getattr(dp, "id", "?"), exc)
            elapsed = time.perf_counter() - start
            sleep_for = max(0.0, self.poll_interval_s - elapsed)
            hub.sleep(sleep_for)
            if self.adaptive_polling and not self._adaptive_polling_frozen():
                qd = self.runtime.queue_depth_now()
                next_interval = self._next_poll_interval(qd)
                self.poll_interval_s = float(next_interval)
                self.feature_builder.poll_interval_s = float(next_interval)
            now = time.time()
            if now - self._last_state_eviction_ts >= 30.0:
                self.feature_builder.evict_idle_states(current_ts=now, idle_timeout_s=120.0)
                self._last_state_eviction_ts = now
            self._publish_controller_metrics()

    def _observation_priority(self, payload: dict[str, object]) -> tuple[float, float, float, float]:
        info = payload["info"]
        state = self.feature_builder.get_state(info.key())
        if state is None:
            return (0.0, float(payload["packet_count"]), float(payload["byte_count"]), 0.0)
        pkt_delta = max(0.0, float(payload["packet_count"]) - float(state.last_packet_count))
        byte_delta = max(0.0, float(payload["byte_count"]) - float(state.last_byte_count))
        history_bias = 1.0 if getattr(state, "seen_polls", 0) > 0 else 0.0
        return (history_bias, pkt_delta, byte_delta, float(payload["packet_count"]))

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        now = time.time()
        more_flag = getattr(dp.ofproto, "OFPMPF_REPLY_MORE", 0)
        pending = self._pending_flow_stats.get(dp.id)
        poll_request_ts = float(pending.get("started_at", now)) if pending is not None else now
        poll_cycle_id = int(pending.get("poll_cycle_id", 0)) if pending is not None else 0
        if pending is not None:
            pending["last_progress_at"] = now
            pending["parts"] = int(pending.get("parts", 0)) + 1
        reply_part_count = int(pending.get("parts", 0)) if pending is not None else 1
        final_reply = False
        try:
            final_reply = not (int(getattr(msg, "flags", 0) or 0) & int(more_flag))
            if final_reply:
                self._pending_flow_stats.pop(dp.id, None)
                self._flow_stats_timeout_count.pop(dp.id, None)
        except Exception:
            final_reply = True
            self._pending_flow_stats.pop(dp.id, None)
            self._flow_stats_timeout_count.pop(dp.id, None)
        aggregates: dict[str, dict[str, object]] = {}
        for stat in msg.body:
            if getattr(stat, "priority", 0) <= 0:
                continue
            match = stat.match
            if hasattr(match, "get"):
                eth_type = match.get("eth_type")
                if eth_type != 0x0800:
                    continue
                if not match.get("ipv4_src") or not match.get("ipv4_dst"):
                    continue
            try:
                info = self.feature_builder.build_key_from_match(dp.id, match)
                if not info.src_ip or not info.dst_ip:
                    continue
                key = info.key()
                entry = aggregates.setdefault(
                    key,
                    {
                        "info": info,
                        "packet_count": 0.0,
                        "byte_count": 0.0,
                        "duration_s": 0.0,
                    },
                )
                pkt_count = float(getattr(stat, "packet_count", 0.0) or 0.0)
                byte_count = float(getattr(stat, "byte_count", 0.0) or 0.0)
                if pkt_count <= 0.0 and byte_count <= 0.0:
                    continue
                entry["packet_count"] = float(entry["packet_count"]) + pkt_count
                entry["byte_count"] = float(entry["byte_count"]) + byte_count
                duration_s = float(getattr(stat, "duration_sec", 0.0) or 0.0) + float(getattr(stat, "duration_nsec", 0.0) or 0.0) / 1e9
                entry["duration_s"] = max(float(entry["duration_s"]), duration_s)
            except Exception as exc:
                self.logger.debug("Failed to aggregate flow stat: %s", exc)

        items = list(aggregates.values())
        qd = self.runtime.queue_depth_now()
        raw_qd = self.raw_stats_queue.qsize()
        switch_metrics = self._ensure_switch_metrics(dp.id)
        pressure = pressure_state(
            runtime_queue_depth=int(qd),
            raw_queue_depth=int(raw_qd),
            timeouts=int(switch_metrics.get("timeouts", 0)),
            avg_reply_delay_ms=float(switch_metrics.get("avg_reply_delay_ms", 0.0)),
            queue_threshold=int(self.high_pressure_threshold),
            raw_threshold=int(self.raw_stats_high_pressure_threshold),
            reply_delay_high_ms=float(self.reply_delay_high_ms),
        )
        trim_limit = 0
        if pressure == "high":
            trim_limit = int(self.max_obs_per_reply_high_pressure)
            if qd >= self.poll_backpressure_threshold or raw_qd >= self.raw_stats_queue_maxsize or int(switch_metrics.get("timeouts", 0)) > 0:
                trim_limit = int(self.max_obs_per_reply_extreme_pressure)
        elif pressure == "medium" and len(items) > int(self.max_obs_per_reply_high_pressure * 2):
            trim_limit = int(self.max_obs_per_reply_high_pressure)
        if trim_limit > 0 and len(items) > trim_limit:
            items.sort(key=self._observation_priority, reverse=True)
            dropped = len(items) - trim_limit
            self._record_poll_trim(dp.id, dropped, pressure)
            if dropped > 0 and (now - self._last_poll_log_ts) >= 5.0:
                self.logger.warning(
                    "Flow reply trim | datapath=%s | pressure=%s | kept=%s | dropped=%s | runtime_q=%s | raw_q=%s",
                    dp.id,
                    pressure,
                    trim_limit,
                    dropped,
                    qd,
                    raw_qd,
                )
                self._last_poll_log_ts = now
            items = items[: trim_limit]
        else:
            self._record_poll_trim(dp.id, 0, pressure)

        batch = RawFlowBatch(
            dpid=int(dp.id),
            poll_cycle_id=int(poll_cycle_id),
            poll_request_ts=float(poll_request_ts),
            poll_reply_ts=float(now),
            reply_part_count=int(reply_part_count),
            items=[
                RawFlowItem(
                    info=payload["info"],
                    packet_count=float(payload["packet_count"]),
                    byte_count=float(payload["byte_count"]),
                    duration_s=float(payload["duration_s"]),
                )
                for payload in items
            ],
        )
        try:
            self.raw_stats_queue.put_nowait(batch)
        except Full:
            self.raw_stats_drop_count += max(len(batch.items), 1)
            if (now - self._last_poll_log_ts) >= 5.0:
                self.logger.warning(
                    "Raw stats queue full | datapath=%s | items=%s | queue_depth=%s",
                    dp.id,
                    len(batch.items),
                    self.raw_stats_queue.qsize(),
                )
                self._last_poll_log_ts = now

        if final_reply:
            reply_delay_ms = max(0.0, (now - float(poll_request_ts)) * 1000.0)
            self._record_poll_reply(dp.id, reply_delay_ms, len(batch.items), int(reply_part_count))
        self._publish_controller_metrics()

    def _feature_worker(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.raw_stats_queue.get(timeout=1.0)
            except Empty:
                self._publish_controller_metrics()
                continue
            feature_dequeue_ts = time.time()
            try:
                for item in batch.items:
                    try:
                        info = item.info
                        if self.emit_only_on_change and not self.feature_builder.should_emit_observation(
                            info,
                            packet_count=float(item.packet_count),
                            byte_count=float(item.byte_count),
                            min_packet_delta=self.min_emit_packet_delta,
                            min_byte_delta=self.min_emit_byte_delta,
                            force_emit_after_s=self.force_emit_after_s,
                            now_ts=feature_dequeue_ts,
                        ):
                            continue
                        obs = self.feature_builder.make_observation_from_info(
                            info=info,
                            packet_count=float(item.packet_count),
                            byte_count=float(item.byte_count),
                            duration_s=float(item.duration_s),
                            now_ts=time.time(),
                            poll_request_ts=float(batch.poll_request_ts),
                            poll_reply_ts=float(batch.poll_reply_ts),
                            feature_dequeue_ts=float(feature_dequeue_ts),
                            poll_cycle_id=int(batch.poll_cycle_id),
                            reply_part_count=int(batch.reply_part_count),
                        )
                        self.runtime.enqueue(obs)
                    except Exception as exc:
                        self.logger.debug("Failed to build observation from raw batch: %s", exc)
            finally:
                try:
                    self.raw_stats_queue.task_done()
                except Exception:
                    pass
                self._publish_controller_metrics()

    def _runtime_worker(self):
        try:
            self.runtime.run_forever()
        except Exception as exc:
            self.logger.exception("Runtime worker crashed: %s", exc)

    def _react_to_alerts(self):
        last_seen_ts = 0.0
        while not self._stop.is_set():
            state = self.runtime.current_state()
            alerts = state.get("recent_alerts", []) or []
            new_alerts = []
            for alert in alerts:
                ts = float(alert.get("timestamp", 0.0))
                if ts > last_seen_ts:
                    new_alerts.append(alert)
                    last_seen_ts = max(last_seen_ts, ts)
            for alert in new_alerts:
                if not self.enable_blocking:
                    continue
                if not bool(alert.get("block", False)):
                    continue
                src_ip = str(alert.get("src_ip", "")).strip()
                if src_ip and src_ip not in self.blocked_ips:
                    self.blocked_ips.add(src_ip)
                    self.logger.warning("Blocking attack source %s", src_ip)
                    self._block_ip(src_ip)
            time.sleep(0.5)

    def _block_ip(self, ip_address: str) -> None:
        try:
            ipaddress.ip_address(ip_address)
        except Exception:
            return
        for dp in list(self.datapaths.values()):
            parser = dp.ofproto_parser
            match_src = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip_address)
            match_dst = parser.OFPMatch(eth_type=0x0800, ipv4_dst=ip_address)
            self.add_flow(dp, priority=100, match=match_src, actions=[])
            self.add_flow(dp, priority=100, match=match_dst, actions=[])

    def stop(self) -> None:
        self._stop.set()
        try:
            self.runtime.stop(timeout_s=5.0)
        except Exception:
            pass
        try:
            super().stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Ryu controller for telemetry-based SDN NIDS.")
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--bundle-path", default=str(DEFAULT_BUNDLE_DIR / 'lstm'))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--min-poll-interval", type=float, default=0.75)
    parser.add_argument("--max-poll-interval", type=float, default=3.0)
    parser.add_argument("--adaptive-polling", action="store_true")
    parser.add_argument("--enable-blocking", action="store_true")
    parser.parse_known_args()
    return None


if __name__ == "__main__":
    main()
