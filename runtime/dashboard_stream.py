from __future__ import annotations

import json
import socket
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any


def _truthy_env_value(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class RuntimeDashboardPublisher:
    def __init__(
        self,
        enabled: bool = True,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_payload_bytes: int = 64000,
    ) -> None:
        self.enabled = bool(enabled)
        self.host = str(host)
        self.port = int(port)
        self.max_payload_bytes = int(max_payload_bytes)
        self.drop_count = 0
        self._sock: socket.socket | None = None
        if self.enabled:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setblocking(False)

    def _send(self, event: str, run_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self._sock is None:
            return
        message = {
            "event": str(event),
            "run_id": str(run_id),
            "payload": payload,
            "sent_ts": time.time(),
        }
        try:
            raw = json.dumps(message, separators=(",", ":")).encode("utf-8")
            if len(raw) > self.max_payload_bytes:
                self.drop_count += 1
                return
            self._sock.sendto(raw, (self.host, self.port))
        except Exception:
            self.drop_count += 1

    def publish_state(self, run_id: str, payload: dict[str, Any]) -> None:
        self._send("state", run_id, payload)

    def publish_alert(self, run_id: str, payload: dict[str, Any]) -> None:
        self._send("alert", run_id, payload)


class DashboardEventHub:
    def __init__(self, runtime_root: str | Path, alerts_limit: int = 200, subscriber_queue_size: int = 256) -> None:
        self.runtime_root = Path(runtime_root)
        self.alerts_limit = int(alerts_limit)
        self.subscriber_queue_size = int(subscriber_queue_size)
        self._lock = threading.Lock()
        self._latest_state: dict[str, dict[str, Any]] = {}
        self._recent_alerts: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.alerts_limit))
        self._subscribers: dict[str, set[Queue]] = defaultdict(set)
        self._all_subscribers: set[Queue] = set()
        self._listener_sock: socket.socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def list_runs(self) -> list[str]:
        runs = set(self._latest_state.keys()) | set(self._recent_alerts.keys())
        if self.runtime_root.exists():
            runs |= {item.name for item in self.runtime_root.iterdir() if item.is_dir()}
        return sorted(r for r in runs if r)

    def get_state(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_state.get(run_id, {}))

    def get_alerts(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._recent_alerts.get(run_id, deque()))
        return rows[-int(limit):]

    def subscribe(self, run_id: str) -> Queue:
        q: Queue = Queue(maxsize=self.subscriber_queue_size)
        with self._lock:
            self._subscribers[run_id].add(q)
        return q

    def unsubscribe(self, run_id: str, q: Queue) -> None:
        with self._lock:
            bucket = self._subscribers.get(run_id)
            if bucket is not None:
                bucket.discard(q)
                if not bucket:
                    self._subscribers.pop(run_id, None)
            self._all_subscribers.discard(q)

    def _queue_put(self, q: Queue, item: dict[str, Any]) -> None:
        try:
            q.put_nowait(item)
            return
        except Full:
            pass
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass

    def _broadcast(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        message = {"event": event, "run_id": run_id, "payload": payload}
        with self._lock:
            subscribers = list(self._subscribers.get(run_id, set()))
        for q in subscribers:
            self._queue_put(q, message)

    def ingest(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", "")).strip().lower()
        run_id = str(message.get("run_id", "")).strip()
        payload = message.get("payload")
        if not run_id or not isinstance(payload, dict):
            return
        if event == "state":
            with self._lock:
                self._latest_state[run_id] = dict(payload)
            self._broadcast(run_id, "state", dict(payload))
            return
        if event == "alert":
            with self._lock:
                self._recent_alerts[run_id].append(dict(payload))
                rows = list(self._recent_alerts[run_id])[-20:]
            self._broadcast(run_id, "alerts", {"rows": rows})

    def start_udp_listener(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        if self._listener_thread is not None:
            return
        self._listener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listener_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener_sock.bind((host, int(port)))
        self._listener_sock.settimeout(1.0)
        self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self._listener_thread.start()

    def _listener_loop(self) -> None:
        assert self._listener_sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._listener_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if isinstance(message, dict):
                self.ingest(message)

    def close(self) -> None:
        self._stop_event.set()
        if self._listener_sock is not None:
            try:
                self._listener_sock.close()
            except Exception:
                pass
            self._listener_sock = None
