"""Control Dashboard IPC client for the Agent Engine sidecar.

Connects to ``localhost:50051`` (JSON lines over TCP). Auto-reconnects within
~500ms after a hot-restart without freezing the UI. Degrades gracefully when
the daemon is unavailable (headless tests / offline CI).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051
RECONNECT_BADGE = "[RECONNECTING DAEMON...]"

ConnectionState = Literal["connected", "reconnecting", "disconnected", "unavailable"]
StateCallback = Callable[[ConnectionState, str], None]


def daemon_ipc_enabled() -> bool:
    """Opt-in IPC attach for the GUI (default on when unset is False for tests).

    Set ``DONNA_ENGINE_DAEMON=1`` to auto-connect from the Control Dashboard.
    Tests may inject a client directly without flipping the env flag.
    """
    raw = (os.environ.get("DONNA_ENGINE_DAEMON") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class DaemonClient:
    """Synchronous, thread-safe JSON-lines client with auto-reconnect."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    reconnect_delay_s: float = 0.5
    connect_timeout_s: float = 0.35
    request_timeout_s: float = 10.0
    on_state_change: StateCallback | None = None
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _state: ConnectionState = field(default="disconnected", init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _reconnector: threading.Thread | None = field(default=None, init=False, repr=False)
    _badge: str = field(default="", init=False)

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def badge_text(self) -> str:
        return self._badge

    @property
    def connected(self) -> bool:
        return self._state == "connected" and self._sock is not None

    def _set_state(self, state: ConnectionState, badge: str = "") -> None:
        self._state = state
        self._badge = badge
        cb = self.on_state_change
        if cb is not None:
            try:
                cb(state, badge)
            except Exception:  # noqa: BLE001
                pass

    def connect(self, *, retries: int = 1) -> bool:
        """Attempt connection. Returns False without raising when unavailable."""
        with self._lock:
            if self._sock is not None:
                return True
            attempts = max(1, int(retries))
            last_exc: Exception | None = None
            for i in range(attempts):
                try:
                    sock = socket.create_connection(
                        (self.host, int(self.port)),
                        timeout=self.connect_timeout_s,
                    )
                    sock.settimeout(self.request_timeout_s)
                    self._sock = sock
                    self._set_state("connected", "")
                    return True
                except OSError as exc:
                    last_exc = exc
                    if i + 1 < attempts:
                        time.sleep(self.reconnect_delay_s)
            self._sock = None
            # Quiet degrade — not an error for headless tests.
            self._set_state("unavailable" if last_exc else "disconnected", "")
            return False

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            sock = self._sock
            self._sock = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._set_state("disconnected", "")

    def start_auto_reconnect(self) -> None:
        """Background loop: keep trying to (re)connect every ``reconnect_delay_s``."""
        if self._reconnector is not None and self._reconnector.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            saw_connected = self._state == "connected"
            while not self._stop.is_set():
                if self.connected:
                    saw_connected = True
                    time.sleep(self.reconnect_delay_s)
                    continue
                # Badge only after a live session drops (hot-update / crash).
                if saw_connected or self._state == "reconnecting":
                    self._set_state("reconnecting", RECONNECT_BADGE)
                ok = self.connect(retries=1)
                if ok:
                    saw_connected = True
                    self._set_state("connected", "")
                elif saw_connected:
                    self._set_state("reconnecting", RECONNECT_BADGE)
                self._stop.wait(self.reconnect_delay_s)

        self._reconnector = threading.Thread(
            target=_loop,
            name="DanaDaemonReconnect",
            daemon=True,
        )
        self._reconnector.start()

    def stop_auto_reconnect(self) -> None:
        self._stop.set()

    def _drop_socket(self) -> None:
        with self._lock:
            sock = self._sock
            self._sock = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _transact(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        _attempt: int = 0,
    ) -> list[dict[str, Any]]:
        """Send one request; collect event frames until a result/error."""
        req_id = uuid.uuid4().hex
        frame = {
            "id": req_id,
            "method": method,
            "params": dict(params or {}),
        }
        line = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        out: list[dict[str, Any]] = []

        with self._lock:
            if self._sock is None and not self.connect(retries=1):
                return [
                    {
                        "id": req_id,
                        "ok": False,
                        "type": "error",
                        "error": "daemon unavailable",
                        "data": {},
                    }
                ]
            sock = self._sock
            assert sock is not None
            try:
                sock.sendall(line)
                while True:
                    raw = self._recv_line(sock)
                    if raw is None:
                        raise ConnectionError("daemon closed connection")
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("id") or "") not in {"", req_id}:
                        continue
                    out.append(msg)
                    if msg.get("type") in {"result", "error"}:
                        break
            except (OSError, ConnectionError, json.JSONDecodeError, TimeoutError):
                self._drop_socket()
                self._set_state("reconnecting", RECONNECT_BADGE)
                if _attempt >= 1:
                    return [
                        {
                            "id": req_id,
                            "ok": False,
                            "type": "error",
                            "error": "daemon reconnect failed",
                            "data": {},
                        }
                    ]
                # One fast reconnect + single retry (hot-update path ~500ms).
                time.sleep(self.reconnect_delay_s)
                if not self.connect(retries=1):
                    return [
                        {
                            "id": req_id,
                            "ok": False,
                            "type": "error",
                            "error": "daemon reconnect failed",
                            "data": {},
                        }
                    ]
                return self._transact(method, params, _attempt=_attempt + 1)
        return out

    @staticmethod
    def _recv_line(sock: socket.socket) -> str | None:
        buf = bytearray()
        while True:
            chunk = sock.recv(1)
            if not chunk:
                return None if not buf else buf.decode("utf-8", errors="replace")
            if chunk == b"\n":
                return buf.decode("utf-8", errors="replace")
            buf.extend(chunk)
            if len(buf) > 4_000_000:
                raise ConnectionError("IPC frame too large")

    def ping(self) -> dict[str, Any]:
        frames = self._transact("ping")
        return frames[-1] if frames else {"ok": False, "error": "no response"}

    def system_status(self) -> dict[str, Any]:
        frames = self._transact("system_status")
        last = frames[-1] if frames else {"ok": False, "type": "error", "error": "no response"}
        if last.get("type") == "result":
            return dict(last.get("data") or {})
        return {"ok": False, "error": last.get("error") or "status failed"}

    def hot_restart(
        self,
        *,
        conversation: list[Any] | None = None,
        task_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if conversation is not None:
            params["conversation"] = conversation
        if task_state is not None:
            params["task_state"] = task_state
        frames = self._transact("hot_restart", params)
        last = frames[-1] if frames else {"ok": False, "error": "no response"}
        # Expect disconnect after swap — mark reconnecting for the UI badge.
        self._drop_socket()
        self._set_state("reconnecting", RECONNECT_BADGE)
        if last.get("type") == "result":
            data = dict(last.get("data") or {})
            data["ok"] = True
            return data
        return {"ok": False, "error": last.get("error") or "hot_restart failed"}

    def stream_chat(self, message: str, **extra: Any) -> Iterator[dict[str, Any]]:
        """Yield event/result frames for a chat turn."""
        params = {"message": message}
        params.update(extra)
        frames = self._transact("stream_chat", params)
        yield from frames

    def stream_chat_text(self, message: str) -> str:
        """Convenience: concatenate token events into a single string."""
        parts: list[str] = []
        for frame in self.stream_chat(message):
            if frame.get("type") == "event" and frame.get("event") == "token":
                data = frame.get("data") or {}
                if isinstance(data, dict) and data.get("text"):
                    parts.append(str(data["text"]))
            if frame.get("type") == "error":
                return f"[daemon error] {frame.get('error')}"
        return "".join(parts)
