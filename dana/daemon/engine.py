"""Agent Engine sidecar — asyncio loopback TCP server (JSON lines).

Hosts a thin facade over the LangGraph / tool runtime for Phase 2A.
Full extraction of audio loops from ``core_agent`` is progressive; this
module prioritizes solid IPC, process isolation, and hot-restart hooks.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Awaitable

from dana.daemon.protocol import METHODS, make_error, make_event, make_result
from dana.daemon.watchdog import ProcessWatchdog, default_session_path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051

StreamChatHandler = Callable[[str, dict[str, Any]], AsyncIterator[dict[str, Any]]]
StatusProvider = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]


async def _default_stream_chat(
    message: str, _params: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Offline graph stub — emits status / token / tool events then completes."""
    yield {"event": "status", "data": {"phase": "thinking", "wake_state": "awake"}}
    await asyncio.sleep(0)
    text = f"[stub] {message}"
    yield {"event": "token", "data": {"text": text}}
    yield {
        "event": "vision",
        "data": {"summary": "offline stub — no camera frame"},
    }
    yield {
        "event": "tool",
        "data": {"name": "graph_stub", "status": "ok"},
    }


def _default_status() -> dict[str, Any]:
    cpu: float | None = None
    try:
        import psutil  # type: ignore[import-untyped]

        cpu = float(psutil.cpu_percent(interval=0.0))
    except Exception:  # noqa: BLE001
        cpu = None
    return {
        "pid": os.getpid(),
        "wake_state": "awake",
        "cpu_percent": cpu,
        "gpu_percent": None,
        "uptime_s": None,
        "engine": "sidecar",
        "version": "2a",
    }


class EngineDaemon:
    """Injectable Agent Engine IPC server (loopback TCP, JSON lines)."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        session_path: Path | str | None = None,
        stream_chat_handler: StreamChatHandler | None = None,
        status_provider: StatusProvider | None = None,
        initial_session: dict[str, Any] | None = None,
        on_hot_restart: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.session_path = Path(session_path) if session_path else default_session_path()
        self._stream_chat = stream_chat_handler or _default_stream_chat
        self._status_provider = status_provider or _default_status
        self._on_hot_restart = on_hot_restart
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._started_at = time.monotonic()
        self._closing = asyncio.Event()
        self._swap_requested = False
        self._lock = asyncio.Lock()
        self.session: dict[str, Any] = {
            "conversation": [],
            "task_state": {},
            "meta": {},
        }
        if initial_session:
            self.session.update(initial_session)
        else:
            loaded = ProcessWatchdog.load_session(self.session_path)
            if loaded:
                self.session.update(loaded)

    @property
    def swap_requested(self) -> bool:
        return self._swap_requested

    @property
    def is_serving(self) -> bool:
        return self._server is not None and self._server.is_serving()

    async def start(self) -> tuple[str, int]:
        """Bind and begin accepting connections. Returns (host, bound_port)."""
        if self._server is not None:
            socks = self._server.sockets or []
            if socks:
                addr = socks[0].getsockname()
                return str(addr[0]), int(addr[1])
            return self.host, self.port

        self._closing.clear()
        self._swap_requested = False
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
        )
        socks = self._server.sockets or []
        if not socks:
            raise RuntimeError("EngineDaemon failed to bind")
        addr = socks[0].getsockname()
        self.host = str(addr[0])
        self.port = int(addr[1])
        return self.host, self.port

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._closing.wait()
        await self._close_server()

    async def stop(self) -> None:
        self._closing.set()
        await self._close_server()

    async def _close_server(self) -> None:
        server = self._server
        self._server = None
        writers = list(self._clients)
        self._clients.clear()
        for w in writers:
            try:
                w.close()
                await w.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._clients.add(writer)
        try:
            while not self._closing.is_set():
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    await self._send(
                        writer,
                        make_error("?", "invalid JSON"),
                    )
                    continue
                if not isinstance(msg, dict):
                    await self._send(writer, make_error("?", "request must be object"))
                    continue
                await self._dispatch(writer, msg)
                if self._swap_requested:
                    break
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _dispatch(
        self,
        writer: asyncio.StreamWriter,
        msg: dict[str, Any],
    ) -> None:
        req_id = str(msg.get("id") or uuid.uuid4().hex)
        method = str(msg.get("method") or "")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        if method not in METHODS:
            await self._send(writer, make_error(req_id, f"unknown method: {method}"))
            return
        try:
            if method == "ping":
                await self._send(
                    writer,
                    make_result(req_id, {"pong": True, "pid": os.getpid()}),
                )
            elif method == "system_status":
                await self._send(writer, make_result(req_id, await self._collect_status()))
            elif method == "stream_chat":
                await self._handle_stream_chat(writer, req_id, params)
            elif method == "hot_restart":
                await self._handle_hot_restart(writer, req_id, params)
        except Exception as exc:  # noqa: BLE001
            await self._send(writer, make_error(req_id, f"{type(exc).__name__}: {exc}"))

    async def _handle_stream_chat(
        self,
        writer: asyncio.StreamWriter,
        req_id: str,
        params: dict[str, Any],
    ) -> None:
        message = str(params.get("message") or params.get("text") or "")
        conversation = self.session.setdefault("conversation", [])
        if isinstance(conversation, list):
            conversation.append({"role": "user", "content": message, "ts": time.time()})
        async for item in self._stream_chat(message, params):
            event = str(item.get("event") or "update")
            data = item.get("data") if isinstance(item.get("data"), dict) else dict(item)
            await self._send(writer, make_event(req_id, event, data))
        if isinstance(conversation, list):
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"[stub] {message}",
                    "ts": time.time(),
                }
            )
        await self._send(
            writer,
            make_result(
                req_id,
                {"done": True, "message": message, "turns": len(conversation)},
            ),
        )

    async def _handle_hot_restart(
        self,
        writer: asyncio.StreamWriter,
        req_id: str,
        params: dict[str, Any],
    ) -> None:
        async with self._lock:
            # Merge optional client-supplied task/conversation patches.
            if isinstance(params.get("conversation"), list):
                self.session["conversation"] = list(params["conversation"])
            if isinstance(params.get("task_state"), dict):
                self.session["task_state"] = dict(params["task_state"])
            self.session.setdefault("meta", {})
            if isinstance(self.session["meta"], dict):
                self.session["meta"]["hot_restart_at"] = time.time()
                self.session["meta"]["pid"] = os.getpid()
            path = ProcessWatchdog.save_session(self.session, self.session_path)
            payload = {
                "flushed": True,
                "session_path": str(path),
                "swap": True,
                "pid": os.getpid(),
            }
            await self._send(writer, make_result(req_id, payload))
            self._swap_requested = True
            if self._on_hot_restart is not None:
                try:
                    self._on_hot_restart(dict(self.session))
                except Exception:  # noqa: BLE001
                    pass
            # Close ports after the result is on the wire.
            await asyncio.sleep(0)
            self._closing.set()

    async def _collect_status(self) -> dict[str, Any]:
        raw = self._status_provider()
        if inspect.isawaitable(raw):
            data = await raw  # type: ignore[misc]
        else:
            data = raw
        if not isinstance(data, dict):
            data = {}
        out = dict(data)
        out.setdefault("pid", os.getpid())
        out.setdefault("port", self.port)
        out.setdefault("host", self.host)
        out["uptime_s"] = round(time.monotonic() - self._started_at, 3)
        out["conversation_turns"] = len(self.session.get("conversation") or [])
        return out

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, frame: dict[str, Any]) -> None:
        payload = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        writer.write(payload)
        await writer.drain()


async def run_engine_daemon(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    session_path: Path | str | None = None,
) -> None:
    daemon = EngineDaemon(host=host, port=port, session_path=session_path)
    await daemon.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dana Agent Engine sidecar daemon")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--session",
        default=None,
        help="Path to session_state.json (default: ~/.dana/session_state.json)",
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(
            run_engine_daemon(
                host=args.host,
                port=int(args.port),
                session_path=args.session,
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
