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
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Awaitable

from dana.daemon.protocol import METHODS, make_error, make_event, make_result
from dana.daemon.watchdog import ProcessWatchdog, default_session_path
from dana.logging import enable_runtime_file_logging, log

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051

StreamChatHandler = Callable[[str, dict[str, Any]], AsyncIterator[dict[str, Any]]]
StatusProvider = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]


def _daemon_system_prompt(user_text: str) -> str:
    """Build live agent system prompt; fall back if core_agent is unavailable."""
    try:
        from dana.core_agent import build_donna_system_prompt

        return build_donna_system_prompt([], user_text=user_text)
    except Exception as exc:  # noqa: BLE001
        log("Daemon", f"system prompt fallback ({type(exc).__name__}: {exc})")
        return (
            "You are Dānā, a local voice agent with tools. "
            "Prefer concise spoken answers. Use tools when helpful."
        )


def _daemon_execute_tool(tc: Any) -> str:
    """Dispatch through the live tool executor (lazy import)."""
    from dana.core_agent import execute_tool_call

    return execute_tool_call(tc)


async def _default_stream_chat(
    message: str, params: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Thin IPC adapter: lightweight chat fast-path or ReAct / LangGraph."""
    yield {"event": "status", "data": {"phase": "thinking", "wake_state": "awake"}}
    yield {
        "event": "STATE_CHANGE",
        "data": {"status": "routing", "message": "Supervisor Routing..."},
    }
    await asyncio.sleep(0)
    log("Daemon", f"stream_chat start chars={len(message)}")

    from dana.agentic import (
        REACT_MAX_ITERS,
        requires_tool_graph,
        run_lightweight_chat,
        run_react_loop,
    )

    tool_starts: list[dict[str, Any]] = []

    def _on_tool_start(tc: Any, phrase: str = "") -> None:
        name = getattr(tc, "tool_id", None) or getattr(tc, "name", None) or "tool"
        tool_starts.append(
            {"name": str(name), "status": "running", "ack": str(phrase or "")}
        )
        try:
            from dana.ui.status_bus import emit_state_change

            emit_state_change("executing", tool=str(name))
        except Exception:  # noqa: BLE001
            pass

    model = str(params.get("model") or os.environ.get("DONNA_OLLAMA_MODEL") or "llama3.2")
    prior = params.get("prior_messages")
    if not isinstance(prior, list):
        prior = None
    visual = params.get("visual_context")
    if visual is not None:
        visual = str(visual) or None
    max_iters = int(params.get("max_iters") or REACT_MAX_ITERS)
    enable_reflection = bool(params.get("enable_reflection", False))

    # Greetings / small-talk bypass LangGraph instantly.
    use_graph = bool(requires_tool_graph(message))
    log(
        "Daemon",
        f"stream_chat route={'langgraph' if use_graph else 'lightweight'} "
        f"chars={len(message)}",
    )

    def _run_lightweight() -> Any:
        from dana.core_agent import ask_ollama_messages

        return run_lightweight_chat(
            user_text=message,
            ask_fn=ask_ollama_messages,
            model=model,
            visual_context=visual,
            prior_messages=prior,
        )

    def _run_live_graph() -> Any:
        return run_react_loop(
            user_text=message,
            system_prompt=_daemon_system_prompt(message),
            execute_fn=_daemon_execute_tool,
            max_iters=max_iters,
            enable_reflection=enable_reflection,
            prior_messages=prior,
            on_tool_start=_on_tool_start,
            visual_context=visual,
            model=model,
            tts_callback=None,
        )

    worker = _run_live_graph if use_graph else _run_lightweight
    route_name = "react_graph" if use_graph else "lightweight_chat"

    # Heartbeat status events so IPC clients (default 10s sock timeout) stay alive.
    graph_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            done, _pending = await asyncio.wait({graph_task}, timeout=2.0)
            if graph_task in done:
                break
            yield {
                "event": "status",
                "data": {"phase": "thinking", "wake_state": "awake"},
            }
        result = graph_task.result()
    except Exception as exc:  # noqa: BLE001 — keep daemon alive if graph/LLM missing
        if not graph_task.done():
            graph_task.cancel()
        log("Daemon", f"stream_chat {route_name} error: {type(exc).__name__}: {exc}")
        yield {
            "event": "token",
            "data": {"text": f"I hit a graph error: {type(exc).__name__}: {exc}"},
        }
        yield {
            "event": "vision",
            "data": {"summary": str(visual) if visual else "no camera frame"},
        }
        yield {
            "event": "tool",
            "data": {
                "name": route_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
        return

    for start in tool_starts:
        yield {"event": "tool", "data": start}
        yield {
            "event": "STATE_CHANGE",
            "data": {
                "status": "executing",
                "tool": str(start.get("name") or "tool"),
            },
        }

    emitted_tools = {str(t.get("name") or "") for t in tool_starts}
    for entry in list(getattr(result, "tool_trace", None) or []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("tool") or entry.get("name") or "")
        if not name:
            if entry.get("error"):
                emitted_tools.add("react_graph")
                yield {
                    "event": "tool",
                    "data": {
                        "name": "react_graph",
                        "status": "error",
                        "error": str(entry.get("error")),
                    },
                }
            continue
        if name in emitted_tools:
            continue
        emitted_tools.add(name)
        status = "error" if entry.get("error") else "ok"
        payload: dict[str, Any] = {"name": name, "status": status}
        if entry.get("error"):
            payload["error"] = str(entry["error"])
        yield {"event": "tool", "data": payload}

    text = str(getattr(result, "final_text", None) or "").strip()
    if not text:
        text = "I finished the turn but had nothing to say."
    yield {"event": "token", "data": {"text": text}}
    yield {
        "event": "vision",
        "data": {"summary": str(visual) if visual else "no camera frame"},
    }
    if not emitted_tools:
        yield {
            "event": "tool",
            "data": {
                "name": "react_graph",
                "status": "ok" if not getattr(result, "had_errors", False) else "error",
                "iterations": int(getattr(result, "iterations", 0) or 0),
            },
        }
    yield {"event": "STATE_CHANGE", "data": {"status": "idle"}}
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("idle")
    except Exception:  # noqa: BLE001
        pass
    log("Daemon", f"stream_chat done chars={len(text)} iters={getattr(result, 'iterations', 0)}")


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
        assistant_parts: list[str] = []
        async for item in self._stream_chat(message, params):
            event = str(item.get("event") or "update")
            data = item.get("data") if isinstance(item.get("data"), dict) else dict(item)
            if event == "token" and isinstance(data, dict) and data.get("text"):
                assistant_parts.append(str(data["text"]))
            await self._send(writer, make_event(req_id, event, data))
        if isinstance(conversation, list):
            conversation.append(
                {
                    "role": "assistant",
                    "content": "".join(assistant_parts) or message,
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
    t0 = time.perf_counter()
    enable_runtime_file_logging()
    log("Daemon", f"engine boot host={host} port={port} pid={os.getpid()}")
    daemon = EngineDaemon(host=host, port=port, session_path=session_path)
    bound_host, bound_port = await daemon.start()
    try:
        from dana.perf import log_perf

        log_perf(
            "daemon_startup",
            (time.perf_counter() - t0) * 1000.0,
            host=bound_host,
            port=bound_port,
            pid=os.getpid(),
        )
    except Exception:  # noqa: BLE001
        pass
    log(
        "Daemon",
        f"engine listening host={bound_host} port={bound_port} pid={os.getpid()}",
    )

    def _warmup_audio_probe() -> None:
        """~50ms open+close InputStream so PortAudio/Silero buffers allocate early."""
        t_audio = time.perf_counter()
        try:
            import sounddevice as sd

            # Default input; short read forces driver + ring-buffer alloc.
            frames = max(1, int(16000 * 0.05))
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=min(512, frames),
            )
            stream.start()
            try:
                stream.read(frames)
            finally:
                stream.stop()
                stream.close()
            elapsed_ms = (time.perf_counter() - t_audio) * 1000.0
            try:
                from dana.perf import log_perf

                log_perf("audio_warmup", elapsed_ms, phase="complete")
            except Exception:  # noqa: BLE001
                pass
            log("Daemon", f"audio_warmup=complete ({elapsed_ms:.0f} ms)")
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t_audio) * 1000.0
            try:
                from dana.perf import log_perf

                log_perf(
                    "audio_warmup",
                    elapsed_ms,
                    phase="complete",
                    error=type(exc).__name__,
                )
            except Exception:  # noqa: BLE001
                pass
            log(
                "Daemon",
                f"audio_warmup=complete (skipped {type(exc).__name__}: {exc})",
            )

    def _warmup_llm_background() -> None:
        """Tiny silent ping so Ollama weights land in VRAM (non-blocking).

        Runs the ~50ms mic probe in the same warm-up thread before the LLM ping
        so PortAudio/Silero buffers allocate without delaying serve_forever.
        """
        _warmup_audio_probe()
        t_warm = time.perf_counter()
        try:
            from dana.perf import log_perf

            log_perf("llm_warmup", 0.0, phase="start")
        except Exception:  # noqa: BLE001
            pass
        log("Daemon", "LLM warm-up start (background ping)")
        try:
            from dana.core_agent import ask_ollama_messages

            ask_ollama_messages(
                [{"role": "user", "content": "ping"}],
                num_predict=1,
            )
            elapsed_ms = (time.perf_counter() - t_warm) * 1000.0
            try:
                from dana.perf import log_perf

                log_perf("llm_warmup", elapsed_ms, phase="done")
            except Exception:  # noqa: BLE001
                pass
            log("Daemon", f"LLM warm-up done ({elapsed_ms:.0f} ms)")
        except Exception as exc:  # noqa: BLE001
            log("Daemon", f"LLM warm-up skipped ({type(exc).__name__}: {exc})")

    threading.Thread(
        target=_warmup_llm_background,
        name="DaemonOllamaWarmup",
        daemon=True,
    ).start()
    await daemon.serve_forever()


def main(argv: list[str] | None = None) -> int:
    enable_runtime_file_logging()
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
