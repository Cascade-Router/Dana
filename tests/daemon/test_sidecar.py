"""Phase 2A — process-isolated sidecar daemon + UI IPC client (offline)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dana.daemon.engine import EngineDaemon
from dana.daemon.watchdog import ProcessWatchdog
from dana.schema import AgenticResult
from dana.ui.daemon_client import RECONNECT_BADGE, DaemonClient


def _fast_react_loop(**kwargs: Any) -> AgenticResult:
    """Offline stand-in for the live ReAct graph (keeps daemon tests hermetic)."""
    user = str(kwargs.get("user_text") or "")
    on_tool = kwargs.get("on_tool_start")
    if callable(on_tool):
        try:
            from dana.tools.schema import ToolCall

            on_tool(
                ToolCall(tool_id="graph_probe", arguments={}, raw_text=user, confidence=1.0),
                "working",
            )
        except Exception:  # noqa: BLE001
            pass
    return AgenticResult(
        final_text=f"live:{user}",
        iterations=1,
        tool_trace=[{"tool": "graph_probe", "observation": "ok"}],
        reply_lang="en",
        had_errors=False,
    )


def _fast_ask_ollama_messages(messages: Any, model: str = "", **_kwargs: Any) -> str:
    """Offline stand-in for the lightweight-chat LLM call.

    Greetings/small talk (e.g. "ping") route through ``run_lightweight_chat``
    instead of the ReAct graph (see ``dana.agentic.requires_tool_graph``), so
    patching only ``run_react_loop`` never exercises that path — this mock
    keeps it hermetic too.
    """
    user_msgs = [m for m in (messages or []) if m.get("role") == "user"]
    text = str(user_msgs[-1].get("content") or "") if user_msgs else ""
    return f"live:{text}"


class _DaemonThread:
    """Run ``EngineDaemon.serve_forever`` on a background asyncio loop."""

    def __init__(self, daemon: EngineDaemon) -> None:
        self.daemon = daemon
        self.host = daemon.host
        self.port = 0
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> tuple[str, int]:
        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _boot() -> None:
                try:
                    host, port = await self.daemon.start()
                    self.host = host
                    self.port = port
                    self._ready.set()
                    await self.daemon.serve_forever()
                except BaseException as exc:  # noqa: BLE001
                    self._error = exc
                    self._ready.set()

            try:
                loop.run_until_complete(_boot())
            finally:
                try:
                    loop.close()
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(target=_run, name="TestEngineDaemon", daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "daemon failed to bind"
        if self._error is not None:
            raise RuntimeError(f"daemon boot failed: {self._error}")
        assert self.port > 0
        return self.host, self.port

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.daemon.stop(), loop)
            try:
                fut.result(timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)


@pytest.fixture
def session_path(tmp_path: Path) -> Path:
    return tmp_path / "session_state.json"


def test_ui_client_connects_and_streams(session_path: Path) -> None:
    """UI IPC client connects and receives streaming graph/tool events."""
    daemon = EngineDaemon(host="127.0.0.1", port=0, session_path=session_path)
    runner = _DaemonThread(daemon)
    host, port = runner.start()
    client = DaemonClient(
        host=host,
        port=port,
        reconnect_delay_s=0.05,
        request_timeout_s=30.0,
    )
    try:
        assert client.connect(retries=3) is True
        assert client.connected is True

        with patch(
            "dana.agentic.run_react_loop", side_effect=_fast_react_loop
        ), patch(
            "dana.core_agent.ask_ollama_messages",
            side_effect=_fast_ask_ollama_messages,
        ):
            frames = list(client.stream_chat("hello sidecar"))
            assert frames, "expected streaming frames"
            types = [f.get("type") for f in frames]
            assert "event" in types
            assert types[-1] == "result"
            events = {f.get("event") for f in frames if f.get("type") == "event"}
            assert "token" in events
            assert "tool" in events or "status" in events
            text = client.stream_chat_text("ping")
            assert "live:ping" in text
            assert "[stub]" not in text

        status = client.system_status()
        assert status.get("pid")
        assert status.get("engine") == "sidecar"
    finally:
        client.close()
        runner.stop()


def test_hot_restart_serializes_and_recovers(session_path: Path) -> None:
    """Simulate /hot_restart → state serialization + clean daemon recovery."""
    pid_path = session_path.parent / "engine.pid"
    daemon = EngineDaemon(
        host="127.0.0.1",
        port=0,
        session_path=session_path,
        initial_session={
            "conversation": [{"role": "user", "content": "prior"}],
            "task_state": {"goal": "phase2a"},
            "meta": {},
        },
    )
    runner = _DaemonThread(daemon)
    host, port = runner.start()

    states: list[tuple[str, str]] = []
    client = DaemonClient(
        host=host,
        port=port,
        reconnect_delay_s=0.05,
        on_state_change=lambda s, b: states.append((s, b)),
    )
    assert client.connect(retries=3) is True
    list(client.stream_chat("before restart"))

    result = client.hot_restart(task_state={"goal": "phase2a", "step": 2})
    assert result.get("ok") is True
    assert result.get("flushed") is True
    assert Path(str(result["session_path"])).is_file()
    assert any(s == "reconnecting" and b == RECONNECT_BADGE for s, b in states)

    # Wait for first daemon to finish closing ports.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and runner._thread and runner._thread.is_alive():
        if daemon.swap_requested and not daemon.is_serving:
            break
        time.sleep(0.05)
    runner.stop()

    saved = ProcessWatchdog.load_session(session_path)
    assert saved.get("task_state", {}).get("goal") == "phase2a"
    assert len(saved.get("conversation") or []) >= 1

    # Watchdog launches a replacement engine that restores the session file.
    launched: dict[str, Any] = {"runner": None}

    def _launch() -> int:
        last_err: BaseException | None = None
        for _ in range(40):
            try:
                nxt = EngineDaemon(
                    host="127.0.0.1",
                    port=port,  # same port — UI reconnects without reconfiguration
                    session_path=session_path,
                )
                nxt_runner = _DaemonThread(nxt)
                nxt_runner.start()
                launched["runner"] = nxt_runner
                launched["daemon"] = nxt
                return 42  # synthetic PID for injectable watchdog
            except Exception as exc:  # noqa: BLE001 — port still releasing
                last_err = exc
                time.sleep(0.05)
        raise RuntimeError(f"recovery daemon failed to bind: {last_err}")

    killed: list[int] = []
    # Synthetic prior engine PID (must not be this pytest process).
    prior_pid = 424242

    wd = ProcessWatchdog(
        session_path=session_path,
        pid_path=pid_path,
        pid=prior_pid,
        launch_fn=_launch,
        kill_fn=lambda p: killed.append(int(p)),
    )
    # Treat synthetic PID as alive once so kill_fn is exercised, then gone.
    alive = {"n": 1}

    def _fake_alive(pid: int | None = None) -> bool:
        target = prior_pid if pid is None else int(pid)
        if target != prior_pid:
            return False
        if alive["n"] > 0:
            alive["n"] -= 1
            return True
        return False

    wd.is_alive = _fake_alive  # type: ignore[method-assign]
    # State already on disk; restart launches the recovery daemon.
    report = wd.hot_update_restart(saved, wait_dead=True)
    assert report["ok"] is True
    assert report["new_pid"] == 42
    assert report["restored_turns"] >= 1
    assert killed == [prior_pid], "watchdog must terminate prior engine pid"

    # Client reconnects within ~500ms and sees restored session telemetry.
    client.start_auto_reconnect()
    ok = False
    for _ in range(40):
        if client.connect(retries=1):
            ok = True
            break
        time.sleep(0.05)
    assert ok, "client failed to reconnect after hot_restart"
    status = client.system_status()
    assert int(status.get("conversation_turns") or 0) >= 1
    assert client.state == "connected"
    assert client.badge_text == ""

    client.close()
    nxt_runner = launched.get("runner")
    if nxt_runner is not None:
        nxt_runner.stop()


def test_client_degrades_when_daemon_unavailable() -> None:
    """Headless / CI path — no freeze, clear error frame when daemon is down."""
    client = DaemonClient(host="127.0.0.1", port=59999, connect_timeout_s=0.1)
    assert client.connect(retries=1) is False
    frames = list(client.stream_chat("noop"))
    assert frames
    assert frames[-1].get("ok") is False
    assert "unavailable" in str(frames[-1].get("error") or "").lower() or frames[
        -1
    ].get("type") == "error"
    client.close()


def test_session_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sess.json"
    state = {
        "conversation": [{"role": "user", "content": "hi"}],
        "task_state": {"n": 1},
        "meta": {"v": 1},
    }
    ProcessWatchdog.save_session(state, path)
    loaded = ProcessWatchdog.load_session(path)
    assert loaded["conversation"][0]["content"] == "hi"
    assert loaded["task_state"]["n"] == 1

