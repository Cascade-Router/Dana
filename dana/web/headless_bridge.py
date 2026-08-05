"""Headless Meta-Broker bridge for Gradio / Hugging Face Spaces (no Tkinter).

Wraps ``run_meta_broker_isolated`` + IPC telemetry so the web UI can submit
prompts on a background thread and poll events without blocking the Gradio
mainloop.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any

# Force headless before any optional Dana imports that might probe GUI flags.
os.environ.setdefault("DONNA_NO_GUI", "1")
os.environ.setdefault("DONNA_HEADLESS", "1")
os.environ.setdefault("DONNA_SKIP_BOOT_READY", "1")

_STATUS_IDLE = "idle"
_STATUS_LISTENING = "listening"
_STATUS_PROCESSING = "processing"

_VALID_STATUS = frozenset({_STATUS_IDLE, _STATUS_LISTENING, _STATUS_PROCESSING})


def _ensure_headless_env() -> None:
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    os.environ.setdefault("DONNA_FORCE_LOCAL", "1")
    os.environ.setdefault("DONNA_OLLAMA_KEEP_ALIVE", "0")
    os.environ.setdefault("DONNA_SKIP_RAM_BREAKER", "1")
    os.environ.setdefault("DONNA_META_BROKER_TIMEOUT_S", "600")
    if not (os.environ.get("DONNA_META_BROKER_LOG") or "").strip():
        os.environ["DONNA_META_BROKER_LOG"] = "logs/hf_space_meta_broker.log"


def status_label(status: str) -> str:
    st = str(status or "idle").strip().lower()
    if st == _STATUS_LISTENING:
        return "● Listening"
    if st in {_STATUS_PROCESSING, "routing", "executing"}:
        return "● Processing"
    return "● Idle"


class HeadlessBrokerBridge:
    """Process-wide singleton: one Meta-Broker job + telemetry fan-out queue."""

    _instance: HeadlessBrokerBridge | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._telemetry: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._status = _STATUS_IDLE
        self._log: list[str] = []
        self._result: dict[str, Any] | None = None
        self._error: str = ""
        self._prompt: str = ""
        self._started_at = 0.0
        self._finished_at = 0.0

    @classmethod
    def instance(cls) -> HeadlessBrokerBridge:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def status(self) -> str:
        with self._lock:
            return self._status

    def log_text(self, *, max_lines: int = 200) -> str:
        with self._lock:
            lines = self._log[-max(1, int(max_lines)) :]
        return "\n".join(lines) if lines else "(no telemetry yet)"

    def result(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._result) if isinstance(self._result, dict) else None

    def error(self) -> str:
        with self._lock:
            return self._error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "status_label": status_label(self._status),
                "running": self._running,
                "prompt": self._prompt,
                "error": self._error,
                "log_lines": list(self._log[-80:]),
                "result_status": (
                    str((self._result or {}).get("status") or "")
                    if self._result
                    else ""
                ),
            }

    def drain_telemetry(self, *, max_items: int = 64) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_items))):
            try:
                out.append(self._telemetry.get_nowait())
            except queue.Empty:
                break
        return out

    def _set_status(self, status: str) -> None:
        st = str(status or _STATUS_IDLE).strip().lower()
        if st not in _VALID_STATUS:
            if st in {"routing", "executing", "planning"}:
                st = _STATUS_PROCESSING
            else:
                st = _STATUS_IDLE
        with self._lock:
            self._status = st

    def _append_log(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        with self._lock:
            self._log.append(text)
            if len(self._log) > 400:
                self._log = self._log[-400:]

    def _push_event(self, event: dict[str, Any]) -> None:
        payload = dict(event or {})
        try:
            self._telemetry.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._telemetry.get_nowait()
            except queue.Empty:
                pass
            try:
                self._telemetry.put_nowait(payload)
            except queue.Full:
                pass
        msg = str(payload.get("message") or payload.get("error") or "").strip()
        phase = str(payload.get("phase") or "")
        status = str(payload.get("status") or "")
        kind = str(payload.get("type") or "telemetry")
        bits = [f"[{kind}]"]
        if phase:
            bits.append(f"phase={phase}")
        if status:
            bits.append(f"status={status}")
        if msg:
            bits.append(msg[:240])
        self._append_log(" ".join(bits))
        if kind == "telemetry" and not payload.get("terminal"):
            self._set_status(_STATUS_PROCESSING)
        elif payload.get("terminal") or kind == "result":
            self._set_status(_STATUS_IDLE)

    def submit(self, prompt: str) -> tuple[bool, str]:
        """Start Meta-Broker on a daemon thread. Returns ``(ok, note)``."""
        text = str(prompt or "").strip()
        if not text:
            return False, "Empty prompt."
        with self._lock:
            if self._running:
                return False, "Meta-Broker already running — wait for completion."
            self._running = True
            self._error = ""
            self._result = None
            self._prompt = text
            self._started_at = time.time()
            self._finished_at = 0.0
            self._status = _STATUS_PROCESSING
            self._log.append(f"[ui] submitted prompt chars={len(text)}")
        self._thread = threading.Thread(
            target=self._worker,
            args=(text,),
            name="HFHeadlessBroker",
            daemon=True,
        )
        self._thread.start()
        return True, "Meta-Broker started (isolated process)."

    def _worker(self, prompt: str) -> None:
        _ensure_headless_env()
        try:
            from dana.graph.artifact_manifest import META_BROKER_STDLIB_RULE
            from dana.graph.meta_broker_process import (
                run_meta_broker_isolated,
                start_headless_telemetry_drainer,
            )
            from dana.graph.task_tracker import emit_meta_broker_telemetry
        except Exception as exc:  # noqa: BLE001
            self._push_event(
                {
                    "type": "telemetry",
                    "status": "failed",
                    "message": f"headless import failed: {exc}",
                    "terminal": True,
                }
            )
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._running = False
                self._status = _STATUS_IDLE
                self._finished_at = time.time()
            return

        try:
            start_headless_telemetry_drainer(
                log_path=os.environ.get("DONNA_META_BROKER_LOG")
            )
        except Exception:  # noqa: BLE001
            pass

        macro = prompt
        if not prompt.lower().lstrip().startswith("/broker"):
            # Natural chat → Meta-Broker epic shell when user didn't prefix.
            if "epic " not in prompt.lower():
                macro = f"/broker {prompt}"
        if META_BROKER_STDLIB_RULE not in macro:
            macro = f"{META_BROKER_STDLIB_RULE}\n\n{macro}"

        def _on_event(event: dict[str, Any]) -> None:
            self._push_event(event)
            if str(event.get("type") or "") != "telemetry":
                return
            try:
                emit_meta_broker_telemetry(
                    task_id="meta_broker",
                    prompt=prompt,
                    phase=str(event.get("phase") or ""),
                    status=str(event.get("status") or ""),
                    message=str(event.get("message") or ""),
                    epic_title=str(event.get("epic_title") or ""),
                    terminal=bool(event.get("terminal")),
                )
            except Exception:  # noqa: BLE001
                pass

        self._push_event(
            {
                "type": "telemetry",
                "phase": "start",
                "status": "planning",
                "message": "Spawning isolated Meta-Broker process…",
            }
        )
        try:
            timeout_s = float(os.environ.get("DONNA_META_BROKER_TIMEOUT_S") or "600")
        except (TypeError, ValueError):
            timeout_s = 600.0
        try:
            result = run_meta_broker_isolated(
                macro,
                on_event=_on_event,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "final_response": f"Meta-Broker failed: {exc}",
            }
            self._push_event(
                {
                    "type": "telemetry",
                    "status": "failed",
                    "message": str(result["error"]),
                    "terminal": True,
                }
            )

        with self._lock:
            self._result = dict(result or {})
            self._error = str((result or {}).get("error") or "")
            self._running = False
            self._status = _STATUS_IDLE
            self._finished_at = time.time()
        final_msg = str(
            (result or {}).get("final_response")
            or (result or {}).get("error")
            or (result or {}).get("status")
            or "done"
        )
        self._push_event(
            {
                "type": "result",
                "status": str((result or {}).get("status") or ""),
                "message": final_msg[:400],
                "terminal": True,
            }
        )

    def task_tracker_text(self) -> str:
        """Human-readable Task Tracker snapshot (shared in-process tracker)."""
        try:
            from dana.graph.task_tracker import get_shared_task_tracker

            tracker = get_shared_task_tracker()
            activities = tracker.list_activities(limit=24)
        except Exception as exc:  # noqa: BLE001
            return f"(task tracker unavailable: {exc})"
        if not activities:
            return "(no active tasks)"
        lines: list[str] = []
        for ev in activities:
            lines.append(
                f"[{ev.status}] {ev.message}  ·  {ev.task_id} @ {ev.timestamp}"
            )
        return "\n".join(lines)


def get_bridge() -> HeadlessBrokerBridge:
    return HeadlessBrokerBridge.instance()


def assert_no_tkinter_loaded() -> None:
    """Raise if Tkinter was imported into this process (HF Space guard)."""
    import sys

    bad = [name for name in ("tkinter", "_tkinter", "customtkinter") if name in sys.modules]
    if bad:
        raise RuntimeError(f"Tkinter modules loaded in web process: {bad}")


__all__ = (
    "HeadlessBrokerBridge",
    "assert_no_tkinter_loaded",
    "get_bridge",
    "status_label",
)
