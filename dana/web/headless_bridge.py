"""Headless Meta-Broker bridge for Gradio / Hugging Face Spaces (no Tkinter).

Wraps ``run_meta_broker_isolated`` + IPC telemetry so the web UI can submit
prompts on a background thread and poll events without blocking the Gradio
mainloop.
"""

from __future__ import annotations

import json
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
_STATUS_EPIC = "epic_executing"
_STATUS_PENDING_APPROVAL = "pending_approval"

_VALID_STATUS = frozenset(
    {
        _STATUS_IDLE,
        _STATUS_LISTENING,
        _STATUS_PROCESSING,
        _STATUS_EPIC,
        _STATUS_PENDING_APPROVAL,
    }
)


def _ensure_headless_env(*, force_local: bool = True, verbose: bool = False) -> None:
    os.environ["DONNA_NO_GUI"] = "1"
    os.environ["DONNA_HEADLESS"] = "1"
    os.environ.setdefault("DONNA_OLLAMA_KEEP_ALIVE", "0")
    os.environ.setdefault("DONNA_SKIP_RAM_BREAKER", "1")
    os.environ.setdefault("DONNA_META_BROKER_TIMEOUT_S", "600")
    if force_local:
        os.environ["DONNA_FORCE_LOCAL"] = "1"
    else:
        os.environ.pop("DONNA_FORCE_LOCAL", None)
    if verbose:
        os.environ["DONNA_DEBUG"] = "1"
    if not (os.environ.get("DONNA_META_BROKER_LOG") or "").strip():
        os.environ["DONNA_META_BROKER_LOG"] = "logs/hf_space_meta_broker.log"


def status_label(status: str) -> str:
    st = str(status or "idle").strip().lower()
    if st == _STATUS_LISTENING:
        return "● Listening"
    if st == _STATUS_PENDING_APPROVAL or st == "pending_user_approval":
        return "● Awaiting Approval"
    if st == _STATUS_EPIC or st in {"dispatch_epic", "feedback"}:
        return "● Epic Executing"
    if st in {_STATUS_PROCESSING, "routing", "executing", "planning"}:
        return "● Processing"
    return "● Idle"


def load_manifest_dict() -> dict[str, Any]:
    """Read ``.dana_scratch/manifest.json`` (empty contract if missing)."""
    try:
        from dana.graph.artifact_manifest import load_manifest

        return load_manifest()
    except Exception:  # noqa: BLE001
        pass
    try:
        path = os.path.join(".dana_scratch", "manifest.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {"version": 1, "artifacts": []}
    except Exception:  # noqa: BLE001
        pass
    return {"version": 1, "artifacts": []}


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
        self._prompt_echoed = False
        self._started_at = 0.0
        self._finished_at = 0.0
        self._stop_event = threading.Event()
        self._force_local = True
        self._verbose = False
        self._epic_lines: list[str] = []
        # Structured epic rows for interactive Task Tracker.
        self._epic_records: list[dict[str, Any]] = []
        self._workspace_path: str = ""
        self._workspace_label: str = "(waiting for artifacts…)"
        self._pending_approval: dict[str, Any] | None = None
        self._approval_event = threading.Event()
        self._approval_decision: str = ""  # approve | cancel | ""
        self._approved_macro: str = ""

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

    def configure(self, *, force_local: bool = True, verbose: bool = False) -> None:
        with self._lock:
            self._force_local = bool(force_local)
            self._verbose = bool(verbose)

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
                "epic_lines": list(self._epic_lines[-40:]),
                "force_local": self._force_local,
                "verbose": self._verbose,
                "pending_approval": (
                    dict(self._pending_approval)
                    if isinstance(self._pending_approval, dict)
                    else None
                ),
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
            if st in {"dispatch_epic", "feedback"} or "epic" in st:
                st = _STATUS_EPIC
            elif st in {"pending_user_approval", "pending_approval", "awaiting_approval"}:
                st = _STATUS_PENDING_APPROVAL
            elif st in {"routing", "executing", "planning"}:
                st = _STATUS_PROCESSING
            else:
                st = _STATUS_IDLE
        with self._lock:
            self._status = st

    def _is_prompt_echo(self, msg: str) -> bool:
        """True when telemetry is blindly repeating the submitted user prompt."""
        text = str(msg or "").strip()
        if not text:
            return False
        with self._lock:
            prompt = self._prompt.strip()
        if not prompt:
            return False
        # Exact / near-exact echo of the user macro (with or without /broker prefix).
        low = text.lower()
        plow = prompt.lower()
        if low == plow or low == f"/broker {plow}".lstrip():
            return True
        if plow.startswith("/broker") and low == plow[len("/broker") :].strip():
            return True
        # Long prompt prefix dumped into a telemetry line.
        if len(plow) >= 40 and (plow[:80] in low or low[:80] in plow):
            if low.startswith("prompt:") or "submitted prompt" in low:
                return True
            if text.startswith(prompt[:60]) or text.startswith(prompt[8:68] if prompt.lower().startswith("/broker") else "\0"):
                return True
        return False

    def _append_log(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        if self._is_prompt_echo(text):
            return
        # Strip a leading "PROMPT: …" mirror that duplicates the chat history.
        if text.upper().startswith("PROMPT:"):
            return
        with self._lock:
            # Dedup consecutive identical lines.
            if self._log and self._log[-1] == text:
                return
            self._log.append(text)
            if len(self._log) > 400:
                self._log = self._log[-400:]

    def _upsert_epic_record(
        self,
        *,
        title: str,
        status: str,
        message: str = "",
        phase: str = "",
        exit_code: Any = None,
    ) -> None:
        title_s = (title or message or "Epic").strip()[:120]
        with self._lock:
            rec = None
            for row in self._epic_records:
                if row.get("title") == title_s or (
                    title_s and title_s in str(row.get("message") or "")
                ):
                    rec = row
                    break
            if rec is None:
                rec = {
                    "id": len(self._epic_records) + 1,
                    "title": title_s,
                    "status": "Idle",
                    "phase": "",
                    "message": "",
                    "exit_code": None,
                    "prompt_snippet": (self._prompt or "")[:400],
                }
                self._epic_records.append(rec)
            if status:
                rec["status"] = status
            if phase:
                rec["phase"] = phase
            if message:
                rec["message"] = message[:400]
            if exit_code is not None:
                rec["exit_code"] = exit_code

    def _append_epic(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        with self._lock:
            self._epic_lines.append(text)
            if len(self._epic_lines) > 80:
                self._epic_lines = self._epic_lines[-80:]

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
        epic_title = str(payload.get("epic_title") or "").strip()

        if self._is_prompt_echo(msg):
            return

        bits = [f"[{kind}]"]
        if phase:
            bits.append(f"phase={phase}")
        if status:
            bits.append(f"status={status}")
        if msg:
            bits.append(msg[:240])
        line = " ".join(bits)
        self._append_log(line)

        # Map telemetry → interactive epic cards.
        st_ui = "Idle"
        low_st = status.lower()
        low_msg = msg.lower()
        if "abort" in low_st or "fail" in low_st or "fail" in low_msg:
            st_ui = "Failed"
        elif "pass" in low_msg or "validated ok" in low_msg or low_st in {
            "completed",
            "ok",
            "success",
        }:
            st_ui = "Passed"
        elif "validat" in low_msg or "harness" in low_msg or phase == "feedback":
            st_ui = "Validating"
        elif phase in {"dispatch_epic", "await_supervisor", "repair"} or "starting epic" in low_msg:
            st_ui = "Running"
        elif self._running:
            st_ui = "Running"

        if (
            "starting epic" in low_msg
            or "dispatch epic" in low_msg
            or epic_title
            or phase in {"dispatch_epic", "feedback"}
        ):
            self._upsert_epic_record(
                title=epic_title or msg[:80] or f"phase:{phase}",
                status=st_ui,
                message=msg,
                phase=phase,
                exit_code=payload.get("exit_code"),
            )
            self._append_epic(msg or line)
            self._set_status(_STATUS_EPIC)
        elif kind == "telemetry" and not payload.get("terminal"):
            self._set_status(_STATUS_PROCESSING)
        elif payload.get("terminal") or kind == "result":
            if msg:
                self._append_epic(msg)
                self._upsert_epic_record(
                    title=epic_title or "Meta-Broker",
                    status=st_ui if st_ui != "Idle" else (
                        "Passed" if low_st in {"completed", "ok", "success"} else "Failed"
                    ),
                    message=msg,
                    phase=phase or "done",
                )
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
            self._prompt_echoed = False
            self._started_at = time.time()
            self._finished_at = 0.0
            self._status = _STATUS_PROCESSING
            self._epic_lines = []
            self._epic_records = []
            self._workspace_path = ""
            self._workspace_label = "(waiting for artifacts…)"
            self._pending_approval = None
            self._approval_decision = ""
            self._approved_macro = ""
            # One UI-facing note only — never dump the full prompt into the log.
            self._log.append("[ui] Meta-Broker job accepted")
            force_local = self._force_local
            verbose = self._verbose
        self._stop_event.clear()
        self._approval_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(text, force_local, verbose),
            name="HFHeadlessBroker",
            daemon=True,
        )
        self._thread.start()
        return True, "Meta-Broker started (isolated process)."
    # Alias used by Gradio Space docs / app wiring.
    submit_prompt = submit

    def pending_approval(self) -> dict[str, Any] | None:
        with self._lock:
            if isinstance(self._pending_approval, dict):
                return dict(self._pending_approval)
        return None

    def approve_spec(self, compiled_spec: str | None = None) -> tuple[bool, str]:
        """HITL: Approve & Run — release the worker to dispatch Meta-Broker."""
        with self._lock:
            pending = self._pending_approval
            if not isinstance(pending, dict):
                return False, "No compiled spec awaiting approval."
            macro = str(
                compiled_spec
                or pending.get("compiled_spec")
                or self._prompt
                or ""
            ).strip()
            if not macro:
                return False, "Empty compiled spec."
            self._approved_macro = macro
            self._approval_decision = "approve"
            self._pending_approval = None
        self._append_log("[ui] Spec approved — dispatching Meta-Broker")
        self._set_status(_STATUS_PROCESSING)
        self._approval_event.set()
        return True, "Spec approved — Meta-Broker dispatching."

    def cancel_spec(self) -> tuple[bool, str]:
        """HITL: Cancel — abort pending compiled spec without dispatch."""
        with self._lock:
            if not isinstance(self._pending_approval, dict):
                return False, "No compiled spec awaiting approval."
            self._approval_decision = "cancel"
            self._pending_approval = None
            self._approved_macro = ""
            self._running = False
            self._status = _STATUS_IDLE
            self._finished_at = time.time()
            self._result = {
                "status": "ABORTED",
                "error": "Spec approval cancelled by user",
                "final_response": "Spec approval cancelled.",
            }
        self._append_log("[ui] Spec approval cancelled")
        self._approval_event.set()
        self._stop_event.set()
        return True, "Spec approval cancelled."

    def stop(self) -> tuple[bool, str]:
        """Request cancellation of the running Meta-Broker child."""
        with self._lock:
            if not self._running and not isinstance(self._pending_approval, dict):
                return False, "No Meta-Broker job is running."
            if isinstance(self._pending_approval, dict):
                self._approval_decision = "cancel"
                self._pending_approval = None
                self._running = False
                self._status = _STATUS_IDLE
        self._approval_event.set()
        self._stop_event.set()
        self._append_log("[ui] stop requested — terminating Meta-Broker child")
        self._set_status(_STATUS_IDLE)
        return True, "Stop signal sent."

    def _worker(self, prompt: str, force_local: bool, verbose: bool) -> None:
        _ensure_headless_env(force_local=force_local, verbose=verbose)
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
        # Spec Compiler: plain English → strict /broker (or REJECT without spawn).
        try:
            from dana.graph.nodes.spec_compiler import (
                PENDING_USER_APPROVAL,
                build_spec_approval_payload,
                compile_user_spec,
                hitl_spec_approval_enabled,
                is_broker_ready_spec,
                is_reject_spec,
            )

            if not is_broker_ready_spec(macro):
                compiled = compile_user_spec(macro)
                if is_reject_spec(compiled):
                    self._push_event(
                        {
                            "type": "telemetry",
                            "phase": "spec_compiler",
                            "status": "ABORTED",
                            "message": compiled,
                            "terminal": True,
                        }
                    )
                    with self._lock:
                        self._result = {
                            "status": "ABORTED",
                            "error": compiled,
                            "final_response": compiled,
                        }
                        self._error = compiled
                        self._running = False
                        self._status = _STATUS_IDLE
                        self._finished_at = time.time()
                    return
                macro = compiled
                self._append_log(
                    f"[spec_compiler] compiled chars={len(macro)}"
                )
            # Human-in-the-loop gate before isolated Meta-Broker spawn.
            if hitl_spec_approval_enabled():
                payload = build_spec_approval_payload(
                    compiled_spec=macro,
                    raw_intent=prompt,
                )
                with self._lock:
                    self._pending_approval = payload
                    self._status = _STATUS_PENDING_APPROVAL
                self._push_event(dict(payload))
                self._append_log(
                    f"[ui] {PENDING_USER_APPROVAL} — waiting for Approve & Run"
                )
                # Block until Approve / Cancel / Stop (or timeout).
                try:
                    wait_s = float(
                        os.environ.get("DONNA_HITL_APPROVAL_TIMEOUT_S") or "3600"
                    )
                except (TypeError, ValueError):
                    wait_s = 3600.0
                self._approval_event.wait(timeout=max(30.0, wait_s))
                with self._lock:
                    decision = self._approval_decision
                    approved_macro = self._approved_macro
                    stop_hit = self._stop_event.is_set()
                if decision == "cancel" or stop_hit:
                    with self._lock:
                        self._running = False
                        self._status = _STATUS_IDLE
                        self._finished_at = time.time()
                        if not self._result:
                            self._result = {
                                "status": "ABORTED",
                                "error": "Spec approval cancelled",
                                "final_response": "Spec approval cancelled.",
                            }
                    return
                if decision != "approve":
                    with self._lock:
                        self._running = False
                        self._status = _STATUS_IDLE
                        self._finished_at = time.time()
                        self._error = "Spec approval timed out"
                        self._result = {
                            "status": "ABORTED",
                            "error": self._error,
                            "final_response": self._error,
                        }
                    self._append_log("[ui] Spec approval timed out")
                    return
                if approved_macro:
                    macro = approved_macro
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[spec_compiler] skipped ({exc})")
            if not prompt.lower().lstrip().startswith("/broker"):
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
                stop_event=self._stop_event,
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
        """Compact epic corridor summary (no duplicated full user prompt)."""
        parts: list[str] = []
        with self._lock:
            epic_lines = list(self._epic_lines[-24:])
            records = [dict(r) for r in self._epic_records]
            running = self._running
        if records:
            parts.append("EPIC CORRIDOR")
            for row in records:
                parts.append(
                    f"  • [{row.get('status')}] {row.get('title')}"
                    + (
                        f"  exit={row.get('exit_code')}"
                        if row.get("exit_code") is not None
                        else ""
                    )
                )
            parts.append("─" * 40)
        elif epic_lines:
            parts.append("EPIC CORRIDOR")
            for line in epic_lines:
                parts.append(f"  • {line}")
            parts.append("─" * 40)
        try:
            from dana.graph.task_tracker import get_shared_task_tracker

            tracker = get_shared_task_tracker()
            activities = tracker.list_activities(limit=20)
            tasks = tracker.list_tasks(limit=8)
        except Exception as exc:  # noqa: BLE001
            parts.append(f"(task tracker unavailable: {exc})")
            return "\n".join(parts) if parts else "(no active tasks)"

        if tasks:
            parts.append("TASKS")
            for rec in tasks:
                st = getattr(rec.status, "value", str(rec.status))
                title = str(rec.task_id or "task")
                parts.append(f"  [{st}] {title}")
        if activities:
            parts.append("ACTIVITY")
            for ev in activities[:16]:
                msg = str(ev.message or "")
                if self._is_prompt_echo(msg):
                    continue
                parts.append(f"  [{ev.status}] {msg[:120]}")
        if not parts:
            return "(no active tasks)" if not running else "Meta-Broker warming up…"
        return "\n".join(parts)

    def epic_choices(self) -> list[str]:
        with self._lock:
            rows = list(self._epic_records)
        if not rows:
            return ["(no epics yet)"]
        return [f"{r.get('id')}. [{r.get('status')}] {r.get('title')}" for r in rows]

    def epic_detail_markdown(self, choice: str | None = None) -> str:
        with self._lock:
            rows = [dict(r) for r in self._epic_records]
            prompt = self._prompt
        if not rows:
            return (
                "_No epic selected — submit a `/broker` command "
                "to populate the tracker._"
            )
        idx = 0
        if choice:
            try:
                idx = max(0, int(str(choice).split(".", 1)[0]) - 1)
            except (TypeError, ValueError):
                idx = 0
        idx = min(idx, len(rows) - 1)
        row = rows[idx]
        exit_code = row.get("exit_code")
        exit_s = "—" if exit_code is None else str(exit_code)
        return (
            f"### Epic {row.get('id')}: {row.get('title')}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| **Status** | `{row.get('status')}` |\n"
            f"| **Phase** | `{row.get('phase') or '—'}` |\n"
            f"| **Harness exit** | `{exit_s}` |\n\n"
            f"**Latest message**\n\n```\n{row.get('message') or '(none)'}\n```\n\n"
            f"**Job prompt (shared)**\n\n"
            f"```\n{(prompt or row.get('prompt_snippet') or '')[:800]}\n```\n"
        )

    def workspace_viewer(self) -> tuple[str, str]:
        """Return ``(label, code_text)`` for the Live Workspace panel."""
        label, body = self._pick_workspace_file()
        with self._lock:
            self._workspace_label = label
            self._workspace_path = label
        return label, body

    def _pick_workspace_file(self) -> tuple[str, str]:
        """Prefer newest epic artifact, then manifest.json."""
        candidates: list[str] = []
        try:
            man = load_manifest_dict()
            for art in man.get("artifacts") or []:
                if isinstance(art, dict) and art.get("file_path"):
                    candidates.append(str(art["file_path"]))
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            for row in self._epic_records:
                msg = str(row.get("message") or "")
                for token in msg.replace("\\", "/").split():
                    tok = token.strip("`'\",")
                    if tok.endswith(".py"):
                        candidates.append(tok)

        seen: set[str] = set()
        ordered: list[str] = []
        for rel in candidates:
            key = rel.replace("\\", "/").lstrip("./")
            if not key or key in seen:
                continue
            if key.startswith(("dana/", "donna_security/", "website/")):
                continue
            seen.add(key)
            ordered.append(key)

        for rel in reversed(ordered):
            path = rel
            if not os.path.isfile(path):
                alt = os.path.join(".dana_scratch", rel)
                if os.path.isfile(alt):
                    path = alt
                else:
                    continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if len(text) > 12000:
                text = text[:12000] + "\n...[truncated]...\n"
            return rel, text

        try:
            man = load_manifest_dict()
            body = json.dumps(man, indent=2, ensure_ascii=False)
            return ".dana_scratch/manifest.json", body
        except Exception:  # noqa: BLE001
            return (
                "(waiting for artifacts…)",
                "# Live workspace will appear here as files are generated.\n",
            )


def get_bridge() -> HeadlessBrokerBridge:
    return HeadlessBrokerBridge.instance()


def assert_no_tkinter_loaded() -> None:
    """Raise if desktop GUI stacks were imported into this web process.

    Hugging Face / Gradio must not pull CustomTkinter or the desktop agent
    monolith. Plain ``tkinter`` may appear transitively in some test hosts; we
    only fail when Dānā desktop UI modules are present.
    """
    import sys

    forbidden = (
        "customtkinter",
        "dana.ui.assistive_orb",
        "dana.ui.main",
        "dana.ui.trace_window",
        "dana.core_agent",
    )
    bad = [name for name in forbidden if name in sys.modules]
    if bad:
        raise RuntimeError(f"Desktop GUI modules loaded in web process: {bad}")


__all__ = (
    "HeadlessBrokerBridge",
    "assert_no_tkinter_loaded",
    "get_bridge",
    "load_manifest_dict",
    "status_label",
)
