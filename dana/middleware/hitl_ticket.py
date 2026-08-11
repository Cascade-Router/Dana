"""Stage 8.6 — HITL ticket approval bridge (LangGraph interrupt ↔ GUI).

LangGraph pauses inside the ``ticket_approval`` node via ``interrupt()``.
The ReAct runner publishes the drafted ticket here; the Live Trace GUI
calls ``submit_decision`` (Approve / Deny). Headless runs auto-approve
unless ``DANA_HITL_REQUIRE_GUI=1``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_DECISION_EVENT = threading.Event()
_PENDING: dict[str, Any] | None = None
_DECISION: dict[str, Any] | None = None
_THREAD_ID: str = ""
# Stage 8.9.3 — process-wide consecutive HITL denials (mirrors graph state).
_CONSECUTIVE_DENIALS = 0
_ACTIVE_TICKET_FP = ""


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("HITL", msg)
    except Exception:  # noqa: BLE001
        print(f"[HITL] {msg}", flush=True)


def hitl_enabled() -> bool:
    """Master switch — ``DANA_HITL_TICKET=0`` disables the gate entirely."""
    raw = (os.environ.get("DANA_HITL_TICKET") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _gui_listening() -> bool:
    try:
        from dana.core import shared_state

        gui = shared_state.get_gui_instance()
        if gui is None:
            return False
        try:
            return bool(gui.winfo_exists())
        except Exception:  # noqa: BLE001
            return True
    except Exception:  # noqa: BLE001
        return False


def should_auto_resolve() -> bool:
    """True when Approve should not block (tests / headless / explicit env)."""
    if _env_flag("DANA_HITL_AUTO_APPROVE") or _env_flag("DANA_HITL_AUTO_DENY"):
        return True
    if _env_flag("DANA_HITL_REQUIRE_GUI"):
        return False
    return not _gui_listening()


def _ticket_fingerprint(payload: dict[str, Any] | None) -> str:
    data = dict(payload or {})
    return (
        f"{str(data.get('objective') or '').strip()}|"
        f"{str(data.get('context') or '').strip()}"
    )


def get_consecutive_denials() -> int:
    with _LOCK:
        return int(_CONSECUTIVE_DENIALS)


def reset_consecutive_denials(*, reason: str = "") -> None:
    """Force counter to 0 (approve / new distinct task)."""
    global _CONSECUTIVE_DENIALS
    with _LOCK:
        _CONSECUTIVE_DENIALS = 0
    if reason:
        _log(f"consecutive_denials reset ({reason})")


def begin_ticket_hitl(payload: dict[str, Any] | None) -> int:
    """Call when entering HITL for a drafted ticket.

    Resets the denial counter when the ticket fingerprint changes (new task).
    Returns the current ``consecutive_denials`` value for GUI / graph state.
    """
    global _CONSECUTIVE_DENIALS, _ACTIVE_TICKET_FP
    fp = _ticket_fingerprint(payload)
    with _LOCK:
        if _ACTIVE_TICKET_FP and fp and fp != _ACTIVE_TICKET_FP:
            _CONSECUTIVE_DENIALS = 0
            _log("consecutive_denials reset (new distinct task)")
        if fp:
            _ACTIVE_TICKET_FP = fp
        return int(_CONSECUTIVE_DENIALS)


def record_hitl_decision(approved: bool) -> int:
    """Increment on Deny; reset on Approve. Returns updated counter."""
    global _CONSECUTIVE_DENIALS
    with _LOCK:
        if approved:
            _CONSECUTIVE_DENIALS = 0
        else:
            _CONSECUTIVE_DENIALS = int(_CONSECUTIVE_DENIALS) + 1
        n = int(_CONSECUTIVE_DENIALS)
    _log(f"consecutive_denials -> {n} (approved={bool(approved)})")
    return n


def extract_files_line(context: str, files_hint: Any = None) -> str:
    """Best-effort Target Files line for HITL / Orb display."""
    if isinstance(files_hint, (list, tuple)):
        joined = ", ".join(str(x).strip() for x in files_hint if str(x).strip())
        if joined:
            return joined
    if isinstance(files_hint, str) and files_hint.strip():
        return files_hint.strip()
    import re

    ctx = context or ""
    found = re.findall(
        r"\b[\w./\\-]+\.(?:py|md|json|txt|yml|yaml)\b",
        ctx,
        flags=re.IGNORECASE,
    )
    # Preserve order, drop dupes.
    seen: set[str] = set()
    ordered: list[str] = []
    for path in found:
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ", ".join(ordered) if ordered else "(see context)"


def format_ticket_payload(payload: dict[str, Any] | None) -> str:
    """Human-readable drafted ticket for Payload Viewer + Orb (full body)."""
    data = dict(payload or {})
    objective = str(data.get("objective") or "").strip()
    context = str(data.get("context") or "").strip()
    tool = str(data.get("tool") or "draft_cursor_prompt").strip()
    critique = str(data.get("jason_critique") or "").strip()
    files = extract_files_line(context, data.get("files"))
    try:
        denials = int(data.get("consecutive_denials") or 0)
    except (TypeError, ValueError):
        denials = 0
    lines = [
        "=== HITL TICKET APPROVAL REQUIRED ===",
        f"tool: {tool}",
        f"consecutive_denials: {denials}",
        "",
        "=== JASON REVIEW ===",
        critique or "(no critique)",
        "",
        "Objective:",
        objective or "(empty)",
        "",
        "Context:",
        context or "(empty)",
        "",
        "Files:",
        files,
        "",
        "Awaiting operator: [Approve & Submit] or [Deny / Edit]",
    ]
    if denials >= 2:
        lines.append("")
        lines.append(
            "ESCALATION: Report Issue on GitHub is available "
            "(Jason failed twice)."
        )
    return "\n".join(lines)


def publish_pending(
    payload: dict[str, Any],
    *,
    thread_id: str = "",
) -> None:
    """Publish a pending interrupt payload for the GUI (thread-safe)."""
    global _PENDING, _DECISION, _THREAD_ID
    data = dict(payload or {})
    denials = begin_ticket_hitl(data)
    data["consecutive_denials"] = int(
        data.get("consecutive_denials")
        if data.get("consecutive_denials") is not None
        else denials
    )
    # Prefer live counter (may have just reset for a new task).
    data["consecutive_denials"] = get_consecutive_denials()
    text = format_ticket_payload(data)
    with _LOCK:
        _PENDING = data
        _PENDING["_formatted"] = text
        _DECISION = None
        _THREAD_ID = str(thread_id or "")
        _DECISION_EVENT.clear()
    _log(
        f"pending approval thread={_THREAD_ID or '-'} "
        f"objective_chars={len(str(data.get('objective') or ''))} "
        f"denials={data['consecutive_denials']}"
    )
    try:
        from dana.ui.trace_bus import emit_trace_event

        emit_trace_event(
            "status",
            node="ticket_approval",
            tool=str(payload.get("tool") or "draft_cursor_prompt"),
            message="HITL_PENDING_APPROVAL",
            payload=text,
            mode="developer",
            state_keys=("hitl_ticket", "objective", "context"),
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from dana.core_agent import emit_trace

        emit_trace(
            "ticket_approval",
            "active",
            "HITL: awaiting Approve / Deny for draft_cursor_prompt",
            mode="developer",
        )
    except Exception:  # noqa: BLE001
        pass


def clear_pending() -> None:
    """Clear pending latch after resume (or cancel)."""
    global _PENDING, _DECISION, _THREAD_ID
    with _LOCK:
        _PENDING = None
        _DECISION = None
        _THREAD_ID = ""
        _DECISION_EVENT.clear()


def get_pending() -> dict[str, Any] | None:
    with _LOCK:
        return dict(_PENDING) if _PENDING is not None else None


def is_pending() -> bool:
    with _LOCK:
        return _PENDING is not None and _DECISION is None


def _log_pending_feedback(decision: dict[str, Any]) -> None:
    """Stage 8.9 — persist human decision + Jason critique locally (JSONL)."""
    with _LOCK:
        pending = dict(_PENDING) if _PENDING is not None else None
    if not pending:
        return
    try:
        from dana.memory.feedback_log import log_human_feedback

        log_human_feedback(
            task_id=str(
                pending.get("tool_call_id")
                or pending.get("task_id")
                or ""
            ),
            human_decision=str(decision.get("action") or "deny"),
            jason_critique=str(pending.get("jason_critique") or ""),
            ticket_content=pending,
            session_id=str(pending.get("session_id") or _THREAD_ID or ""),
            note=str(decision.get("note") or ""),
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARNING: feedback log failed ({exc})")


def submit_decision(
    approved: bool,
    *,
    action: str | None = None,
    note: str = "",
) -> bool:
    """GUI / tests: resolve the pending interrupt (returns False if none)."""
    global _DECISION
    with _LOCK:
        if _PENDING is None:
            return False
        act = (action or ("approve" if approved else "deny")).strip().lower()
        _DECISION = {
            "approved": bool(approved),
            "action": act,
            "note": str(note or ""),
            "ts": time.time(),
        }
        decision_snap = dict(_DECISION)
        _DECISION_EVENT.set()
    denials = record_hitl_decision(bool(approved))
    _log(
        f"decision submitted action={act} approved={bool(approved)} "
        f"consecutive_denials={denials}"
    )
    _log_pending_feedback(decision_snap)
    try:
        from dana.ui.trace_bus import emit_trace_event

        emit_trace_event(
            "status",
            node="ticket_approval",
            message="HITL_RESOLVED",
            payload=(
                f"decision={act} approved={bool(approved)} "
                f"consecutive_denials={denials}"
            ),
            mode="developer",
        )
    except Exception:  # noqa: BLE001
        pass
    return True


def wait_for_decision(*, timeout_s: float | None = None) -> dict[str, Any]:
    """Block the LangGraph worker until Approve/Deny (or auto-resolve)."""
    if _env_flag("DANA_HITL_AUTO_DENY"):
        decision = {"approved": False, "action": "deny", "note": "auto_deny", "ts": time.time()}
        _log("auto-deny (DANA_HITL_AUTO_DENY)")
        clear_after = True
    elif should_auto_resolve():
        decision = {
            "approved": True,
            "action": "approve",
            "note": "auto_approve",
            "ts": time.time(),
        }
        _log("auto-approve (headless / DANA_HITL_AUTO_APPROVE)")
        clear_after = True
    else:
        clear_after = False
        wait_timeout = None if timeout_s is None else max(0.1, float(timeout_s))
        ok = _DECISION_EVENT.wait(timeout=wait_timeout)
        with _LOCK:
            decision = dict(_DECISION) if _DECISION is not None else None
        if not ok or decision is None:
            # Timeout / missing UI → fail closed (do not submit ticket).
            _log("WARNING: HITL wait timed out or empty — denying ticket")
            decision = {
                "approved": False,
                "action": "deny",
                "note": "timeout_or_missing",
                "ts": time.time(),
            }
            clear_after = True

    out = {
        "approved": bool(decision.get("approved")),
        "action": str(decision.get("action") or "deny"),
        "note": str(decision.get("note") or ""),
    }
    if clear_after:
        # Auto-resolve paths never call submit_decision — still log + count.
        denials = record_hitl_decision(bool(out["approved"]))
        out["consecutive_denials"] = denials
        _log_pending_feedback(out)
    else:
        out["consecutive_denials"] = get_consecutive_denials()
    return out


def decision_is_approved(decision: Any) -> bool:
    if decision is True:
        return True
    if decision is False or decision is None:
        return False
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "true", "1"}
    if isinstance(decision, dict):
        if "approved" in decision:
            return bool(decision.get("approved"))
        act = str(decision.get("action") or "").strip().lower()
        return act in {"approve", "approved", "yes", "submit"}
    return False
