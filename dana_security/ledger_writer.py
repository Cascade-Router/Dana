"""Append escalation ``[PENDING]`` tickets to ``dana_security/patch_ledger.md``.

Fatal / fail-closed / sub-graph escalate paths call this helper so tickets are
flushed to disk immediately (not only held on ``AgentState.drafted_ticket``).

Does **not** touch ToolForge AST/subprocess gates.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LEDGER_HEADER = "# Donna Patch Ledger\n"

# Injectable override for tests (``None`` → ``dana.paths.PATCH_LEDGER_PATH``).
_LEDGER_PATH_OVERRIDE: Path | None = None


def set_ledger_path_override(path: Path | str | None) -> None:
    """Set or clear the process-wide ledger path override (tests)."""
    global _LEDGER_PATH_OVERRIDE
    _LEDGER_PATH_OVERRIDE = Path(path) if path is not None else None


def resolve_ledger_path(ledger_path: Path | str | None = None) -> Path:
    """Resolve the patch ledger file path (explicit → override → paths constant)."""
    if ledger_path is not None:
        return Path(ledger_path)
    if _LEDGER_PATH_OVERRIDE is not None:
        return Path(_LEDGER_PATH_OVERRIDE)
    try:
        from dana.paths import DONNA_SECURITY_DIR, PATCH_LEDGER_PATH

        DONNA_SECURITY_DIR.mkdir(parents=True, exist_ok=True)
        return Path(PATCH_LEDGER_PATH)
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parent / "patch_ledger.md"


def _new_task_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"esc_{stamp}_{uuid.uuid4().hex[:8]}"


def _error_trace_from_state(state: dict[str, Any] | None) -> str:
    """Prefer zero-copy ``raw_state_buffer``; fall back to ``execution_error``."""
    try:
        from dana.graph.buffer import get_raw_trace

        trace = get_raw_trace(state)
    except Exception:  # noqa: BLE001
        trace = None
    if isinstance(trace, dict):
        tb = str(trace.get("traceback") or "").strip()
        exc_type = str(trace.get("exception_type") or "").strip()
        exc_msg = str(trace.get("exception_message") or "").strip()
        if tb:
            return tb
        if exc_type or exc_msg:
            return f"{exc_type}: {exc_msg}".strip(": ").strip()
    err = str((state or {}).get("execution_error") or "").strip()
    return err or "(no error trace captured)"


def format_escalation_ticket(
    *,
    task_id: str,
    error_trace: str,
    recommended_fix: str,
    objective: str = "Escalation: fatal or exhausted retries",
    timestamp: str | None = None,
) -> str:
    """Format a clean markdown ``[PENDING]`` escalation block."""
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    obj = (objective or "").strip() or "Escalation: fatal or exhausted retries"
    trace = (error_trace or "").strip() or "(no error trace captured)"
    fix = (recommended_fix or "").strip() or (
        "Inspect host environment / permissions; do not retry the self-heal loop."
    )
    title = obj if len(obj) <= 72 else obj[:69].rstrip() + "..."
    lines = [
        "---",
        f"### Ticket: {title}",
        f"**ID:** `{task_id}`",
        "**Status:** `[PENDING]`",
        f"**Timestamp:** {stamp}",
        f"**Task ID:** `{task_id}`",
        f"**Objective:** {obj}",
        "**Error Trace:**",
        "```",
        trace,
        "```",
        "**Recommended Fix:**",
        fix,
        "",
        "**Security & Guardrails:** Keep diffs minimal. Do not modify offline "
        "routing constraints or ToolForge security gates.",
        "**Cursor Receipt:** ",
        "*(Awaiting compilation...)*",
    ]
    return "\n" + "\n".join(lines) + "\n"


def append_pending_ticket(
    ticket: str,
    *,
    ledger_path: Path | str | None = None,
) -> Path:
    """Append ``ticket`` to the ledger and flush immediately."""
    dest = resolve_ledger_path(ledger_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        try:
            from dana.tools.task_queue import shadow_backup_before_write

            shadow_backup_before_write(dest)
        except Exception:  # noqa: BLE001
            pass
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.write_text(_LEDGER_HEADER + "\n", encoding="utf-8")
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(ticket)
        fh.flush()
    logger.info("patch_ledger: wrote [PENDING] escalation ticket path=%s", dest)
    return dest


def write_escalation_ticket(
    state: dict[str, Any] | None,
    *,
    reason: str = "escalation",
    objective: str | None = None,
    recommended_fix: str | None = None,
    ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    """Format + flush a ``[PENDING]`` ticket from AgentState; never raises.

    Returns a small metadata dict (``ticket_id``, ``ledger_path``, ``ok``).
    """
    st = state or {}
    # Prefer explicit arg, then per-state inject (tests), then process override.
    effective_path = ledger_path
    if effective_path is None:
        injected = st.get("patch_ledger_path")
        if injected:
            effective_path = injected

    task_id = _new_task_id()
    session = str(st.get("session_id") or "").strip()
    if session:
        task_id = f"{task_id}_{session[:24]}"

    error_trace = _error_trace_from_state(st)
    drafted = st.get("drafted_ticket") if isinstance(st.get("drafted_ticket"), dict) else {}
    obj = (
        (objective or "").strip()
        or str((drafted or {}).get("objective") or "").strip()
        or f"Escalation ({reason}): fatal block or retries exhausted"
    )
    if recommended_fix and str(recommended_fix).strip():
        fix = str(recommended_fix).strip()
    else:
        ctx = str((drafted or {}).get("context") or "").strip()
        fix = ctx or (
            "Resolve the host OS / dependency / permission failure; "
            "preserve raw_state_buffer for diagnostics; do not spin Critic retries."
        )

    ticket = format_escalation_ticket(
        task_id=task_id,
        error_trace=error_trace,
        recommended_fix=fix,
        objective=obj,
    )
    try:
        dest = append_pending_ticket(ticket, ledger_path=effective_path)
    except Exception as exc:  # noqa: BLE001 — never crash the graph
        logger.warning(
            "patch_ledger: FAILED to write [PENDING] ticket id=%s (%s: %s)",
            task_id,
            type(exc).__name__,
            exc,
        )
        return {
            "ok": False,
            "ticket_id": task_id,
            "ledger_path": str(resolve_ledger_path(effective_path)),
            "error": f"{type(exc).__name__}: {exc}",
        }

    logger.info(
        "patch_ledger: escalation ticket written id=%s reason=%s path=%s",
        task_id,
        reason,
        dest,
    )
    return {
        "ok": True,
        "ticket_id": task_id,
        "ledger_path": str(dest),
        "reason": reason,
    }


__all__ = (
    "append_pending_ticket",
    "format_escalation_ticket",
    "resolve_ledger_path",
    "set_ledger_path_override",
    "write_escalation_ticket",
)
