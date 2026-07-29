"""Self-healing python_repl critic + fail-closed nodes (offline-friendly)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from dana.schema import ReactGraphState

logger = logging.getLogger(__name__)

# Optional injectable LLM: (error, code) -> critique text (may include FIXED_CODE block).
CriticLLM = Callable[[str, str], str]

_DEFAULT_MAX_RETRIES = 3

# OS / env / missing-dependency blocks — never self-heal; HITL ticket corridor.
FATAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    PermissionError,
    FileNotFoundError,
    ModuleNotFoundError,
    ConnectionRefusedError,
    TimeoutError,
    OSError,
)

FATAL_OS_BLOCK_MSG = (
    "Fatal OS Block: Missing dependency or permission denied"
)

_FATAL_TYPE_NAMES: tuple[str, ...] = (
    "PermissionError",
    "FileNotFoundError",
    "ModuleNotFoundError",
    "ConnectionRefusedError",
    "TimeoutError",
    "OSError",
    "WinError",
)

# Code-level faults the Critic may attempt to patch.
_FIXABLE_TYPE_NAMES: tuple[str, ...] = (
    "SyntaxError",
    "NameError",
    "KeyError",
    "ValueError",
    "ZeroDivisionError",
    "TypeError",
    "AttributeError",
    "IndentationError",
)


def is_fatal_execution_error(error: BaseException | str | None) -> bool:
    """True for fatal OS / permission / missing-dependency execution errors."""
    if error is None:
        return False
    if isinstance(error, BaseException):
        # ModuleNotFoundError is ImportError subclass; keep before generic OSError.
        if isinstance(error, FATAL_EXCEPTIONS):
            return True
        return False
    text = str(error or "")
    if not text.strip():
        return False
    # Explicit type tokens (fatal wins over overlapping fixable noise).
    for name in _FATAL_TYPE_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            return True
    lower = text.lower()
    if "missing dependency" in lower or "no module named" in lower:
        return True
    return False


def is_fixable_execution_error(error: BaseException | str | None) -> bool:
    """True when the error looks like a Critic-healable code fault (not fatal)."""
    if is_fatal_execution_error(error):
        return False
    if error is None:
        return False
    if isinstance(error, BaseException):
        return type(error).__name__ in _FIXABLE_TYPE_NAMES
    text = str(error or "")
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in _FIXABLE_TYPE_NAMES)


def is_python_repl_failure(observation: str) -> bool:
    """True when a python_repl observation indicates a code-execution failure."""
    text = str(observation or "")
    if not text.strip():
        return False
    if text.startswith("ERROR:"):
        return True
    if text.startswith("WARNING: python_repl timed out"):
        return True
    m = re.search(r"exit_code\s*=\s*(-?\d+)", text)
    if m is not None and int(m.group(1)) != 0:
        return True
    failure_markers = (
        "Traceback (most recent call last)",
        "ZeroDivisionError",
        "NameError",
        "SyntaxError",
        "ImportError",
        "ModuleNotFoundError",
        "TypeError",
        "ValueError",
        "AttributeError",
        "IndentationError",
        "OSError",
        "WinError",
        "PermissionError",
        "FileNotFoundError",
        "ConnectionRefusedError",
        "TimeoutError",
        "KeyError",
    )
    return any(marker in text for marker in failure_markers)


def python_repl_state_patch(*, code: str, observation: str) -> dict[str, Any]:
    """State update after a python_repl tool result (only code-exec failures)."""
    src = code if isinstance(code, str) else str(code or "")
    obs = str(observation or "")
    if is_python_repl_failure(obs):
        fatal = is_fatal_execution_error(obs)
        return {
            "execution_error": obs[:2000],
            "last_code_snippet": src,
            "fatal_block": fatal,
        }
    return {
        "execution_error": None,
        "last_code_snippet": src,
        "fatal_block": False,
    }


def _fatal_ticket_draft(error: str, *, session_id: str = "") -> dict[str, Any]:
    """Structured ticket payload for the existing HITL corridor (fail_closed path)."""
    err = str(error or "")[:1500]
    context = (
        "Root cause: Fatal OS / environment block during python_repl "
        "(permission denied, missing dependency, or OS error).\n"
        f"Error detail:\n{err}\n"
        "Step-by-step changes: resolve the missing package or filesystem "
        "permission on the host; do not retry the Critic self-heal loop.\n"
        "Acceptance criteria: dependency/permission fixed; repl exits with "
        "exit_code=0 without fatal_block.\n"
        "Target files: dana/graph/nodes/critic.py, dana/exec/shadow_workspace.py\n"
    )
    return {
        "type": "ticket_approval",
        "tool": "draft_cursor_prompt",
        "objective": FATAL_OS_BLOCK_MSG,
        "context": context,
        "session_id": str(session_id or ""),
        "active_intent": "fatal_os_block",
    }


def _extract_last_code(state: ReactGraphState | dict[str, Any]) -> str:
    snippet = str(state.get("last_code_snippet") or "").strip()
    if snippet:
        return snippet
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = str(tc.get("name") or "")
                args = tc.get("args") or {}
            else:
                name = str(getattr(tc, "name", "") or "")
                args = getattr(tc, "args", None) or {}
            if name == "python_repl":
                return str((args or {}).get("code") or "")
    return ""


def _parse_fixed_code(critique: str, fallback: str) -> str:
    """Prefer a FIXED_CODE fenced block from the critic response."""
    text = str(critique or "")
    m = re.search(r"FIXED_CODE\s*:\s*```(?:python)?\s*([\s\S]*?)```", text, re.I)
    if m:
        return m.group(1).strip() or fallback
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        body = m.group(1).strip()
        if body and body != fallback:
            return body
    return fallback


def heuristic_critique(error: str, code: str) -> str:
    """Offline diagnosis + correction plan (no network)."""
    err = str(error or "")
    src = str(code or "")
    fixed = src

    if "ZeroDivisionError" in err or re.search(r"/\s*0\b", src):
        fixed = re.sub(r"/\s*0\b", "/ 1", src)
        fixed = fixed.replace("1/0", "1/1")
        plan = "ZeroDivisionError: divisor is zero — use a non-zero divisor."
    elif "NameError" in err:
        m = re.search(r"name '([^']+)' is not defined", err)
        name = m.group(1) if m else "missing_name"
        if name not in src.split("=", 1)[0] or f"{name} =" not in src:
            fixed = f"{name} = None  # critic: define before use\n{src}"
        plan = f"NameError: define `{name}` before use."
    elif "ModuleNotFoundError" in err or "ImportError" in err:
        m = re.search(r"No module named '([^']+)'", err)
        pkg = m.group(1) if m else "missing_package"
        plan = (
            f"Missing package `{pkg}` — install it or replace with stdlib. "
            "Avoid inventing imports."
        )
    elif "SyntaxError" in err or "IndentationError" in err:
        plan = "Syntax/Indentation error — balance quotes, colons, and indentation."
    elif "WinError" in err or "Win32" in err or "OSError" in err:
        plan = "Win32/OS error — check paths, permissions, and quoting on Windows."
    else:
        plan = f"Execution failed — inspect traceback and patch the failing line. ({err[:180]})"

    return f"{plan}\nFIXED_CODE:\n```python\n{fixed}\n```"


def make_critic_node(critic_llm: CriticLLM | None = None) -> Callable[[ReactGraphState], dict[str, Any]]:
    """Build a critic node; ``critic_llm(error, code) -> str`` optional."""

    def critic_node(state: ReactGraphState) -> dict[str, Any]:
        error = str(state.get("execution_error") or "")
        # Fatal OS / dependency blocks: never invoke Critic LLM or retry tools.
        if state.get("fatal_block") or is_fatal_execution_error(error):
            sid = str(state.get("session_id") or "")
            return {
                "fatal_block": True,
                "halt": True,
                "final_raw": FATAL_OS_BLOCK_MSG,
                "last_obs": FATAL_OS_BLOCK_MSG,
                "current_agent": "Critic",
                "execution_error": error or FATAL_OS_BLOCK_MSG,
                "drafted_ticket": _fatal_ticket_draft(error, session_id=sid),
                "always_include": [],
            }
        code = _extract_last_code(state)
        if critic_llm is not None:
            try:
                raw = str(critic_llm(error, code) or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("critic_llm failed (%s); using heuristic", exc)
                raw = heuristic_critique(error, code)
        else:
            raw = heuristic_critique(error, code)

        critique = raw.strip() or heuristic_critique(error, code)
        fixed = _parse_fixed_code(critique, code or "pass")
        history = list(state.get("critique_history") or [])
        history.append(critique[:1000])
        retry = int(state.get("retry_count") or 0) + 1

        from langchain_core.messages import AIMessage, SystemMessage

        return {
            "critique_history": history,
            "retry_count": retry,
            "last_code_snippet": fixed,
            "execution_error": None,
            "current_agent": "Critic",
            "messages": [
                SystemMessage(content=f"CRITIC retry={retry}: {critique[:800]}"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "python_repl",
                            "args": {"code": fixed},
                            "id": f"critic-retry-{retry}",
                            "type": "tool_call",
                        }
                    ],
                ),
            ],
        }

    return critic_node


critic_node = make_critic_node()


def _flush_escalation_ledger(
    state: ReactGraphState,
    *,
    reason: str,
    objective: str | None = None,
    recommended_fix: str | None = None,
) -> None:
    """Append ``[PENDING]`` ticket to ``dana_security/patch_ledger.md`` (best-effort)."""
    try:
        from dana_security.ledger_writer import write_escalation_ticket

        meta = write_escalation_ticket(
            dict(state) if not isinstance(state, dict) else state,
            reason=reason,
            objective=objective,
            recommended_fix=recommended_fix,
        )
        if meta.get("ok"):
            logger.info(
                "fail_closed: flushed [PENDING] ticket id=%s path=%s",
                meta.get("ticket_id"),
                meta.get("ledger_path"),
            )
        else:
            logger.warning(
                "fail_closed: ledger flush failed id=%s err=%s",
                meta.get("ticket_id"),
                meta.get("error"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fail_closed: ledger writer unavailable (%s)", exc)


def fail_closed_node(state: ReactGraphState) -> dict[str, Any]:
    """Halt after exhausted REPL self-heal attempts; log critique summary.

    Fatal OS blocks skip the Critic loop entirely and land here with a
    drafted HITL ticket payload (existing ticket corridor fields).
    Always flushes a ``[PENDING]`` block to ``dana_security/patch_ledger.md``.
    """
    error = str(state.get("execution_error") or "")
    if state.get("fatal_block") or is_fatal_execution_error(error):
        sid = str(state.get("session_id") or "")
        logger.error("fail_closed: fatal_block — %s", error[:400])
        drafted = _fatal_ticket_draft(error, session_id=sid)
        _flush_escalation_ledger(
            {**dict(state), "drafted_ticket": drafted},
            reason="fatal_block",
            objective=str(drafted.get("objective") or FATAL_OS_BLOCK_MSG),
            recommended_fix=str(drafted.get("context") or ""),
        )
        return {
            "halt": True,
            "fatal_block": True,
            "final_raw": FATAL_OS_BLOCK_MSG,
            "last_obs": FATAL_OS_BLOCK_MSG,
            "current_agent": "FailClosed",
            "drafted_ticket": drafted,
            "always_include": [],
        }

    history = [str(x) for x in (state.get("critique_history") or [])]
    retry = int(state.get("retry_count") or 0)
    summary = " | ".join(h[:200] for h in history) if history else "(no critiques)"
    logger.error(
        "fail_closed: python_repl self-heal exhausted retries=%s critiques=%s",
        retry,
        summary[:500],
    )
    msg = (
        f"FAIL_CLOSED: python_repl self-heal exhausted after {retry} retries. "
        f"Critiques: {summary}"
    )
    _flush_escalation_ledger(
        state,
        reason="fail_closed_exhausted",
        objective="FAIL_CLOSED: python_repl self-heal retries exhausted",
        recommended_fix=(
            f"Retries={retry}. Critique summary: {summary[:800]}\n"
            "Inspect last_code_snippet / execution_error; patch the failing "
            "code path rather than looping Critic indefinitely."
        ),
    )
    return {
        "halt": True,
        "final_raw": msg,
        "last_obs": msg,
        "current_agent": "FailClosed",
        "always_include": [],
    }
