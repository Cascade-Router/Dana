"""Self-healing python_repl critic + fail-closed nodes (offline-friendly)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from donna.schema import ReactGraphState

logger = logging.getLogger(__name__)

# Optional injectable LLM: (error, code) -> critique text (may include FIXED_CODE block).
CriticLLM = Callable[[str, str], str]

_DEFAULT_MAX_RETRIES = 3


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
    )
    return any(marker in text for marker in failure_markers)


def python_repl_state_patch(*, code: str, observation: str) -> dict[str, Any]:
    """State update after a python_repl tool result (only code-exec failures)."""
    src = code if isinstance(code, str) else str(code or "")
    obs = str(observation or "")
    if is_python_repl_failure(obs):
        return {
            "execution_error": obs[:2000],
            "last_code_snippet": src,
        }
    return {
        "execution_error": None,
        "last_code_snippet": src,
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


def fail_closed_node(state: ReactGraphState) -> dict[str, Any]:
    """Halt after exhausted REPL self-heal attempts; log critique summary."""
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
    return {
        "halt": True,
        "final_raw": msg,
        "last_obs": msg,
        "current_agent": "FailClosed",
        "always_include": [],
    }
