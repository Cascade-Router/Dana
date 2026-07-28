"""Graph node: execute a saved desktop macro by id.

Invocation
----------
Users can trigger macros with natural language such as::

    Run macro build_and_test
    execute macro login_flow
    replay macro smoke_check

``parse_macro_command(text)`` extracts the id. Wire this node into the ReAct
graph only when an injectable-node slot is available; until then callers may
invoke ``execute_macro_node(state)`` directly (does not touch the HITL corridor).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dana.macros.engine import MacroEngine, sanitize_macro_id
from dana.schema import ReactGraphState

logger = logging.getLogger(__name__)

_MACRO_CMD_RE = re.compile(
    r"(?is)\b(?:run|execute|replay)\s+macro\s+([A-Za-z0-9_\-]+)\b"
)


def parse_macro_command(text: str) -> str | None:
    """Return macro_id from commands like ``Run macro build_and_test``, else None."""
    m = _MACRO_CMD_RE.search(text or "")
    if not m:
        return None
    return sanitize_macro_id(m.group(1))


def _latest_user_text(state: ReactGraphState | dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        if isinstance(msg, dict):
            role = str(msg.get("role") or msg.get("type") or "")
            content = msg.get("content") or msg.get("text") or ""
        else:
            role = str(
                getattr(msg, "type", None)
                or getattr(msg, "role", None)
                or ""
            )
            content = getattr(msg, "content", None) or ""
        role_l = role.lower()
        if role_l in {"human", "user"} or role_l.endswith("humanmessage"):
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(str(p.get("text") or ""))
                    else:
                        parts.append(str(p))
                return "\n".join(parts)
            return str(content or "")
    return ""


def resolve_macro_id(state: ReactGraphState | dict[str, Any]) -> str | None:
    """Prefer explicit ``macro_id``, then intent, then last user message."""
    direct = state.get("macro_id")
    if direct:
        return sanitize_macro_id(str(direct))
    intent = str(state.get("active_intent") or "")
    parsed = parse_macro_command(intent)
    if parsed:
        return parsed
    return parse_macro_command(_latest_user_text(state))


def execute_macro_node(
    state: ReactGraphState | dict[str, Any],
    *,
    engine: MacroEngine | None = None,
) -> dict[str, Any]:
    """Replay a macro; returns a state patch with ``last_obs`` (injectable engine)."""
    macro_id = resolve_macro_id(state)
    if not macro_id:
        obs = (
            "[macro] No macro_id found. Say e.g. 'Run macro build_and_test' "
            "or set state['macro_id']."
        )
        return {"last_obs": obs}

    eng = engine or MacroEngine()
    try:
        result = eng.replay_macro(macro_id)
    except FileNotFoundError as exc:
        obs = f"[macro] ERROR: {exc}"
        logger.warning(obs)
        return {"last_obs": obs}
    except Exception as exc:  # noqa: BLE001
        obs = f"[macro] ERROR replaying {macro_id!r}: {exc}"
        logger.warning(obs)
        return {"last_obs": obs}

    if result.get("ok"):
        obs = (
            f"[macro] ok macro_id={result.get('macro_id')!r} "
            f"steps={result.get('steps')}"
        )
    else:
        obs = (
            f"[macro] FAILED macro_id={result.get('macro_id')!r}: "
            f"{result.get('error') or 'unknown error'}"
        )
    return {"last_obs": obs, "macro_result": result}
