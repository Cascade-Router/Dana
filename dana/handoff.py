"""Module 4 — deterministic Swarm Handoff parse + execute."""

from __future__ import annotations

import json
import re
from typing import Any

from dana.schema import Handoff

_HANDOFF_JSON_RE = re.compile(
    r"\{[^{}]*\"target_agent\"[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)
_HANDOFF_BLOCK_RE = re.compile(
    r"(?is)\bHANDOFF\s*:\s*(.+?)(?=\n\s*[A-Z][A-Z0-9_ ]{1,24}\s*:|\Z)",
)


def parse_handoff_payload(text: str) -> Handoff | None:
    """Best-effort extract a ``Handoff`` from agent prose / JSON."""
    raw = (text or "").strip()
    if not raw:
        return None

    # 1) Explicit JSON object containing target_agent.
    for m in _HANDOFF_JSON_RE.finditer(raw):
        blob = m.group(0)
        try:
            data = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "target_agent" in data:
            try:
                return Handoff.model_validate(
                    {
                        "target_agent": data.get("target_agent"),
                        "reason": data.get("reason") or data.get("why") or "handoff",
                        "intent_context": data.get("intent_context")
                        or data.get("context")
                        or data.get("intent")
                        or raw[:500],
                    }
                )
            except Exception:  # noqa: BLE001
                continue

    # 2) HANDOFF: target=... reason=... context=...
    block_m = _HANDOFF_BLOCK_RE.search(raw)
    if block_m:
        body = block_m.group(1).strip()
        fields: dict[str, str] = {}
        for km in re.finditer(
            r"(?im)\b(target_agent|target|reason|intent_context|context)\s*[:=]\s*(.+)$",
            body,
        ):
            key = km.group(1).lower()
            val = km.group(2).strip().strip("'\"")
            if key in {"target", "target_agent"}:
                fields["target_agent"] = val
            elif key == "reason":
                fields["reason"] = val
            else:
                fields["intent_context"] = val
        if fields.get("target_agent"):
            try:
                return Handoff.model_validate(
                    {
                        "target_agent": fields["target_agent"],
                        "reason": fields.get("reason") or "capability switch",
                        "intent_context": fields.get("intent_context") or body[:500],
                    }
                )
            except Exception:  # noqa: BLE001
                pass

    return None


def execute_handoff(
    handoff: Handoff,
    *,
    session_id: str = "",
    current_agent: str = "",
) -> dict[str, Any]:
    """Apply handoff deterministically (mode / Blackboard / telemetry)."""
    target = handoff.target_agent
    # Map swarm agents → Mode Manager when applicable.
    mode_map = {
        "Vision_Agent": "vision",
        "Chat_Node": "chat",
        "MoA_Reasoner": "developer",
        "ReAct_Agent": "developer",
        "Mailroom": "chat",
    }
    mode = mode_map.get(target)
    if mode:
        try:
            from dana.agentic import set_dana_mode

            set_dana_mode(mode)
        except Exception:  # noqa: BLE001
            pass
    try:
        from dana.memory import set_session_meta

        if session_id:
            set_session_meta(
                session_id,
                current_agent=target,
                active_intent=handoff.reason[:120],
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        from dana.telemetry import log_handoff

        log_handoff(
            target,
            session_id=session_id,
            current_agent=current_agent or "ReAct_Agent",
            active_intent=handoff.reason[:80],
            payload={
                "reason": handoff.reason,
                "intent_context": handoff.intent_context[:500],
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "current_agent": target,
        "active_intent": handoff.reason[:120],
        "mode": mode or "",
        "ack": f"Handoff complete → {target}. {handoff.reason}",
    }
