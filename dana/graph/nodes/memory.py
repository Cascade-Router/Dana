"""Episodic memory hydrate / consolidate nodes for the ReAct corridor."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dana.memory.store import EpisodicMemoryStore, get_episodic_store
from dana.schema import ReactGraphState

logger = logging.getLogger(__name__)

# Optional injectable LLM: (user_text, assistant_text) -> list of fact dicts.
ConsolidateLLM = Callable[[str, str], list[dict[str, Any]]]

_PREF_PATTERNS: tuple[tuple[re.Pattern[str], str, Any], ...] = (
    (
        re.compile(
            r"\b(?:prefer|use|enable|want)\s+(?:dark\s*mode|dark\s*theme)\b",
            re.I,
        ),
        "prefer_dark_mode",
        True,
    ),
    (
        re.compile(
            r"\b(?:prefer|use|enable|want)\s+(?:light\s*mode|light\s*theme)\b",
            re.I,
        ),
        "prefer_dark_mode",
        False,
    ),
    (
        re.compile(
            r"\b(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][A-Za-z0-9_-]{1,40})\b",
            re.I,
        ),
        "user_name",
        None,  # capture group 1
    ),
    (
        re.compile(
            r"\b(?:i\s+live\s+in|home\s+city\s+is)\s+([A-Za-z][A-Za-z .'-]{1,60})\b",
            re.I,
        ),
        "home_city",
        None,
    ),
)


def _extract_user_text(state: ReactGraphState | dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role in {"human", "user"} and isinstance(content, str) and content.strip():
            return content.split("\n\nVisual Context:", 1)[0].strip()
        if isinstance(msg, dict):
            if msg.get("role") == "user" and str(msg.get("content") or "").strip():
                return str(msg["content"]).split("\n\nVisual Context:", 1)[0].strip()
    return str(state.get("active_intent") or "").strip()


def _extract_assistant_text(state: ReactGraphState | dict[str, Any]) -> str:
    final = str(state.get("final_raw") or "").strip()
    if final:
        return final
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role in {"ai", "assistant"} and isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(msg, dict) and str(msg.get("role") or "") == "assistant":
            body = str(msg.get("content") or "").strip()
            if body:
                return body
    return str(state.get("last_obs") or "").strip()


def heuristic_extract_facts(
    user_text: str,
    assistant_text: str = "",
) -> list[dict[str, Any]]:
    """Offline preference / environment extraction (no cloud API)."""
    blob = f"{user_text or ''}\n{assistant_text or ''}"
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, key, value in _PREF_PATTERNS:
        m = pattern.search(blob)
        if not m:
            continue
        if key in seen:
            continue
        if value is None:
            captured = (m.group(1) or "").strip().rstrip(".,;:")
            if not captured:
                continue
            val: Any = captured
            cat = "environment_fact" if key == "home_city" else "user_preference"
        else:
            val = value
            cat = "user_preference"
        seen.add(key)
        facts.append(
            {
                "category": cat,
                "key": key,
                "value": val,
                "confidence_score": 0.85,
            }
        )
    # Generic "remember that X=Y" / "prefer X"
    for m in re.finditer(
        r"\b(?:remember|note)\s+(?:that\s+)?([a-z_][a-z0-9_]{1,40})\s*(?:=|is|:)\s*"
        r"([^\n,;.]{1,120})",
        blob,
        re.I,
    ):
        key = m.group(1).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            {
                "category": "environment_fact",
                "key": key,
                "value": m.group(2).strip(),
                "confidence_score": 0.7,
            }
        )
    return facts


def make_hydrate_memory_node(
    store: EpisodicMemoryStore | None = None,
    *,
    db_path: str | Path | None = None,
) -> Callable[[ReactGraphState], dict[str, Any]]:
    """Build hydrate node; queries store for user-input keywords → memory_context."""

    def hydrate_memory_node(state: ReactGraphState) -> dict[str, Any]:
        mem = store if store is not None else get_episodic_store(db_path)
        query = _extract_user_text(state)
        matches = mem.search_facts(query) if query else []
        # Always staple active preferences so dark-mode etc. survive sparse queries.
        prefs = mem.get_all_preferences()
        context: dict[str, Any] = {
            "preferences": prefs,
            "matches": matches,
        }
        # Flat convenience keys for callers / tests (prefer_dark_mode etc.).
        for k, v in prefs.items():
            context[k] = v
        for fact in matches:
            k = str(fact.get("key") or "")
            if k and k not in context:
                raw = fact.get("value")
                try:
                    context[k] = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError, json.JSONDecodeError):
                    context[k] = raw
        return {
            "memory_context": context,
            "current_agent": "MemoryHydrate",
        }

    return hydrate_memory_node


def make_consolidate_memory_node(
    store: EpisodicMemoryStore | None = None,
    *,
    db_path: str | Path | None = None,
    consolidate_llm: ConsolidateLLM | None = None,
) -> Callable[[ReactGraphState], dict[str, Any]]:
    """Build consolidate node; extracts facts after successful task halt."""

    def consolidate_memory_node(state: ReactGraphState) -> dict[str, Any]:
        # Skip non-successful / fail-closed / empty outcomes.
        final = str(state.get("final_raw") or "")
        if final.startswith("FAIL_CLOSED"):
            return {"current_agent": "MemoryConsolidate"}
        if state.get("execution_error"):
            return {"current_agent": "MemoryConsolidate"}

        mem = store if store is not None else get_episodic_store(db_path)
        user_text = _extract_user_text(state)
        assistant_text = _extract_assistant_text(state)
        facts: list[dict[str, Any]] = []
        if consolidate_llm is not None:
            try:
                facts = list(consolidate_llm(user_text, assistant_text) or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("consolidate_llm failed (%s); using heuristic", exc)
                facts = []
        if not facts:
            facts = heuristic_extract_facts(user_text, assistant_text)

        written: list[str] = []
        for fact in facts:
            try:
                cat = str(fact.get("category") or "environment_fact")
                key = str(fact.get("key") or "").strip()
                if not key:
                    continue
                conf = float(fact.get("confidence_score") or 0.8)
                mem.add_fact(
                    cat,
                    key,
                    fact.get("value"),
                    confidence_score=conf,
                )
                written.append(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("add_fact skipped (%s): %s", fact, exc)

        ctx = dict(state.get("memory_context") or {})
        if written:
            ctx["consolidated_keys"] = written
            # Refresh preferences into context.
            for k, v in mem.get_all_preferences().items():
                ctx[k] = v
            ctx["preferences"] = mem.get_all_preferences()

        return {
            "memory_context": ctx,
            "current_agent": "MemoryConsolidate",
        }

    return consolidate_memory_node


hydrate_memory_node = make_hydrate_memory_node()
consolidate_memory_node = make_consolidate_memory_node()
