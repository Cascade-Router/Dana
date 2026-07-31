"""Pre-answer planning corridor — Plan-Then-Execute for the ReAct graph.

Hydrates AgentState with tool + desktop context, forces a structured plan
before tool execution / final speech, then advances required tools into the
existing HITL corridor (ticket_validate → jason → ticket_approval → tools).

The structured plan lives in ``execution_plan`` / ``always_include`` state only.
It is NOT injected into ReAct chat prompts — tool calling is enforced at the
inference layer via ``llm.bind_tools(..., strict=True)`` (+ ``tool_choice``).
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from dana.schema import ReactGraphState

# Identity block for system prompt builders (persona / context — no plan JSON).
DANA_CONTROL_PLANE_IDENTITY = (
    "You are Dānā, a local Windows control plane with direct access to desktop "
    "tools (Florence-2 vision, Win32 ROI, HITL ticket gate). "
    "Do not issue generic refusals for desktop or window requests."
)

# Offline planner schema (state machine only — not ReAct chat text).
_PLANNER_FIELDS = (
    "intended_goal",
    "environment_assessment",
    "required_tools",
    "execution_steps",
)

# Desktop / window / ticket intents that must leave lightweight chat.
_DESKTOP_PLAN_RE = re.compile(
    r"("
    r"\b(?:summarize|summary|describe)\b.*\b(?:window|screen|desktop|display)\b|"
    r"\b(?:active|foreground)\s+window\b|"
    r"\b(?:on[- ]?screen|screenshot|ocr)\b|"
    r"\b(?:desktop|window)\b.*\b(?:log|ticket|ledger)\b|"
    r"\b(?:log|create|file|write)\b.*\b(?:ticket|ledger)\b|"
    r"\bdraft_cursor_prompt\b|"
    r"\b(?:florence|vision|ui\s+grounding)\b|"
    # App management / window-tab inspection / log inspection (chat soft-drop fixes).
    r"\b(?:close|quit|kill|launch|open)\s+(?:[a-zA-Z0-9_-]+\s+)?"
    r"(?:app|application|discord|browser|window|program)\b|"
    r"\b(?:how\s+many|list|show|count)\s+(?:windows|tabs|apps)\b|"
    r"\b(?:open|show|read|retrack|view)\s+.*\b(?:logs?|runtime\.log|conversation\.log)\b"
    r")",
    re.IGNORECASE,
)

_VISION_PLAN_RE = re.compile(
    r"("
    r"\b(?:summarize|summary|describe|read|ocr|see|look|watch|screen|window|"
    r"desktop|display|on[- ]?screen|active\s+window|foreground)\b"
    r")",
    re.IGNORECASE,
)

_TICKET_PLAN_RE = re.compile(
    r"("
    r"\b(?:ticket|ledger|patch\s*ledger|log\s+(?:a\s+)?ticket|draft_cursor|"
    r"cursor\s+prompt|desktop\s+log)\b"
    r")",
    re.IGNORECASE,
)

_DONNA_TITLE_MARKERS = (
    "donna",
    "dānā",
    "dana",
    "control dashboard",
    "live trace",
)


def desktop_plan_intent(text: str) -> bool:
    """True when the utterance needs the Plan-Then-Execute tool corridor."""
    return bool(_DESKTOP_PLAN_RE.search(text or ""))


def hydrate_tool_catalog(*, limit: int = 24) -> list[str]:
    """Return known tool ids for planner context (best-effort)."""
    ids: list[str] = []
    try:
        from dana.tools.registry import get_tool_registry

        ids = list(get_tool_registry().as_spec_dict().keys())
    except Exception:  # noqa: BLE001
        try:
            from dana.tools.broker import get_broker

            ids = list(get_broker().registry.keys())
        except Exception:  # noqa: BLE001
            ids = []
    # Prefer desktop-critical tools near the front of the card.
    priority = (
        "analyze_visual_context",
        "ocr_with_region",
        "capture_and_analyze_screen",
        "draft_cursor_prompt",
        "read_local_file",
        "file_editor",
    )
    ordered = [t for t in priority if t in ids]
    ordered.extend(t for t in ids if t not in ordered)
    return ordered[: max(1, int(limit))]


def active_window_metadata() -> dict[str, Any]:
    """Best-effort foreground window info; excludes Donna's own UI titles."""
    meta: dict[str, Any] = {
        "platform": sys.platform,
        "title": "",
        "pid": None,
        "excluded_self": False,
        "available": False,
    }
    if sys.platform != "win32":
        return meta
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return meta
        length = int(user32.GetWindowTextLengthW(hwnd))
        buf = ctypes.create_unicode_buffer(length + 2)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = (buf.value or "").strip()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        meta["title"] = title
        meta["pid"] = int(pid.value) if pid.value else None
        meta["available"] = True
        low = title.lower()
        if any(m in low for m in _DONNA_TITLE_MARKERS):
            meta["excluded_self"] = True
            meta["title"] = ""
            meta["note"] = "Foreground window is Donna UI — ignored for planning."
    except Exception as exc:  # noqa: BLE001
        meta["error"] = type(exc).__name__
    return meta


def build_structured_plan(
    user_text: str,
    *,
    tool_ids: list[str] | None = None,
    window_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic structured plan (Goal / Environment / Tools / Steps)."""
    text = (user_text or "").strip()
    tools = list(tool_ids or hydrate_tool_catalog())
    window = window_meta if isinstance(window_meta, dict) else active_window_metadata()
    required: list[str] = []
    steps: list[str] = []

    wants_vision = bool(_VISION_PLAN_RE.search(text))
    wants_ticket = bool(_TICKET_PLAN_RE.search(text))

    if wants_vision:
        if "ocr_with_region" in tools and re.search(
            r"\b(?:ocr|text|read|label|button|ui)\b", text, re.I
        ):
            required.append("ocr_with_region")
            steps.append("OCR / UI-ground the active screen with Florence-2 (ocr_with_region).")
        elif "analyze_visual_context" in tools:
            required.append("analyze_visual_context")
            steps.append("Capture and summarize the active window via analyze_visual_context(source=screen).")
        elif "capture_and_analyze_screen" in tools:
            required.append("capture_and_analyze_screen")
            steps.append("Screenshot + analyze the desktop via capture_and_analyze_screen.")

    if wants_ticket and "draft_cursor_prompt" in tools:
        required.append("draft_cursor_prompt")
        steps.append(
            "Log a desktop/HITL ticket with draft_cursor_prompt (goes through ticket gate)."
        )

    # Deduplicate while preserving order.
    required = list(dict.fromkeys(required))
    if not steps:
        steps = [
            "Assess user intent against the bound tool catalog.",
            "Call the minimum tools required; then speak a short confirmation.",
        ]

    env_bits = []
    title = str(window.get("title") or "").strip()
    if title:
        env_bits.append(f"Active window title: {title}")
    elif window.get("excluded_self"):
        env_bits.append("Active window is Donna UI (excluded); use screen capture tools.")
    else:
        env_bits.append("Active window title unavailable; prefer screen capture tools.")
    if tools:
        env_bits.append("Bound tools sample: " + ", ".join(tools[:12]))

    return {
        "intended_goal": text or "(empty)",
        "environment_assessment": " | ".join(env_bits),
        "required_tools": required,
        "execution_steps": steps,
        "status": "planned",
        "plan_index": 0,
    }


def format_plan_block(plan: dict[str, Any]) -> str:
    """Serialize plan for traces/logs — not for ReAct chat injection."""
    payload = {k: plan.get(k) for k in _PLANNER_FIELDS}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_user_text(state: ReactGraphState | dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role in {"human", "user"} and isinstance(content, str) and content.strip():
            # Strip injected visual context tails if present.
            return content.split("\n\nVisual Context:", 1)[0].strip()
        if isinstance(msg, dict):
            if msg.get("role") == "user" and str(msg.get("content") or "").strip():
                return str(msg["content"]).split("\n\nVisual Context:", 1)[0].strip()
    return str(state.get("active_intent") or "").strip()


def planner_node(state: ReactGraphState) -> dict[str, Any]:
    """LangGraph node: hydrate context + attach structured plan (pre-answer).

    Plan-Then-Execute advances tools via ``execution_plan`` / ``always_include``.
    Does not inject STRUCTURED_PLAN JSON into ReAct chat messages.
    """
    user_text = _extract_user_text(state)
    tool_ids = hydrate_tool_catalog()
    window = active_window_metadata()
    plan = build_structured_plan(
        user_text, tool_ids=tool_ids, window_meta=window
    )
    try:
        from dana.ui.trace_bus import emit_trace_event

        emit_trace_event(
            "state_update",
            node="planner",
            message="Pre-answer plan ready",
            payload=json.dumps(
                {
                    "required_tools": plan.get("required_tools"),
                    "goal": (plan.get("intended_goal") or "")[:160],
                },
                ensure_ascii=False,
            )[:500],
        )
    except Exception:  # noqa: BLE001
        pass

    always = list(state.get("always_include") or [])
    for tid in plan.get("required_tools") or []:
        if tid not in always:
            always.append(tid)

    return {
        "execution_plan": plan,
        "plan_index": 0,
        "env_context": {
            "tools": tool_ids,
            "active_window": window,
        },
        "always_include": always,
        "current_agent": "Planner",
        "active_intent": user_text or state.get("active_intent") or "",
    }


def executor_node(state: ReactGraphState) -> dict[str, Any]:
    """LangGraph node: promote planned tools into the ReAct bind set.

    Does not bypass HITL — ``draft_cursor_prompt`` still flows
    agent → ticket_validate → jason → ticket_approval → tools.
    """
    plan = dict(state.get("execution_plan") or {})
    required = [str(t) for t in (plan.get("required_tools") or []) if str(t).strip()]
    always = list(state.get("always_include") or [])
    for tid in required:
        if tid not in always:
            always.append(tid)

    plan["status"] = "executing"
    try:
        from dana.ui.trace_bus import emit_trace_event

        emit_trace_event(
            "state_update",
            node="executor",
            message="Executor armed required tools",
            payload=",".join(required)[:300],
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "execution_plan": plan,
        "always_include": always,
        "current_agent": "Executor",
        "plan_index": int(state.get("plan_index") or 0),
    }


def planning_system_preamble() -> str:
    """Short identity for system prompt builders (no STRUCTURED_PLAN text)."""
    return DANA_CONTROL_PLANE_IDENTITY
