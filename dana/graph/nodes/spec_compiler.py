"""Spec Compiler Agent — raw intent → strict ``/broker`` macro (or REJECT)."""

from __future__ import annotations

import re
from typing import Any

SPEC_COMPILER_NODE = "spec_compiler"

_REJECT_PREFIX = "REJECT:"
_BROKER_PREFIX_RE = re.compile(r"^\s*/broker\b", re.I)
_EPIC_MARK_RE = re.compile(r"\bepic\s*\d+\s*:", re.I)

_SPEC_COMPILER_SYSTEM = """You are the Dānā Spec Compiler Agent for a local Meta-Broker.

Analyze the user's intent and respond with EXACTLY one of:
1) A single line starting with REJECT: followed by a short reason, OR
2) A strict /broker specification (may span multiple lines).

Rules for REJECT:
- Vague goals with no concrete deliverable (e.g. "make it better", "build an AI").
- Requires unsupported third-party packages (torch, tensorflow, langchain, fastapi,
  django, pandas, numpy, requests, selenium, opencv, etc.) unless the user
  explicitly asked for that package by name AND the local environment mandate
  allows it — default is REJECT for non-stdlib.
- Needs cloud APIs, paid services, GUI automation of external SaaS, or hardware
  the local agent cannot access.
- Unbounded research / open-ended essays with no code artifact.
- If the task is clearly too large / multi-system for a local 7B model (e.g. full
  ROS2 fleet controller + DB + web UI), respond EXACTLY:
  REJECT: Task too complex for local model

Rules for compiled /broker specs:
- Start with `/broker` then sequential `Epic N:` blocks (usually 2–3 epics).
- Force Python Standard Library only (collections, asyncio, json, math, os,
  pathlib, re, dataclasses, typing, unittest/pytest for tests), unless an MCP
  tool listed below is required — then name the MCP tool explicitly in the epic.
- Name exact files (e.g. `astar_planner.py`, `tests/test_astar.py`).
- Specify exact class/method signatures.
- For graph/search algorithms require a `visited` set (or equivalent) to bound loops.
- Epic 3 (or the test epic) must use a targeted pytest command and require
  `@pytest.mark` / `asyncio.wait_for(..., timeout=5.0)` when async.
- Keep the whole job small enough for a single-pass local model (no repair loops).
- No markdown fences, no prose before `/broker`.

Example output:
/broker Epic 1: Write grid_map.py with class GridMap(width, height, obstacles=None) \
and methods is_valid_cell(x, y), get_neighbors(x, y) using only the stdlib. \
Epic 2: Write astar_planner.py importing GridMap; class AStarPlanner with \
plan_path(grid, start, goal) using A* and an explicit visited/closed set. \
Epic 3: Write tests/test_astar.py with pytest covering obstacles, start==goal, \
and unreachable goals; each test must finish under 5 seconds.
"""


def _system_prompt_with_mcp() -> str:
    try:
        from dana.mcp.client import format_mcp_tools_block

        return _SPEC_COMPILER_SYSTEM + "\n\n" + format_mcp_tools_block()
    except Exception:  # noqa: BLE001
        return _SPEC_COMPILER_SYSTEM


def _normalize_compiler_output(out: str) -> str:
    out = str(out or "").strip()
    if "```" in out:
        parts = out.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.lower().startswith("python"):
                chunk = chunk[6:].strip()
            if is_reject_spec(chunk) or is_broker_ready_spec(chunk):
                out = chunk
                break
    for marker in ("REJECT:", "/broker", "/BROKER", "Epic 1:"):
        idx = out.find(marker)
        if idx >= 0:
            out = out[idx:].strip()
            break
    if is_reject_spec(out) or is_broker_ready_spec(out):
        m = _BROKER_PREFIX_RE.search(out)
        if m:
            out = out[m.start() :].strip()
        elif is_broker_ready_spec(out) and not is_reject_spec(out):
            out = f"/broker {out}"
    return out


def is_reject_spec(text: str) -> bool:
    return str(text or "").strip().upper().startswith("REJECT:")


def is_broker_ready_spec(text: str) -> bool:
    s = str(text or "").strip()
    if not s or is_reject_spec(s):
        return False
    return bool(_BROKER_PREFIX_RE.search(s) or _EPIC_MARK_RE.search(s))


PENDING_USER_APPROVAL = "PENDING_USER_APPROVAL"

_EPIC_SPLIT_RE = re.compile(r"(?=(?:\bEpic\s*\d+\s*:))", re.I)


def hitl_spec_approval_enabled() -> bool:
    """Human-in-the-loop Approve & Run gate (default on; set DONNA_HITL_SPEC_APPROVAL=0 to skip)."""
    import os

    raw = str(os.environ.get("DONNA_HITL_SPEC_APPROVAL", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def parse_epics_from_spec(spec: str) -> list[dict[str, Any]]:
    """Extract ``Epic N:`` blocks from a compiled ``/broker`` macro."""
    text = str(spec or "").strip()
    if not text:
        return []
    # Drop leading /broker token for splitting.
    body = _BROKER_PREFIX_RE.sub("", text, count=1).strip()
    chunks = [c.strip() for c in _EPIC_SPLIT_RE.split(body) if c and c.strip()]
    epics: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        m = re.match(r"Epic\s*(\d+)\s*:\s*(.*)$", chunk, re.I | re.S)
        if m:
            epics.append(
                {
                    "id": int(m.group(1)),
                    "title": m.group(2).strip().split(".")[0][:80],
                    "body": m.group(2).strip(),
                }
            )
        else:
            epics.append({"id": i, "title": chunk[:80], "body": chunk})
    return epics


def build_spec_approval_payload(
    *,
    compiled_spec: str,
    raw_intent: str = "",
) -> dict[str, Any]:
    """Structured HITL payload emitted before Meta-Broker dispatch."""
    spec = str(compiled_spec or "").strip()
    return {
        "type": "spec_approval_request",
        "status": PENDING_USER_APPROVAL,
        "compiled_spec": spec,
        "raw_intent": str(raw_intent or "").strip(),
        "epics": parse_epics_from_spec(spec),
    }


def _heuristic_compile(user_input: str) -> str:
    """Offline / hermetic fallback when the LLM is unavailable."""
    text = str(user_input or "").strip()
    if not text:
        return "REJECT: Empty intent — provide a concrete coding goal."

    low = text.lower()
    vague = (
        len(text) < 12,
        low in {"help", "hi", "hello", "test", "fix it", "make it better"},
        re.fullmatch(r"(please\s+)?(build|make|create)\s+(something|an?\s+app)", low),
    )
    if any(vague):
        return "REJECT: Intent is too vague — name the artifact, API, and tests."

    banned = (
        "tensorflow",
        "pytorch",
        "torch",
        "langchain",
        "fastapi",
        "django",
        "flask",
        "pandas",
        "numpy",
        "opencv",
        "selenium",
        "kubernetes",
        "aws ",
        "gcp ",
        "azure ",
    )
    if any(b in low for b in banned):
        return (
            "REJECT: Request depends on unsupported third-party / cloud stack; "
            "rephrase using Python Standard Library only."
        )

    # Multi-system complexity → exact marker so ModelProvider can cloud-ramp.
    complex_markers = (
        "ros2 fleet",
        "full stack web",
        "microservices mesh",
        "distributed consensus cluster",
        "train a neural",
        "fine-tune llm",
    )
    if any(m in low for m in complex_markers):
        from dana.core.model_provider import complexity_reject_marker

        return complexity_reject_marker()

    # Light pattern templates for common planner / bus / cache intents.
    if any(k in low for k in ("a-star", "a*", "astar", "grid planner", "pathfind")):
        return (
            "/broker Epic 1: Write grid_map.py containing class GridMap "
            "initialized with width, height, and optional obstacle tuples; "
            "implement is_valid_cell(x, y) and get_neighbors(x, y) using only "
            "the Python Standard Library. "
            "Epic 2: Write astar_planner.py that imports GridMap; implement "
            "class AStarPlanner with plan_path(grid, start, goal) using A* "
            "with an explicit visited/closed set to prevent infinite loops; "
            "return None when unreachable. "
            "Epic 3: Write tests/test_astar.py using pytest to verify path "
            "generation around obstacles, start-equals-goal, and unreachable "
            "goals; each test must complete in under 5 seconds "
            "(python -m pytest tests/test_astar.py -q)."
        )
    if "token bucket" in low or "async" in low and "bus" in low:
        return (
            "/broker Epic 1: Write token_bucket.py with async class TokenBucket "
            "(capacity: int, refill_rate: float) and async consume(amount=1) that "
            "refills via time.monotonic()/loop.time(), never infinite-loops when "
            "refill_rate<=0, and uses asyncio.sleep only for positive wait times "
            "(stdlib asyncio only). "
            "Epic 2: Write async_bus.py importing TokenBucket; class EventBus with "
            "subscribe(topic, handler_coroutine) and async publish(topic, payload) "
            "using a dedicated TokenBucket per topic. "
            "Epic 3: Write tests/test_async_bus.py with pytest; wrap async bodies in "
            "asyncio.wait_for(..., timeout=5.0); assert burst rate-limiting via "
            "timestamps (python -m pytest tests/test_async_bus.py -q)."
        )

    # Generic three-epic scaffold from free text.
    slug = re.sub(r"[^a-z0-9]+", "_", low)[:32].strip("_") or "module"
    mod = f"{slug}.py" if not slug.endswith(".py") else slug
    test = f"tests/test_{slug}.py"
    return (
        f"/broker Epic 1: Write {mod} implementing the user's goal using ONLY "
        f"the Python Standard Library. Expose clear class/function signatures "
        f"described by: {text[:240]!r}. Bound any search/graph loops with a "
        f"visited set. "
        f"Epic 2: Write helper or integration module only if required by Epic 1; "
        f"otherwise extend {mod} with docstrings and edge-case handling "
        f"(still stdlib-only). "
        f"Epic 3: Write {test} with pytest covering happy path, one edge case, "
        f"and one failure/unreachable case; enforce per-test runtime under 5s "
        f"(python -m pytest {test} -q)."
    )


def compile_user_spec(user_input: str) -> str:
    """Translate raw user intent into a strict ``/broker`` spec or ``REJECT:``.

    Already-valid ``/broker`` / ``Epic N:`` macros are returned unchanged
    (stdlib stamp may be applied by callers).
    """
    text = str(user_input or "").strip()
    if not text:
        return "REJECT: Empty intent — provide a concrete coding goal."
    if is_reject_spec(text):
        return text if text.upper().startswith("REJECT:") else f"REJECT: {text}"
    if is_broker_ready_spec(text):
        m = _BROKER_PREFIX_RE.search(text)
        if m:
            return text[m.start() :].strip()
        if _EPIC_MARK_RE.search(text):
            return f"/broker {text}"
        return text

    try:
        from dana.core.model_provider import ModelProvider

        provider = ModelProvider()
        messages = [
            {"role": "system", "content": _system_prompt_with_mcp()},
            {"role": "user", "content": f"USER INTENT:\n{text}"},
        ]
        raw = provider.complete_with_complexity_fallback(
            messages,
            num_predict=512,
            temperature=0.1,
        )
        out = _normalize_compiler_output(str(raw or ""))
        if is_reject_spec(out) or is_broker_ready_spec(out):
            print(
                f"[SpecCompiler] provider={provider.last_provider} "
                f"chars={len(out)}",
                flush=True,
            )
            return out
    except Exception as exc:  # noqa: BLE001
        print(
            f"[SpecCompiler] LLM compile failed ({type(exc).__name__}: {exc}); "
            "using heuristic",
            flush=True,
        )

    # Heuristic path — still honor cloud fallback for complexity rejects.
    heur = _heuristic_compile(text)
    try:
        from dana.core.model_provider import (
            ModelProvider,
            cloud_fallback_enabled,
            is_complexity_reject,
        )

        if is_complexity_reject(heur) and cloud_fallback_enabled():
            provider = ModelProvider(prefer="cloud")
            raw = provider.complete(
                [
                    {"role": "system", "content": _system_prompt_with_mcp()},
                    {"role": "user", "content": f"USER INTENT:\n{text}"},
                ],
                num_predict=768,
                temperature=0.1,
                allow_cloud=True,
                response_mime_type="text/plain",
            )
            out = _normalize_compiler_output(str(raw or ""))
            if is_broker_ready_spec(out) or is_reject_spec(out):
                return out
    except Exception:  # noqa: BLE001
        pass
    return heur


def make_spec_compiler_node():
    """LangGraph node: compile ``macro_intent`` / ``user_prompt`` before broker."""

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        raw = str(
            state.get("macro_intent") or state.get("user_prompt") or ""
        ).strip()
        compiled = compile_user_spec(raw)
        if is_reject_spec(compiled):
            print(f"[SpecCompiler] {compiled}", flush=True)
            return {
                "macro_intent": compiled,
                "broker_phase": "done",
                "status": "ABORTED",
                "error": compiled,
                "final_response": compiled,
                "epic_log": list(state.get("epic_log") or [])
                + [f"spec_compiler: {compiled}"],
            }
        print(
            f"[SpecCompiler] compiled chars={len(compiled)} "
            f"broker_ready={is_broker_ready_spec(compiled)}",
            flush=True,
        )
        return {
            "macro_intent": compiled,
            "broker_phase": "plan",
            "status": "planning",
            "epic_log": list(state.get("epic_log") or [])
            + ["spec_compiler: compiled user intent → /broker spec"],
        }

    return _node


def route_after_spec_compiler(state: dict[str, Any]) -> str:
    from dana.graph.nodes.broker import BROKER_NODE, END_ROUTE

    if is_reject_spec(str(state.get("macro_intent") or "")):
        return END_ROUTE
    if str(state.get("status") or "").upper() == "ABORTED":
        return END_ROUTE
    return BROKER_NODE


__all__ = (
    "SPEC_COMPILER_NODE",
    "PENDING_USER_APPROVAL",
    "build_spec_approval_payload",
    "compile_user_spec",
    "hitl_spec_approval_enabled",
    "is_broker_ready_spec",
    "is_reject_spec",
    "make_spec_compiler_node",
    "parse_epics_from_spec",
    "route_after_spec_compiler",
)
