"""Workflow routing helpers / docs for the Donna ReAct corridor.

Topology (closed-loop verification)::

    START → hydrate_memory → planner → executor → agent
         ─(tool_calls)→ tools ─(error)→ critic → tools
                       ─(fatal / exhausted)→ fail_closed → END
                       ─(continue)→ agent
                       ─(halt / success)→ verifier
                            ─(verified)→ consolidate_memory → END
                            ─(fail, attempts < 3)→ agent
                            ─(fail, attempts ≥ 3)→ fail_closed → END
         ─(halt, no tools)→ consolidate_memory → END

HITL ticket corridor, completion_gate pending_synthesis, memory hydrate, and
subgraph retries are preserved; ToolForge gates and ``dana.paths`` are untouched.
"""

from __future__ import annotations

from typing import Any

from dana.graph.nodes.verifier import (
    MAX_VERIFICATION_ATTEMPTS,
    route_after_verifier,
)

# Public route labels used by compile_donna_react_graph conditional edges.
ROUTE_AGENT = "agent"
ROUTE_TOOLS = "tools"
ROUTE_CRITIC = "critic"
ROUTE_FAIL_CLOSED = "fail_closed"
ROUTE_VERIFIER = "verifier"
ROUTE_CONSOLIDATE = "consolidate_memory"


def should_enter_verifier(proposed: str, *, end_sentinel: str = "__end__") -> bool:
    """True when a post-tools route would END and must hit the verifier first."""
    if proposed == ROUTE_VERIFIER:
        return True
    if proposed == end_sentinel or str(proposed).upper() == "END":
        return True
    return False


def remap_execution_end_to_verifier(
    proposed: str,
    *,
    end_sentinel: str = "__end__",
) -> str:
    """Rewrite successful post-tools END → ``verifier`` (closed-loop gate)."""
    if should_enter_verifier(proposed, end_sentinel=end_sentinel):
        return ROUTE_VERIFIER
    return proposed


def verification_attempts(state: dict[str, Any] | None) -> int:
    """Read ``verification_result.attempts`` (0 when absent)."""
    vr = (state or {}).get("verification_result")
    if not isinstance(vr, dict):
        return 0
    try:
        return int(vr.get("attempts") or 0)
    except (TypeError, ValueError):
        return 0


__all__ = (
    "MAX_VERIFICATION_ATTEMPTS",
    "ROUTE_AGENT",
    "ROUTE_CONSOLIDATE",
    "ROUTE_CRITIC",
    "ROUTE_FAIL_CLOSED",
    "ROUTE_TOOLS",
    "ROUTE_VERIFIER",
    "remap_execution_end_to_verifier",
    "route_after_verifier",
    "should_enter_verifier",
    "verification_attempts",
)
