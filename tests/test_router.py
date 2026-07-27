"""LangGraph HITL routing corridor — unapproved tickets fail closed.

Exercises production routing helpers in ``donna.agentic_react_graph`` and
``donna.middleware.hitl_ticket.decision_is_approved`` so pending / denied /
missing decisions never route to tool execution.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from donna.agentic_react_graph import (
    _route_after_jason_review,
    _route_after_ticket_approval,
    _route_after_ticket_validate,
)
from donna.middleware.hitl_ticket import decision_is_approved


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"halt": True}, END),
        ({"halt": True, "ticket_validated": True}, END),
        ({"halt": False}, "tools"),
        ({}, "tools"),  # missing halt → continue (approved corridor)
    ],
)
def test_route_after_ticket_approval_fail_closed(
    state: dict[str, Any], expected: Any
) -> None:
    """Deny / halt must END; only non-halt may proceed to tools."""
    assert _route_after_ticket_approval(state) == expected


def test_denied_ticket_never_routes_to_tools() -> None:
    """Explicit fail-closed contract: halt=True is the deny corridor."""
    denied = {"halt": True, "last_obs": "DENIED: ticket cancelled by operator"}
    assert _route_after_ticket_approval(denied) == END
    assert _route_after_ticket_approval(denied) != "tools"


@pytest.mark.parametrize(
    "decision",
    [
        None,
        False,
        {},
        {"approved": False},
        {"approved": False, "action": "deny"},
        {"action": "deny"},
        {"action": "pending"},
        {"status": "PENDING_USER_APPROVAL"},
        "deny",
        "pending",
        "DENIED",
        0,
        [],
    ],
)
def test_decision_is_approved_fails_closed_for_unapproved(decision: Any) -> None:
    """Pending, denied, empty, and unknown values are never treated as approved."""
    assert decision_is_approved(decision) is False


@pytest.mark.parametrize(
    "decision",
    [
        True,
        {"approved": True},
        {"approved": True, "action": "approve"},
        {"action": "approve"},
        "approve",
        "approved",
    ],
)
def test_decision_is_approved_true_only_for_explicit_approve(decision: Any) -> None:
    assert decision_is_approved(decision) is True


def test_routing_corridor_validate_and_jason_gates() -> None:
    """Unvalidated / halted tickets stay off the tools path."""
    assert _route_after_ticket_validate({"halt": True}) == END
    assert _route_after_ticket_validate({"ticket_validated": False}) == "agent"
    assert (
        _route_after_ticket_validate({"ticket_validated": True})
        == "jason_ticket_review"
    )
    assert _route_after_jason_review({"halt": True}) == END
    assert _route_after_jason_review({"halt": False}) == "ticket_approval"
