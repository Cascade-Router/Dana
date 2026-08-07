"""LangGraph HITL routing corridor — unapproved tickets fail closed.

Exercises production routing helpers in ``dana.agentic_react_graph`` and
``dana.middleware.hitl_ticket.decision_is_approved`` so pending / denied /
missing decisions never route to tool execution.

Also covers ``GridMap``/``RobotRouter`` BFS pathfinding (root-level flat
utility modules).
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from dana.agentic_react_graph import (
    _route_after_jason_review,
    _route_after_ticket_approval,
    _route_after_ticket_validate,
)
from dana.middleware.hitl_ticket import decision_is_approved
from grid_map import GridMap
from robot_router import RobotRouter


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


def test_path_around_obstacles():
    grid = GridMap(5, 5, obstacles={(1, 0), (1, 1), (1, 2)})
    router = RobotRouter()
    path = router.plan_path(grid, (0, 0), (2, 0))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    assert all(cell not in grid.obstacles for cell in path)
    # Path should step only to valid adjacent cells.
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert grid.is_valid_cell(*b)


def test_start_equals_goal():
    grid = GridMap(3, 3)
    router = RobotRouter()
    assert router.plan_path(grid, (1, 1), (1, 1)) == [(1, 1)]


def test_unreachable_goal_returns_none():
    # Wall seals goal from start.
    obstacles = {(1, 0), (1, 1), (1, 2)}
    grid = GridMap(3, 3, obstacles=obstacles)
    router = RobotRouter()
    assert router.plan_path(grid, (0, 1), (2, 1)) is None


def test_invalid_start_or_goal_returns_none():
    grid = GridMap(3, 3, obstacles={(0, 0)})
    router = RobotRouter()
    assert router.plan_path(grid, (0, 0), (2, 2)) is None
    assert router.plan_path(grid, (2, 2), (0, 0)) is None
