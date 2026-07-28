"""Plugin: dispatch_research_swarm — delegates to in-process swarm_dispatcher."""

from __future__ import annotations

from typing import Any

TOOL_ID = "dispatch_research_swarm"


def execute(query: str) -> str:
    from dana.tools.swarm_dispatcher import dispatch_research_swarm

    return dispatch_research_swarm(query)


def handle_tool_call(call: Any) -> str:
    from dana.tools.swarm_dispatcher import handle_tool_call as _dispatch

    return _dispatch(call)
