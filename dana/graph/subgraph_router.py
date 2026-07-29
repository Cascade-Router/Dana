"""Autonomous sub-graph local retries — bypass supervisor until exhausted."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dana.graph.buffer import get_raw_trace, store_raw_trace
from dana.graph.nodes.critic import is_fatal_execution_error

DEFAULT_MAX_SUBGRAPH_RETRIES = 2

# Canonical node names for the injectable subgraph retry corridor.
SUBGRAPH_NODE = "subgraph"
SUPERVISOR_NODE = "supervisor"
BUMP_SUBGRAPH_RETRY = "bump_subgraph_retry"
ESCALATE_SUBGRAPH = "escalate_subgraph"


def _max_retries(state: dict[str, Any]) -> int:
    raw = state.get("max_subgraph_retries")
    if raw is None:
        return DEFAULT_MAX_SUBGRAPH_RETRIES
    return int(raw)


def _retry_count(state: dict[str, Any]) -> int:
    return int(state.get("subgraph_retry_count") or 0)


def _error_payload(state: dict[str, Any]) -> BaseException | str | None:
    """Best-effort error object/string for fatal classification."""
    if state.get("fatal_block"):
        return "fatal_block"
    trace = get_raw_trace(state)
    if not trace:
        err = state.get("execution_error")
        return err if err is not None and str(err).strip() else None
    exc_type = str(trace.get("exception_type") or "")
    exc_msg = str(trace.get("exception_message") or "")
    tb = str(trace.get("traceback") or "")
    # Prefer type+message so ``is_fatal_execution_error`` can match type tokens.
    combined = f"{exc_type}: {exc_msg}".strip(": ").strip()
    if tb and tb not in combined:
        combined = f"{combined}\n{tb}" if combined else tb
    return combined or None


def has_subgraph_failure(state: dict[str, Any]) -> bool:
    """True when the zero-copy buffer (or execution_error) records a failure."""
    return _error_payload(state) is not None


def resolve_subgraph_execution(
    state: dict[str, Any],
    *,
    subgraph_node: str = SUBGRAPH_NODE,
    supervisor_node: str = SUPERVISOR_NODE,
    end_node: str = "__end__",
) -> tuple[str, dict[str, Any]]:
    """Decide local retry vs supervisor escalate.

    Returns ``(next_node, state_patch)``.

    - Non-fatal failure and ``subgraph_retry_count < max_subgraph_retries``:
      increment count and route back to the local sub-graph node.
    - Exhausted retries or fatal: reset count to 0 and escalate to supervisor
      (``raw_state_buffer`` is left intact for re-planning).
    - No failure recorded: reset count and return ``end_node`` (bypass supervisor).
    """
    if not has_subgraph_failure(state):
        return end_node, {"subgraph_retry_count": 0}

    err = _error_payload(state)
    fatal = bool(state.get("fatal_block")) or is_fatal_execution_error(err)
    count = _retry_count(state)
    max_r = _max_retries(state)

    if not fatal and count < max_r:
        return subgraph_node, {"subgraph_retry_count": count + 1}

    return supervisor_node, {"subgraph_retry_count": 0}


def route_subgraph_execution(state: dict[str, Any]) -> str:
    """LangGraph conditional-edge helper (node name only).

    Returns ``bump_subgraph_retry`` when a local retry is allowed, otherwise
    ``escalate_subgraph`` (caller wires escalate → supervisor on failure, or
    END on success after the reset node clears the counter).
    """
    if not has_subgraph_failure(state):
        return ESCALATE_SUBGRAPH

    err = _error_payload(state)
    fatal = bool(state.get("fatal_block")) or is_fatal_execution_error(err)
    if not fatal and _retry_count(state) < _max_retries(state):
        return BUMP_SUBGRAPH_RETRY
    return ESCALATE_SUBGRAPH


def route_after_escalate(state: dict[str, Any]) -> str:
    """After reset: failure → supervisor; clean success → END."""
    from langgraph.graph import END

    if has_subgraph_failure(state):
        return SUPERVISOR_NODE
    return END


def bump_subgraph_retry(state: dict[str, Any]) -> dict[str, Any]:
    """Node: increment ``subgraph_retry_count`` before re-entering the sub-graph."""
    return {"subgraph_retry_count": _retry_count(state) + 1}


def escalate_subgraph(state: dict[str, Any]) -> dict[str, Any]:
    """Node: reset retry counter; flush ``[PENDING]`` ledger on failure escalate."""
    if has_subgraph_failure(state):
        try:
            from dana_security.ledger_writer import write_escalation_ticket

            fatal = bool(state.get("fatal_block")) or is_fatal_execution_error(
                _error_payload(state)
            )
            reason = "subgraph_fatal" if fatal else "subgraph_retries_exhausted"
            meta = write_escalation_ticket(
                state,
                reason=reason,
                objective=(
                    "Sub-graph escalation: fatal block"
                    if fatal
                    else "Sub-graph escalation: local retries exhausted"
                ),
                recommended_fix=(
                    "Supervisor re-plan using intact raw_state_buffer; "
                    "fix the underlying OS/dependency/code fault before "
                    "re-entering the sub-graph."
                ),
            )
            if not meta.get("ok"):
                import logging

                logging.getLogger(__name__).warning(
                    "escalate_subgraph: ledger flush failed (%s)",
                    meta.get("error"),
                )
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "escalate_subgraph: ledger writer unavailable (%s)", exc
            )
    return {"subgraph_retry_count": 0}


def apply_subgraph_failure(
    state: dict[str, Any],
    exception: BaseException | str,
    context_metadata: dict[str, Any] | None = None,
    *,
    fatal: bool | None = None,
) -> dict[str, Any]:
    """Store raw trace and optionally mark ``fatal_block`` (composable patch)."""
    patch = store_raw_trace(state, exception, context_metadata)
    if fatal is None:
        fatal = is_fatal_execution_error(exception)
    patch["fatal_block"] = bool(fatal)
    if not fatal and "execution_error" not in patch:
        # Keep a short pointer for corridors that still key off execution_error;
        # the full stack lives only in raw_state_buffer.
        if isinstance(exception, BaseException):
            patch["execution_error"] = f"{type(exception).__name__}: {exception}"
        else:
            patch["execution_error"] = str(exception)
    return patch


def compile_subgraph_retry_graph(
    subgraph_node_fn: Callable[..., Any],
    supervisor_node_fn: Callable[..., Any] | None = None,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compile a minimal injectable subgraph for local retries + supervisor escalate.

    Topology::

        START → subgraph ─(failure, retries left)→ bump_subgraph_retry → subgraph
                       ─(exhausted / fatal)→ escalate_subgraph → supervisor → END
                       ─(success)→ escalate_subgraph → END

    Does not touch HITL, critic REPL, memory hydrate/consolidate, or ToolForge.
    """
    from langgraph.graph import END, START, StateGraph

    from dana.schema import ReactGraphState

    def _default_supervisor(state: dict[str, Any]) -> dict[str, Any]:
        return {"current_agent": "Supervisor"}

    workflow = StateGraph(ReactGraphState)
    workflow.add_node(SUBGRAPH_NODE, subgraph_node_fn)
    workflow.add_node(BUMP_SUBGRAPH_RETRY, bump_subgraph_retry)
    workflow.add_node(ESCALATE_SUBGRAPH, escalate_subgraph)
    workflow.add_node(SUPERVISOR_NODE, supervisor_node_fn or _default_supervisor)

    workflow.add_edge(START, SUBGRAPH_NODE)
    workflow.add_conditional_edges(
        SUBGRAPH_NODE,
        route_subgraph_execution,
        {
            BUMP_SUBGRAPH_RETRY: BUMP_SUBGRAPH_RETRY,
            ESCALATE_SUBGRAPH: ESCALATE_SUBGRAPH,
        },
    )
    workflow.add_edge(BUMP_SUBGRAPH_RETRY, SUBGRAPH_NODE)
    workflow.add_conditional_edges(
        ESCALATE_SUBGRAPH,
        route_after_escalate,
        {
            SUPERVISOR_NODE: SUPERVISOR_NODE,
            END: END,
        },
    )
    workflow.add_edge(SUPERVISOR_NODE, END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


__all__ = (
    "BUMP_SUBGRAPH_RETRY",
    "DEFAULT_MAX_SUBGRAPH_RETRIES",
    "ESCALATE_SUBGRAPH",
    "SUBGRAPH_NODE",
    "SUPERVISOR_NODE",
    "apply_subgraph_failure",
    "bump_subgraph_retry",
    "compile_subgraph_retry_graph",
    "escalate_subgraph",
    "has_subgraph_failure",
    "resolve_subgraph_execution",
    "route_after_escalate",
    "route_subgraph_execution",
)
