"""Zero-copy state buffer + autonomous sub-graph local retries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from dana.graph.buffer import get_raw_trace, store_raw_trace
from dana.graph.subgraph_router import (
    DEFAULT_MAX_SUBGRAPH_RETRIES,
    SUBGRAPH_NODE,
    SUPERVISOR_NODE,
    compile_subgraph_retry_graph,
    resolve_subgraph_execution,
    route_subgraph_execution,
)
from dana.schema import ReactGraphState


def _tmp_ledger(tmp_path: Path) -> str:
    path = tmp_path / "dana_security" / "patch_ledger.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_zero_copy_buffer_preserves_exact_raw_trace() -> None:
    """store_raw_trace must keep the full string — no truncation / summarization."""
    raw = (
        "Traceback (most recent call last):\n"
        '  File "worker.py", line 42, in run_subgraph\n'
        "    raise ValueError('boom-' + ('x' * 500))\n"
        "ValueError: boom-" + ("x" * 500)
    )
    state: dict[str, Any] = {"raw_state_buffer": {"prior": 1}}
    patch = store_raw_trace(
        state,
        raw,
        {"node": "subgraph", "attempt": 1},
    )
    assert "raw_state_buffer" in patch
    assert patch["raw_state_buffer"]["prior"] == 1
    last = patch["raw_state_buffer"]["last_error"]
    assert last["traceback"] == raw
    assert "boom-" + ("x" * 500) in last["traceback"]
    assert last["context"]["node"] == "subgraph"

    merged = {**state, **patch}
    fetched = get_raw_trace(merged)
    assert fetched is not None
    assert fetched["traceback"] == raw

    # Exception path also preserves format_exception output verbatim.
    try:
        raise RuntimeError("exact-runtime-marker-98765")
    except RuntimeError as exc:
        patch2 = store_raw_trace({}, exc, {"phase": "test"})
    tb2 = patch2["raw_state_buffer"]["last_error"]["traceback"]
    assert "exact-runtime-marker-98765" in tb2
    assert "Traceback (most recent call last)" in tb2
    assert "RuntimeError" in tb2


def test_transient_error_retries_locally_without_supervisor(tmp_path: Path) -> None:
    """Non-fatal failures retry locally up to N=2; supervisor never visited."""
    visits: list[str] = []
    attempts = {"n": 0}
    ledger = _tmp_ledger(tmp_path)

    def subgraph(state: ReactGraphState) -> dict[str, Any]:
        visits.append(SUBGRAPH_NODE)
        attempts["n"] += 1
        # Succeed on the second local retry (3rd attempt: initial + 2 retries).
        if attempts["n"] >= 3:
            return {
                "raw_state_buffer": {"last_error": None},
                "execution_error": None,
                "fatal_block": False,
                "current_agent": "SubGraph",
            }
        err = ValueError(f"transient-{attempts['n']}")
        return {
            **store_raw_trace(state, err, {"attempt": attempts["n"]}),
            "fatal_block": False,
            "execution_error": str(err),
            "current_agent": "SubGraph",
        }

    def supervisor(state: ReactGraphState) -> dict[str, Any]:
        visits.append(SUPERVISOR_NODE)
        return {"current_agent": "Supervisor"}

    graph = compile_subgraph_retry_graph(
        subgraph,
        supervisor,
        checkpointer=MemorySaver(),
    )
    seed: ReactGraphState = {
        "session_id": "subretry-1",
        "current_agent": "SubGraph",
        "active_intent": "local-retry",
        "subgraph_retry_count": 0,
        "max_subgraph_retries": DEFAULT_MAX_SUBGRAPH_RETRIES,
        "fatal_block": False,
        "raw_state_buffer": {},
        "patch_ledger_path": ledger,
    }
    out = graph.invoke(seed, config={"configurable": {"thread_id": "t-local"}})

    assert SUPERVISOR_NODE not in visits
    assert visits.count(SUBGRAPH_NODE) == 3
    assert int(out.get("subgraph_retry_count") or 0) == 0
    assert get_raw_trace(out) is None
    # Success escalate must not invent a PENDING ticket.
    assert not Path(ledger).is_file() or "[PENDING]" not in Path(ledger).read_text(
        encoding="utf-8"
    )


def test_exhausted_retries_bubble_to_supervisor_with_full_stack(tmp_path: Path) -> None:
    """After 2 failed local retries, escalate with un-truncated raw_state_buffer."""
    visits: list[str] = []
    long_marker = "STACK_MARKER_" + ("Z" * 800)
    ledger = _tmp_ledger(tmp_path)

    def subgraph(state: ReactGraphState) -> dict[str, Any]:
        visits.append(SUBGRAPH_NODE)
        n = visits.count(SUBGRAPH_NODE)
        try:
            raise KeyError(long_marker)
        except KeyError as exc:
            return {
                **store_raw_trace(
                    state,
                    exc,
                    {"attempt": n, "node": SUBGRAPH_NODE},
                ),
                "fatal_block": False,
                "execution_error": f"KeyError: {long_marker}",
                "current_agent": "SubGraph",
            }

    def supervisor(state: ReactGraphState) -> dict[str, Any]:
        visits.append(SUPERVISOR_NODE)
        # Supervisor must see the full zero-copy buffer for re-planning.
        trace = get_raw_trace(state)
        assert trace is not None
        assert long_marker in str(trace.get("traceback") or "")
        return {
            "current_agent": "Supervisor",
            "active_intent": "replan",
        }

    graph = compile_subgraph_retry_graph(
        subgraph,
        supervisor,
        checkpointer=MemorySaver(),
    )
    seed: ReactGraphState = {
        "session_id": "subretry-2",
        "current_agent": "SubGraph",
        "active_intent": "will-fail",
        "subgraph_retry_count": 0,
        "max_subgraph_retries": 2,
        "fatal_block": False,
        "raw_state_buffer": {},
        "patch_ledger_path": ledger,
    }
    out = graph.invoke(seed, config={"configurable": {"thread_id": "t-escalate"}})

    # initial attempt + 2 local retries, then supervisor
    assert visits.count(SUBGRAPH_NODE) == 3
    assert SUPERVISOR_NODE in visits
    assert visits[-1] == SUPERVISOR_NODE
    assert int(out.get("subgraph_retry_count") or 0) == 0

    trace = get_raw_trace(out)
    assert trace is not None
    assert long_marker in trace["traceback"]
    assert "KeyError" in trace["traceback"]
    assert "Traceback (most recent call last)" in trace["traceback"]
    # No truncation of the long marker payload.
    assert long_marker == "STACK_MARKER_" + ("Z" * 800)
    body = Path(ledger).read_text(encoding="utf-8")
    assert "[PENDING]" in body
    assert long_marker in body


def test_resolve_and_route_helpers_agree_on_retry_budget() -> None:
    """Unit-level: resolve patch + route name stay aligned for N=2."""
    err = ValueError("soft-fail")
    state: dict[str, Any] = {
        **store_raw_trace({}, err, {}),
        "subgraph_retry_count": 0,
        "max_subgraph_retries": 2,
        "fatal_block": False,
    }
    nxt, patch = resolve_subgraph_execution(state)
    assert nxt == SUBGRAPH_NODE
    assert patch["subgraph_retry_count"] == 1
    assert route_subgraph_execution(state) == "bump_subgraph_retry"

    state["subgraph_retry_count"] = 2
    nxt2, patch2 = resolve_subgraph_execution(state)
    assert nxt2 == SUPERVISOR_NODE
    assert patch2["subgraph_retry_count"] == 0
    assert route_subgraph_execution(state) == "escalate_subgraph"

    # Fatal skips local retry even when budget remains.
    fatal_state: dict[str, Any] = {
        **store_raw_trace({}, PermissionError("denied"), {}),
        "subgraph_retry_count": 0,
        "max_subgraph_retries": 2,
        "fatal_block": True,
    }
    nxt3, patch3 = resolve_subgraph_execution(fatal_state)
    assert nxt3 == SUPERVISOR_NODE
    assert patch3["subgraph_retry_count"] == 0
