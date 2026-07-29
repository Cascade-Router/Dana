"""Sub-graph / fail-closed escalations sync ``[PENDING]`` to the patch ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from dana.graph.buffer import store_raw_trace
from dana.graph.nodes.critic import fail_closed_node
from dana.graph.subgraph_router import (
    SUBGRAPH_NODE,
    SUPERVISOR_NODE,
    apply_subgraph_failure,
    compile_subgraph_retry_graph,
    escalate_subgraph,
)
from dana.schema import ReactGraphState


def test_fatal_os_error_fail_closed_writes_pending(tmp_path: Path) -> None:
    ledger = tmp_path / "patch_ledger.md"
    err = "OSError: [Errno 13] Permission denied: 'C:\\\\locked\\\\file'"
    out = fail_closed_node(
        {
            "fatal_block": True,
            "execution_error": err,
            "session_id": "os-fatal",
            "patch_ledger_path": str(ledger),
            **store_raw_trace({}, OSError(err), {"phase": "python_repl"}),
        }
    )
    assert out["halt"] is True
    assert out["fatal_block"] is True
    text = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in text
    assert "OSError" in text
    assert "**Task ID:**" in text
    assert "**Timestamp:**" in text
    assert "**Recommended Fix:**" in text


def test_fatal_subgraph_escalate_writes_pending(tmp_path: Path) -> None:
    """Fatal block skips local retries and flushes ledger on escalate_subgraph."""
    ledger = tmp_path / "sg_ledger.md"
    visits: list[str] = []

    def subgraph(state: ReactGraphState) -> dict[str, Any]:
        visits.append(SUBGRAPH_NODE)
        return {
            **apply_subgraph_failure(
                state,
                PermissionError("denied: /etc/shadow"),
                {"node": SUBGRAPH_NODE},
            ),
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
        "session_id": "sg-fatal",
        "current_agent": "SubGraph",
        "subgraph_retry_count": 0,
        "max_subgraph_retries": 2,
        "fatal_block": False,
        "raw_state_buffer": {},
        "patch_ledger_path": str(ledger),
    }
    out = graph.invoke(seed, config={"configurable": {"thread_id": "t-fatal-esc"}})

    assert visits.count(SUBGRAPH_NODE) == 1
    assert SUPERVISOR_NODE in visits
    assert out.get("fatal_block") is True
    body = ledger.read_text(encoding="utf-8")
    assert "[PENDING]" in body
    assert "PermissionError" in body or "denied" in body


def test_escalate_subgraph_node_direct_flush(tmp_path: Path) -> None:
    ledger = tmp_path / "direct.md"
    state = {
        **store_raw_trace({}, FileNotFoundError("missing.bin"), {}),
        "fatal_block": True,
        "session_id": "direct-esc",
        "subgraph_retry_count": 2,
        "patch_ledger_path": str(ledger),
    }
    patch = escalate_subgraph(state)
    assert patch["subgraph_retry_count"] == 0
    assert "[PENDING]" in ledger.read_text(encoding="utf-8")
