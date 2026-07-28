"""Episodic memory store + hydrate/consolidate graph node tests (temp SQLite)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from dana.agentic_react_graph import compile_donna_react_graph
from dana.graph.nodes.memory import (
    heuristic_extract_facts,
    make_consolidate_memory_node,
    make_hydrate_memory_node,
)
from dana.memory.store import EpisodicMemoryStore
from dana.schema import ReactGraphState


@pytest.fixture
def store(tmp_path: Path) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(tmp_path / "episodic_test.db")


def test_add_fact_upsert_and_preferences(store: EpisodicMemoryStore) -> None:
    """Run 1: store prefer_dark_mode; upsert overwrites value."""
    row = store.add_fact("user_preference", "prefer_dark_mode", True)
    assert row["key"] == "prefer_dark_mode"
    assert row["category"] == "user_preference"
    prefs = store.get_all_preferences()
    assert prefs["prefer_dark_mode"] is True

    store.add_fact("user_preference", "prefer_dark_mode", False, confidence_score=0.9)
    prefs2 = store.get_all_preferences()
    assert prefs2["prefer_dark_mode"] is False
    hits = store.search_facts("dark mode preference")
    assert any(h["key"] == "prefer_dark_mode" for h in hits)


def test_hydrate_memory_across_conversations(store: EpisodicMemoryStore) -> None:
    """Run 2: new conversation; hydrate injects prefer_dark_mode into memory_context."""
    store.add_fact("user_preference", "prefer_dark_mode", True)
    hydrate = make_hydrate_memory_node(store)
    state: ReactGraphState = {
        "messages": [HumanMessage(content="what theme should we use?")],
        "session_id": "ephemeral-2",
        "active_intent": "what theme should we use?",
    }
    out = hydrate(state)
    ctx = out.get("memory_context") or {}
    assert ctx.get("prefer_dark_mode") is True
    assert (ctx.get("preferences") or {}).get("prefer_dark_mode") is True


def test_consolidate_heuristic_writes_preference(store: EpisodicMemoryStore) -> None:
    """Offline consolidate extracts dark-mode preference without a cloud LLM."""
    facts = heuristic_extract_facts("I prefer dark mode for the UI", "OK, noted.")
    assert any(f["key"] == "prefer_dark_mode" and f["value"] is True for f in facts)

    consolidate = make_consolidate_memory_node(store)
    state: ReactGraphState = {
        "messages": [
            HumanMessage(content="Please prefer dark mode going forward"),
            AIMessage(content="FINAL: Got it — I'll use dark mode."),
        ],
        "halt": True,
        "final_raw": "FINAL: Got it — I'll use dark mode.",
        "session_id": "consol-1",
    }
    out = consolidate(state)
    assert store.get_all_preferences().get("prefer_dark_mode") is True
    ctx = out.get("memory_context") or {}
    assert "prefer_dark_mode" in (ctx.get("consolidated_keys") or [])


def test_graph_wires_hydrate_then_consolidate(store: EpisodicMemoryStore) -> None:
    """Corridor: START → hydrate → … → consolidate → END on successful halt."""
    store.add_fact("user_preference", "prefer_dark_mode", True)
    path: list[str] = []
    hydrate = make_hydrate_memory_node(store)
    consolidate = make_consolidate_memory_node(store)

    def _hydrate(state: ReactGraphState) -> dict[str, Any]:
        path.append("hydrate_memory")
        return hydrate(state)

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        # Prove hydrate ran first and injected preference.
        assert (state.get("memory_context") or {}).get("prefer_dark_mode") is True
        return {
            "execution_plan": {"required_tools": [], "status": "planned"},
            "current_agent": "Planner",
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [AIMessage(content="FINAL: theme is dark")],
            "halt": True,
            "final_raw": "FINAL: theme is dark",
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        return {"halt": True, "final_raw": "should_not_run"}

    def _consolidate(state: ReactGraphState) -> dict[str, Any]:
        path.append("consolidate_memory")
        return consolidate(state)

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        hydrate_memory_node_fn=_hydrate,
        consolidate_memory_node_fn=_consolidate,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "episodic-wire"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="use my theme preference")],
                "halt": False,
                "session_id": "episodic-wire",
                "active_intent": "use my theme preference",
            },
            cfg,
            stream_mode="values",
        )
    )
    assert path[0] == "hydrate_memory"
    assert "planner" in path
    assert "agent" in path
    assert path[-1] == "consolidate_memory"
    assert "tools" not in path
