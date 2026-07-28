"""Deep-research swarm unit tests — WebSearchTool, Scratchpad, graph wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dana.swarm.scratchpad import (
    format_findings_for_writer,
    init_scratchpad,
    open_session,
    read_findings,
    write_finding,
)
from dana.swarm.swarm_main import (
    _parse_plan,
    build_swarm_graph,
    planner_agent,
    search_agent,
)
from dana.swarm.web_search_tool import (
    WebSearchTool,
    build_web_search_tool,
    search_once,
)


def test_search_once_rejects_empty_query() -> None:
    summary = search_once("")
    assert summary.ok is False
    assert "empty" in (summary.error or "").lower()
    summary2 = search_once("   ")
    assert summary2.ok is False
    print("[PASS] search_once rejects empty query")


def test_web_search_tool_refuses_empty_invoke() -> None:
    tool = build_web_search_tool()
    assert isinstance(tool, WebSearchTool)
    out = tool.invoke({"query": ""})
    assert "ERROR" in out
    assert "empty" in out.lower()
    print("[PASS] WebSearchTool refuses empty query")


def test_web_search_tool_bindable_schema() -> None:
    tool = WebSearchTool()
    assert tool.name == "web_search"
    # LangChain bind_tools expects a BaseTool with args_schema.
    assert tool.args_schema is not None
    fields = getattr(tool.args_schema, "model_fields", None) or {}
    assert "query" in fields
    print("[PASS] WebSearchTool has bindable query schema")


def test_scratchpad_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "research_scratchpad.db"
    init_scratchpad(db)
    sid = open_session("quantum computing", db_path=db)
    row_id = write_finding(
        sid,
        objective="what is quantum computing",
        query_used="quantum computing basics",
        findings_text="OK: web_search hits=1\n1. QC — qubits",
        db_path=db,
    )
    assert row_id >= 1
    rows = read_findings(sid, db_path=db)
    assert len(rows) == 1
    assert "qubits" in rows[0]["findings_text"]
    blob = format_findings_for_writer(rows)
    assert "Finding 1" in blob
    print("[PASS] scratchpad roundtrip")


def test_parse_plan_json() -> None:
    assert _parse_plan('["a", "b", "c"]') == ["a", "b", "c"]
    assert _parse_plan('```json\n["x"]\n```') == ["x"]
    print("[PASS] plan JSON parse")


def test_build_swarm_graph_has_planner_search_writer() -> None:
    app = build_swarm_graph()
    assert app is not None
    # Compiled graph exposes nodes via get_graph / nodes depending on version.
    g = app.get_graph()
    node_ids = set(g.nodes.keys())
    assert "planner_agent" in node_ids
    assert "search_agent" in node_ids
    assert "writer_agent" in node_ids
    print("[PASS] swarm graph nodes: Planner → Search → Writer")


def test_search_agent_binds_web_search_tool(monkeypatch, tmp_path: Path) -> None:
    """Search Agent must call llm.bind_tools([WebSearchTool]) before searching."""
    db = tmp_path / "sp.db"
    sid = open_session("topic", db_path=db)

    bound: dict[str, object] = {}

    class _FakeLLM:
        def bind_tools(self, tools):
            bound["tools"] = list(tools)
            return self

        def invoke(self, _messages):
            # No tool_calls → fallback search_once path.
            msg = MagicMock()
            msg.tool_calls = []
            msg.content = "done"
            return msg

    monkeypatch.setattr(
        "dana.swarm.swarm_main._build_llm",
        lambda **_k: _FakeLLM(),
    )
    monkeypatch.setattr(
        "dana.swarm.swarm_main.search_once",
        lambda q, **_k: MagicMock(
            ok=True,
            query=q,
            findings_text=f"OK: {q}",
            error=None,
        ),
    )
    monkeypatch.setattr(
        "dana.swarm.swarm_main.write_finding",
        lambda *a, **k: 1,
    )
    monkeypatch.setattr(
        "dana.swarm.swarm_main.read_findings",
        lambda *_a, **_k: [],
    )

    out = search_agent(
        {
            "query": "topic",
            "plan": ["facet one"],
            "session_id": sid,
            "findings": [],
            "report": "",
        }
    )
    assert "tools" in bound
    names = [getattr(t, "name", "") for t in bound["tools"]]  # type: ignore[index]
    assert "web_search" in names
    assert out["findings"]
    print("[PASS] Search Agent binds WebSearchTool")


def test_planner_agent_parses_objectives(monkeypatch) -> None:
    class _FakeLLM:
        def invoke(self, _messages):
            return MagicMock(content='["obj1", "obj2"]')

    monkeypatch.setattr(
        "dana.swarm.swarm_main._build_llm",
        lambda **_k: _FakeLLM(),
    )
    monkeypatch.setattr(
        "dana.swarm.swarm_main.open_session",
        lambda _q: "sess123",
    )
    out = planner_agent(
        {
            "query": "deep topic",
            "plan": [],
            "session_id": "",
            "findings": [],
            "report": "",
        }
    )
    assert out["plan"] == ["obj1", "obj2"]
    assert out["session_id"] == "sess123"
    print("[PASS] PlannerAgent decomposition")


def test_broker_deep_research_routes_to_swarm() -> None:
    from dana.tools.broker import IntentBroker

    broker = IntentBroker()
    call = broker.parse_utterance("Deep research on solid-state batteries")
    assert call is not None
    assert call.tool_id == "dispatch_research_swarm"
    assert str(call.arguments.get("query") or "").strip()
    print("[PASS] broker deep research → dispatch_research_swarm with query")


def test_broker_quick_research_still_web_search() -> None:
    from dana.tools.broker import IntentBroker

    broker = IntentBroker()
    call = broker.parse_utterance("look up the capital of France")
    assert call is not None
    assert call.tool_id == "web_search"
    assert call.arguments.get("query")
    print("[PASS] broker quick lookup → web_search")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
