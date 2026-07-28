#!/usr/bin/env python3
"""Unit checks for architect_new_tool broker mapping + bug tracker + titan repair graph."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dana.bug_tracker import load_bug_tracker, open_bugs
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall


def test_broker_routes_build_a_tool_to_architect_with_goal() -> None:
    broker = IntentBroker()
    raw = "Donna, build a tool that reverses a string."
    call = broker.parse_utterance(raw)
    assert call is not None, "broker returned None"
    assert call.tool_id == "architect_new_tool", call.tool_id
    goal = str(call.arguments.get("goal") or "")
    assert goal.strip(), f"empty goal: {call.arguments}"
    assert "build a tool" in goal.lower()
    assert call.tool_id != "read_vault_memory"
    print("[PASS] broker build-a-tool -> architect_new_tool(goal=utterance)")


def test_broker_never_vault_on_create_tool() -> None:
    broker = IntentBroker()
    for phrase in (
        "create a tool that counts words",
        "code a script to uppercase text",
        "forge a tool for hashing",
    ):
        call = broker.parse_utterance(phrase)
        assert call is not None
        assert call.tool_id == "architect_new_tool", (phrase, call.tool_id)
        assert str(call.arguments.get("goal") or "").strip()
    print("[PASS] create/code/forge tool phrases -> Tool Forge (not vault)")


def test_architect_empty_args_pulls_raw_text() -> None:
    from dana.core_agent import execute_tool_call

    call = ToolCall(
        tool_id="architect_new_tool",
        arguments={},
        raw_text="build a tool that adds two numbers",
    )
    with patch(
        "dana.settings.is_dynamic_tool_synthesis_enabled", lambda: True
    ), patch(
        "dana.swarm.tool_forge_graph.route_tool_not_found",
        lambda q, missing_tool="", model="llama3.2": {
            "status": "loaded",
            "loaded_tool": "add_two_numbers",
            "feedback": "ok",
        },
    ), patch(
        "dana.tools.broker.reload_broker_registry", lambda *a, **k: None
    ):
        obs = execute_tool_call(call)
    assert "OK:" in obs and "add_two_numbers" in obs, obs
    print("[PASS] architect_new_tool empty args -> goal from raw_text -> Tool Forge")


def test_bug_tracker_append_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bug_tracker.json"
        from dana.bug_tracker import log_bug_to_tracker, list_todo_basket

        entry = log_bug_to_tracker(
            "ERROR: missing goal",
            context="user asked to build a tool",
            status="PENDING",
            path=path,
        )
        assert entry["status"] == "PENDING"
        assert "timestamp" in entry and "error" in entry and "context" in entry
        bugs = load_bug_tracker(path)
        assert len(bugs) == 1
        assert bugs[0]["id"] == entry["id"]
        assert open_bugs(path)
        summary = list_todo_basket(path=path)
        assert "PENDING" in summary and "missing goal" in summary
    print("[PASS] bug_tracker PENDING schema + list_todo_basket")


def test_list_todo_basket_broker_route() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("Donna, list the todo basket.")
    assert call is not None
    assert call.tool_id == "list_todo_basket", call.tool_id
    print("[PASS] broker list_todo_basket verbal command")


def test_titan_repair_graph_writes_pending_patch() -> None:
    from dana.swarm import titan_repair as tr

    bug = {
        "id": "bug_test_1",
        "user_query": "build a tool",
        "error": "ValueError: boom",
        "traceback": (
            'Traceback (most recent call last):\n'
            '  File "C:/Users/Example/Project/dana/agentic.py", line 1, in <module>\n'
            "    raise ValueError('boom')\n"
            "ValueError: boom"
        ),
        "status": "PENDING",
    }
    safe_code = (
        "from langchain_core.tools import tool\n"
        "\n\n"
        "@tool\n"
        "def patched_helper(text: str) -> str:\n"
        '    """Safe stub patch."""\n'
        "    return text or ''\n"
    )

    class _LLM:
        def invoke(self, messages):
            blob = ""
            for m in messages:
                blob += str(m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            if "security auditor" in blob.lower() or "zero-trust" in blob.lower():
                return MagicMock(
                    content=json.dumps(
                        {
                            "status": "APPROVED",
                            "threat_assessment": "safe stub",
                            "violations": [],
                            "required_remediation": "",
                        }
                    )
                )
            return MagicMock(
                content=json.dumps(
                    {
                        "target_file": "dana/agentic.py",
                        "summary": "guard empty text",
                        "code": safe_code,
                    }
                )
            )

    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending_patches"
        pending.mkdir()
        tracker = Path(tmp) / "bug_tracker.json"
        tracker.write_text(json.dumps([bug]), encoding="utf-8")
        with patch.object(tr, "PENDING_PATCHES_DIR", pending), patch(
            "dana.swarm.tool_forge_graph._chat_ollama", lambda **k: _LLM()
        ), patch(
            "dana.swarm.titan_repair._chat_ollama", lambda **k: _LLM()
        ), patch(
            "dana.swarm.titan_repair.open_bugs", lambda path=None: [bug]
        ), patch(
            "dana.swarm.titan_repair.mark_bug_status", lambda *a, **k: True
        ):
            summary = tr.run_titan_repair(query="fix bugs", model="llama3.2")
        files = list(pending.glob("*.json"))
        assert files, f"no pending patch written; summary={summary}"
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["status"] == "pending_human_review"
        assert "code" in payload
    assert "Would you like to review" in summary or "pending_patches" in summary.lower() or files
    print("[PASS] titan repair -> pending_patches (AST+security)")


def test_titan_repair_broker_route() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("Donna, run titan repair on the bug tracker.")
    assert call is not None
    assert call.tool_id == "dispatch_titan_repair", call.tool_id
    print("[PASS] broker titan repair verbal command")


if __name__ == "__main__":
    test_broker_routes_build_a_tool_to_architect_with_goal()
    test_broker_never_vault_on_create_tool()
    test_architect_empty_args_pulls_raw_text()
    test_bug_tracker_append_roundtrip()
    test_list_todo_basket_broker_route()
    test_titan_repair_graph_writes_pending_patch()
    test_titan_repair_broker_route()
    print("OK")
