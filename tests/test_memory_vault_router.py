"""Broker + tool-graph routing for Chroma memory vault phrases."""

from __future__ import annotations

from dana.agentic import requires_tool_graph
from dana.tools.broker import get_broker


def test_memory_hint_phrases_route_to_vault_tools() -> None:
    broker = get_broker()
    cases = [
        ("ingest this directory C:\\Users\\Amix\\Desktop\\DANA", "ingest_local_directory"),
        ("please ingest the codebase", "ingest_local_directory"),
        ("search the vault for purple widget", "search_vault"),
        ("look in vault for assembly notes", "search_vault"),
        ("query chroma for widget.py", "search_vault"),
        ("search memory for user preferences", "search_vault"),
        ("index this directory for the vault", "ingest_local_directory"),
    ]
    for phrase, tool_id in cases:
        call = broker.parse_utterance(phrase)
        assert call is not None, phrase
        assert call.tool_id == tool_id, (phrase, call.tool_id)


def test_memory_phrases_bypass_lightweight_chat() -> None:
    prompts = [
        "ingest this directory",
        "search the vault for purple widget",
        "look up chroma embeddings",
        "index the codebase vault",
        "search memory for preferences",
    ]
    for prompt in prompts:
        assert requires_tool_graph(prompt) is True, prompt


def test_clear_chat_memory_does_not_force_vault_tool() -> None:
    broker = get_broker()
    call = broker.parse_utterance("Please clear chat memory")
    assert call is None or call.tool_id not in {
        "search_vault",
        "ingest_local_directory",
    }
