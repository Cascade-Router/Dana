"""Meta-Broker intent override must beat Tool Forge / ReAct swallowing."""

from __future__ import annotations

from dana.tools.broker import (
    IntentBroker,
    extract_meta_broker_prompt,
    is_meta_broker_intent,
    reload_broker_registry,
)


def test_meta_broker_keyword_detection() -> None:
    assert is_meta_broker_intent(
        "Use the Meta-Broker to build a rate limiter with TDD epics"
    )
    assert is_meta_broker_intent("/broker refactor auth across 12 files")
    assert is_meta_broker_intent("/broker")
    assert not is_meta_broker_intent("build a custom tool that reverses a string")
    assert extract_meta_broker_prompt("/broker Epic 1: write tests") == (
        "Epic 1: write tests"
    )


def test_parse_utterance_routes_meta_broker_before_forge() -> None:
    reload_broker_registry()
    broker = IntentBroker()
    call = broker.parse_utterance(
        "Use the Meta-Broker to build a new rate-limiting utility for Cascade Router. "
        "Epic 1: Write a pytest suite. Epic 2: Implement rate_limiter.py."
    )
    assert call is not None
    assert call.tool_id == "meta_broker"
    assert "rate-limiting" in str(call.arguments.get("prompt") or "")
    assert float(call.confidence) >= 0.99

    slash = broker.parse_utterance("/broker multi-file TDD refactor of cascade router")
    assert slash is not None
    assert slash.tool_id == "meta_broker"
    assert "multi-file" in str(slash.arguments.get("prompt") or "")


def test_requires_tool_graph_meta_broker() -> None:
    from dana.agentic import requires_tool_graph

    assert requires_tool_graph("Use the Meta-Broker for TDD epics") is True
    assert requires_tool_graph("/broker do the thing") is True


def test_spatial_prompt_lists_meta_broker() -> None:
    from dana.prompts.spatial_synthesis import build_agent_system_prompt

    prompt = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )
    assert "meta_broker" in prompt
    assert "Meta-Broker routing" in prompt or "meta_broker(prompt=" in prompt
