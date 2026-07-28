"""Headless Spatial Awareness tests — pronoun grounding + explicit hand queries.

Mocks Visual Context injection (no YOLO / mic / Ollama) and asserts the ReAct
loop resolves deictics from the natural-language vision line on the *last*
user message (recency bias), not the system prompt.
"""

from __future__ import annotations

from test_support_react import patch_scripted_llm
from dana.agentic import REACT_MAX_ITERS, run_react_loop, wrap_user_query_for_react
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.prompts.spatial_synthesis import (
    SPATIAL_AWARENESS,
    build_agent_system_prompt,
    format_recency_context_block,
    format_vision_context,
)


def _noop_execute(tc: ToolCall) -> str:
    return f"ERROR: unexpected tool {tc.tool_id} — spatial tests are FINAL-only"


def test_format_vision_context_natural_sentences() -> None:
    assert (
        format_vision_context(["laptop (center)"])
        == "Visual Context: The user is currently in front of a laptop."
    )
    assert (
        format_vision_context(["book (hand)"])
        == "Visual Context: The user is holding a book."
    )
    assert (
        format_vision_context(["apple (hand)"])
        == "Visual Context: The user is holding an apple."
    )
    assert format_vision_context(["none detected"]) == ""
    assert format_vision_context([]) == ""
    assert format_vision_context("none detected") == ""


def test_recency_context_block_format() -> None:
    vision = format_vision_context(["book (hand)"])
    block = format_recency_context_block(vision_line=vision, prior_turn_count=2)
    assert "<visual_context>The user is holding a book.</visual_context>" in block
    assert "<memory>" in block and "2 prior turn(s)" in block
    assert "[SYSTEM:" not in block

    empty = format_recency_context_block(vision_line="", prior_turn_count=0)
    assert empty == ""


def test_system_prompt_soft_unlock_no_gag_order() -> None:
    prompt = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=laptop@center|scene=[laptop@center(a=0.3,d=0.01)]|intent=",
        labels_csv="laptop (center)",
        profile_summary="{}",
        reply_lang="en",
    )
    assert "CRITICAL INSTRUCTION: Do NOT mention" not in prompt
    assert "Spatial Awareness" in prompt
    assert SPATIAL_AWARENESS.splitlines()[1] in prompt or "real-time visual context" in prompt
    # Visual Context belongs on the last user turn — not early in system prompt.
    assert "Visual Context:" not in prompt
    assert "Visible labels:" not in prompt

    empty = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none|scene=[none]|intent=",
        labels_csv="none detected",
        profile_summary="{}",
        reply_lang="en",
    )
    assert "Visual Context:" not in empty
    assert "CRITICAL INSTRUCTION" not in empty


def test_implicit_context_resolves_this_to_laptop(monkeypatch) -> None:
    """'Should I close this?' + laptop Visual Context → reply mentions laptop/computer."""
    vision = "Visual Context: The user is currently in front of a laptop."
    assert format_vision_context(["laptop (center)"]) == vision
    tag = "<visual_context>The user is currently in front of a laptop.</visual_context>"
    assert format_recency_context_block(vision_line=vision) == tag

    system = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=laptop@center|scene=[laptop@center(a=0.4,d=0.01)]|intent=",
        labels_csv="laptop (center)",
        profile_summary="{}",
        reply_lang="en",
    )
    assert vision not in system

    utterance = "Should I close this?"
    saw_vision = {"v": False, "on_last_user": False}

    def ask_fn(messages: list[dict[str, str]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if tag in last_user or "in front of a laptop" in last_user:
            saw_vision["v"] = True
            saw_vision["on_last_user"] = tag in last_user
            return (
                "FINAL: If you're done, go ahead and close the laptop — "
                "or lock your computer first if you'll be back soon."
            )
        return "FINAL: Close what, exactly?"

    patch_scripted_llm(monkeypatch, ask_fn)
    result = run_react_loop(
        user_text=utterance,
        system_prompt=system,
        execute_fn=_noop_execute,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker(),
        enable_reflection=False,
        visual_context=vision,
    )

    assert saw_vision["v"], "Visual Context must appear in ReAct messages"
    assert saw_vision["on_last_user"], "Visual Context must be on the last user message"
    reply = (result.final_text or "").lower()
    assert "laptop" in reply or "computer" in reply, f"pronoun not grounded: {result.final_text!r}"
    assert "yolo" not in reply
    assert "bounding" not in reply
    assert "[" not in reply or "laptop (center)" not in reply
    print(f"[PASS] Implicit Context: {result.final_text}")


def test_explicit_query_identifies_apple_in_hand(monkeypatch) -> None:
    """'what I have in my hand?' + apple Visual Context → apple, no YOLO syntax."""
    vision = "Visual Context: The user is holding an apple."
    assert format_vision_context(["apple (hand)"]) == vision
    tag = "<visual_context>The user is holding an apple.</visual_context>"

    system = build_agent_system_prompt(
        spatial_block="vis=camera|ui=idle|dom=apple@hand|scene=[apple@hand(a=0.2,d=0.05)]|intent=",
        labels_csv="apple (hand)",
        profile_summary="{}",
        reply_lang="en",
    )
    assert vision not in system

    utterance = "Can you tell me what I have in my hand?"
    poison_markers = (
        "yolo",
        "bounding box",
        "bbox",
        "confidence",
        "dom=",
        "apple@hand",
        "scene=[",
        "SpatialIR",
    )

    def ask_fn(messages: list[dict[str, str]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        assert tag in last_user or "holding an apple" in last_user
        assert vision not in (messages[0].get("content") or "")
        return "FINAL: You're holding an apple."

    patch_scripted_llm(monkeypatch, ask_fn)
    result = run_react_loop(
        user_text=utterance,
        system_prompt=system,
        execute_fn=_noop_execute,
        max_iters=REACT_MAX_ITERS,
        broker=IntentBroker(),
        enable_reflection=False,
        visual_context=vision,
    )

    reply = result.final_text or ""
    reply_l = reply.lower()
    assert "apple" in reply_l, f"expected apple identification, got: {reply!r}"
    for marker in poison_markers:
        assert marker.lower() not in reply_l, f"raw vision syntax leaked: {marker!r} in {reply!r}"
    print(f"[PASS] Explicit Query: {result.final_text}")


def test_wrap_appends_suffix_after_utterance() -> None:
    suffix = format_recency_context_block(
        vision_line="Visual Context: The user is holding a phone.",
        prior_turn_count=1,
    )
    wrapped = wrap_user_query_for_react("What is this?", "en", context_suffix=suffix)
    q_idx = wrapped.index("USER'S ACTUAL QUESTION: What is this?")
    v_idx = wrapped.index("<visual_context>The user is holding a phone.</visual_context>")
    m_idx = wrapped.index("<memory>")
    assert q_idx < v_idx < m_idx
    assert "[SYSTEM:" not in wrapped


if __name__ == "__main__":
    test_format_vision_context_natural_sentences()
    test_recency_context_block_format()
    test_system_prompt_soft_unlock_no_gag_order()
    test_implicit_context_resolves_this_to_laptop()
    test_explicit_query_identifies_apple_in_hand()
    test_wrap_appends_suffix_after_utterance()
    print("OK")
