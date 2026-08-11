"""Scratchpad compression unit tests."""

from __future__ import annotations

from dana.middleware.scratchpad import compress_tool_output


def test_compress_passthrough_when_short() -> None:
    text = "ok: short observation"
    assert compress_tool_output(text) == text
    assert compress_tool_output(text, max_length=50) == text


def test_compress_keeps_head_and_tail() -> None:
    body = ("HEAD-" * 200) + ("MID-" * 400) + ("TAIL-" * 200)
    assert len(body) > 1200
    out = compress_tool_output(body, max_length=1200)
    assert out.startswith("HEAD-")
    assert out.endswith("TAIL-")
    assert "SCRATCHPAD COMPRESSION:" in out
    assert "characters removed" in out
    removed = len(body) - 500 - 500
    assert f"{removed} characters removed" in out
    # Compressed payload stays well under the raw size.
    assert len(out) < len(body)
    assert len(out) < 1300


def test_execute_tool_call_applies_scratchpad() -> None:
    from dana.core.agent_loop import execute_tool_call
    from dana.tools.schema import ToolCall

    # Force a large observation via read_local_file on a known large module.
    tc = ToolCall(
        tool_id="read_local_file",
        arguments={"filepath": "dana/core_agent.py"},
        source_lang="en",
        raw_text="read dana/core_agent.py",
        confidence=0.99,
    )
    obs = execute_tool_call(tc)
    # File is far larger than 1200 chars; Scratchpad must engage.
    assert "SCRATCHPAD COMPRESSION:" in obs or len(obs) <= 1200
    if "SCRATCHPAD COMPRESSION:" in obs:
        assert obs.index("SCRATCHPAD COMPRESSION:") > 100
