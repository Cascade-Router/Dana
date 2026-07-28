"""Regression: clipboard must never leak into ReAct on a research query.

Proves the hard gate in ``dana.agentic.run_react_loop`` blocks
``read_clipboard_context`` unless the user explicitly asks about the clipboard,
even when a poisoned OS clipboard (or a hallucinated TOOL call) is present.
"""

from __future__ import annotations

from unittest.mock import patch

from test_support_react import patch_scripted_llm
from dana.agentic import REACT_MAX_ITERS, run_react_loop
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.prompts.spatial_synthesis import build_agent_system_prompt

# Irrelevant raw terminal dump — large enough to poison an LLM context window.
CLIPBOARD_POISON = (
    "PS C:\\Users\\Example> Traceback (most recent call last):\n"
    "  File \"dana.core_agent.py\", line 9999, in <module>\n"
    "RuntimeError: simulated kernel panic dump\n"
    "FATAL: orphaned socket 0xDEADBEEF\n"
) * 40  # >> 1000 characters

QUERY = "Research the latest updates on Python."
PYTHON_SUMMARY = (
    "Python 3.13 continues stabilizing free-threading and the JIT; "
    "pip and typing tools saw incremental improvements this cycle."
)


def _system_prompt() -> str:
    return build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )


def test_clipboard_trap_does_not_leak_into_research(monkeypatch) -> None:
    assert len(CLIPBOARD_POISON) >= 1000, "poison payload must be >= 1000 chars"

    broker = IntentBroker()
    executed: list[str] = []

    # Clipboard tool is intentionally unbound — adversarial native calls are dropped.
    # Research must still execute web_search without ever touching the OS clipboard.
    responses = [
        "TOOL: web_search(query=latest Python language updates)",
        f"FINAL: {PYTHON_SUMMARY}",
    ]
    patch_scripted_llm(monkeypatch, responses)

    def execute_fn(tc: ToolCall) -> str:
        executed.append(tc.tool_id)
        if tc.tool_id == "read_clipboard_context":
            from dana.os_automation import read_clipboard_context

            result = read_clipboard_context()
            text = result.get("text") or ""
            return f"OK: clipboard chars={len(text)} text={text!r}"
        if tc.tool_id == "web_search":
            return (
                "Observation: Python 3.13 free-threading and experimental JIT "
                "updates; typing / pip ecosystem incremental releases. "
                "(source: python.org)"
            )
        return f"ERROR: unexpected tool {tc.tool_id}"

    poison_clipboard = {
        "ok": True,
        "text": CLIPBOARD_POISON,
        "empty": False,
        "truncated": False,
        "chars": len(CLIPBOARD_POISON),
    }

    with patch(
        "dana.os_automation.read_clipboard_context",
        return_value=poison_clipboard,
    ) as mock_clip:
        result = run_react_loop(
            user_text=QUERY,
            system_prompt=_system_prompt(),
            execute_fn=execute_fn,
            max_iters=REACT_MAX_ITERS,
            broker=broker,
            enable_reflection=False,
        )

    # Hard gate: OS clipboard must never be read for a research query.
    mock_clip.assert_not_called()
    assert "read_clipboard_context" not in executed, (
        "execute_fn must never run read_clipboard_context on a research query"
    )
    assert executed == ["web_search"], f"expected only web_search executed, got {executed}"

    web_steps = [t for t in result.tool_trace if t.get("tool") == "web_search"]
    assert web_steps, "expected web_search in ReAct trace"

    final = result.final_text or ""
    final_l = final.lower()
    assert "python" in final_l, f"expected a Python summary, got: {final!r}"
    assert "DEADBEEF" not in final
    assert "kernel panic" not in final_l
    assert "orphan" not in final_l
    assert CLIPBOARD_POISON[:80] not in final

    print("[PASS] Clipboard Trap: only web_search executed; poison did not leak")
    print(f"       executed={executed}")
    print(f"       final={final[:120]}")


def test_generic_reflection_phrases_do_not_force_file_read() -> None:
    broker = IntentBroker()
    assert broker.parse_utterance("relationship reflection") is None
    assert broker.parse_utterance("things are complicated today") is None

    explicit = broker.parse_utterance("read the file docs/project_omega_status.txt")
    assert explicit is not None
    assert explicit.tool_id in {"read_local_file", "file_editor"}
    print("[PASS] generic reflection phrases stay conversational")


if __name__ == "__main__":
    test_clipboard_trap_does_not_leak_into_research()
    test_generic_reflection_phrases_do_not_force_file_read()
    print("OK")
