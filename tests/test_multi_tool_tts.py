"""Multi-tool REPL suite binding + TTS utterance ceiling checks."""

from __future__ import annotations

from dana.core.constants import TTS_UTTERANCE_MAX_SECONDS
from dana.audio.tts_manager import chunk_text_for_tts
from dana.tools.broker import get_broker, merge_bound_tool_ids, repl_suite_tool_ids


MULTI_STEP = (
    "Use python_repl to write a script, execute it, then file_editor to read the result."
)


def test_repl_suite_detects_multi_step_chain() -> None:
    ids = repl_suite_tool_ids(MULTI_STEP)
    assert "python_repl" in ids
    assert "file_editor" in ids
    assert "shell_execute" in ids


def test_merge_binds_full_repl_suite_not_just_file_editor() -> None:
    merged = merge_bound_tool_ids(
        user_text=MULTI_STEP,
        forced_tool_id="file_editor",
        mode="developer",
    )
    assert "python_repl" in merged
    assert "file_editor" in merged
    assert "shell_execute" in merged


def test_merge_binds_execute_powershell_for_network_adapter_prompt() -> None:
    """System/OS PowerShell asks must bind execute_powershell (not vision top-K)."""
    q = "Use PowerShell to check my current active network adapters and IP address"
    merged = merge_bound_tool_ids(
        user_text=q,
        forced_tool_id=None,
        mode="chat",
    )
    assert "execute_powershell" in merged
    # Terminal/cmd execution cues also seed the PowerShell actuator.
    assert "execute_powershell" in merge_bound_tool_ids(
        user_text="Use the terminal to list running processes",
        forced_tool_id=None,
        mode="chat",
    )
    call = get_broker().parse_utterance(q)
    assert call is not None
    assert call.tool_id == "execute_powershell"


def test_broker_foresight_prioritizes_python_repl_on_chain() -> None:
    call = get_broker().parse_utterance(MULTI_STEP)
    assert call is not None
    assert call.tool_id == "python_repl"


def test_tts_utterance_ceiling_allows_long_speech() -> None:
    assert TTS_UTTERANCE_MAX_SECONDS >= 60.0


def test_chunk_text_for_tts_splits_long_ocr() -> None:
    long = ". ".join([f"Sentence number {i} about the screen text" for i in range(20)])
    chunks = chunk_text_for_tts(long, max_chars=120)
    assert len(chunks) >= 3
    assert all(len(c) <= 160 for c in chunks)
