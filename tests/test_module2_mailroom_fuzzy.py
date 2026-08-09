"""Module 2: RapidFuzz mailroom — ASR-tolerant deterministic routing."""

from __future__ import annotations

import json
from pathlib import Path

from dana.agentic import parse_mode_switch, set_dana_mode
from dana.cascade_router import (
    COMMAND_DICTIONARY,
    FUZZY_MATCH_THRESHOLD,
    decide_route,
    fuzzy_match_command,
    match_state_toggle,
)


def test_exact_vision_mode_still_works() -> None:
    hit = fuzzy_match_command("switch to vision mode")
    assert hit is not None
    assert hit.target == "vision"
    assert hit.method == "exact"
    assert hit.score == 100.0
    assert match_state_toggle("switch to vision mode") == "vision"


def test_asr_garble_vision_mounts_routes_to_vision() -> None:
    """Live-test failure mode: Whisper heard 'Vision mounts' for 'vision mode'."""
    hit = fuzzy_match_command("Vision mounts.")
    assert hit is not None
    assert hit.target == "vision"
    assert hit.score >= FUZZY_MATCH_THRESHOLD
    assert match_state_toggle("Vision mounts") == "vision"
    assert parse_mode_switch("Vision mounts.") == "vision"


def test_decide_route_short_circuits_vision_garble() -> None:
    set_dana_mode("chat")
    d = decide_route("vision mounts")
    assert "mailroom" in (d.reason or "").lower()
    assert d.backend == "moa"
    assert "vision" in (d.reason or "").lower()


def test_status_check_and_mute_commands() -> None:
    assert "status check" in COMMAND_DICTIONARY
    assert COMMAND_DICTIONARY["mute"] == "mute"
    st = fuzzy_match_command("status cheque")  # ASR garble
    assert st is not None
    assert st.target == "status_check"
    assert st.score >= FUZZY_MATCH_THRESHOLD
    d = decide_route("status check")
    assert "status_check" in (d.reason or "")
    d2 = decide_route("mute")
    assert "mute" in (d2.reason or "")


def test_fallthrough_below_threshold_emits_voice_asr(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    out = tmp_path / "dana_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    set_dana_mode("chat")
    # Clearly not a command — must fall through mailroom.
    d = decide_route("what is the weather in seattle today")
    assert "mailroom" not in (d.reason or "").lower() or "fall" in (d.reason or "").lower()
    # Fallthrough path logs [VOICE_ASR]
    assert out.is_file()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    tags = [json.loads(x).get("tag") for x in lines]
    assert "[VOICE_ASR]" in tags


def test_mailroom_hit_emits_router_telemetry(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    out = tmp_path / "dana_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    decide_route("switch to vision")
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").strip().splitlines()]
    router = [r for r in lines if r.get("tag") == "[ROUTER]"]
    assert router
    payload = router[-1].get("payload") or {}
    assert payload.get("matched_command")
    assert payload.get("confidence", 0) >= FUZZY_MATCH_THRESHOLD
    assert payload.get("target_node") == "vision"


def test_length_guard_skips_fuzzy_on_long_utterances() -> None:
    """Stage 3.1: >8 words must not fuzzy-hijack MoA / draft_cursor prompts."""
    from dana.cascade_router import MAILROOM_MAX_WORDS

    long_prompt = (
        "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket "
        "to implement a sliding-window garbage collector for our SQLite blackboard "
        "so it doesn't grow infinitely."
    )
    assert len(long_prompt.split()) > MAILROOM_MAX_WORDS
    assert fuzzy_match_command(long_prompt) is None
    set_dana_mode("chat")
    d = decide_route(long_prompt)
    assert "mailroom" not in (d.reason or "").lower()


def test_residual_after_compound_mode_switch() -> None:
    from dana.cascade_router import strip_mailroom_residual

    hit = fuzzy_match_command(
        "Switch back to chat mode. My favorite color is cobalt blue."
    )
    assert hit is not None
    assert hit.target == "chat"
    assert "cobalt" in (hit.residual or "").lower()
    residual = strip_mailroom_residual(
        "Switch back to chat mode. My favorite color is cobalt blue.",
        hit.command,
    )
    assert "cobalt blue" in residual.lower()
    assert "chat mode" not in residual.lower()


def test_long_prompt_does_not_false_positive_vision() -> None:
    long = (
        "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket "
        "to improve cursor rendering performance for deepseek applications."
    )
    hit = fuzzy_match_command(long)
    # Must not hijack MoA ticket text into a vision mode short-circuit.
    assert hit is None or hit.target != "vision"
