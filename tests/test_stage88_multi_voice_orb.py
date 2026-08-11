"""Stage 8.8 — agent_id voice routing + Assistive Orb pulse animation."""

from __future__ import annotations

import math
import os

import pytest

from dana.audio.multi_voice_tts import (
    AGENT_VOICE_MAP,
    PERSONA_COLORS,
    get_active_tts_agent,
    persona_color_for_agent,
    resolve_voice_id,
    set_active_tts_agent,
    svg_logo_loader_notes,
    synthesize_speech,
    uses_receptionist_piper,
    VOICE_PROFILE_LABELS,
)
from dana.audio.tts_manager import (
    _parse_tts_spool_item,
    chunk_text_for_tts,
    enqueue_speech_impl as enqueue_speech,
)


@pytest.fixture(autouse=True)
def _dry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DANA_AUDIO_DRY_RUN", "1")
    set_active_tts_agent("broker")


def test_agent_voice_map_and_profiles() -> None:
    assert resolve_voice_id("broker") == "dana"
    assert resolve_voice_id("llama") == "dana"
    assert resolve_voice_id("moa") == "moa"
    assert resolve_voice_id("MoA_Reasoner") == "moa"
    assert resolve_voice_id("deepseek") == "moa"
    assert resolve_voice_id("vision") == "vision"
    assert resolve_voice_id("yolo") == "vision"
    assert resolve_voice_id("jason") == "jason"
    assert resolve_voice_id("ghost_typist") == "typist"
    assert uses_receptionist_piper("broker")
    assert not uses_receptionist_piper("moa")
    assert "voice_2_deep" in VOICE_PROFILE_LABELS["moa"]
    assert "broker" in AGENT_VOICE_MAP
    assert persona_color_for_agent("broker") == PERSONA_COLORS["broker"]
    assert persona_color_for_agent("moa") == PERSONA_COLORS["moa"]
    assert persona_color_for_agent("vision") == PERSONA_COLORS["vision"]
    notes = svg_logo_loader_notes()
    assert "preferred_asset" in notes


def test_synthesize_speech_by_agent_id(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DANA_AUDIO_DRY_RUN", "1")
    for agent, voice in (("broker", "dana"), ("moa", "moa"), ("vision", "vision")):
        out = synthesize_speech(
            f"Hello from {agent}.",
            agent_id=agent,
            out_path=tmp_path / f"{agent}.wav",
        )
        assert out.is_file() and out.stat().st_size > 44
        assert resolve_voice_id(agent) == voice


def test_enqueue_speech_carries_agent_id(monkeypatch) -> None:  # noqa: ANN001
    from dana.audio import tts_manager
    from dana.core.shared_state import tts_queue
    from dana.agentic import StreamSentenceTtsBuffer, feed_stream_tts, reset_stream_sentence_tts

    # Drain any leftovers.
    tts_manager.flush_tts_queue()
    enqueue_speech("Short line one.", agent_id="moa")
    assert get_active_tts_agent() == "moa"
    item = tts_queue.get_nowait()
    text, flag, agent = _parse_tts_spool_item(item)
    assert "Short" in text
    assert isinstance(flag, bool)
    assert agent == "moa"
    # Streaming path emits sentence-complete chunks before the full node ends.
    buf = StreamSentenceTtsBuffer()
    assert buf.feed("Hello there. ") == ["Hello there."]
    assert buf.feed("More tokens") == []
    assert buf.flush() == "More tokens"
    # Long spool chunking still splits oversized replies.
    long = ("Word. " * 40).strip()
    pieces = chunk_text_for_tts(long, max_chars=80)
    assert len(pieces) >= 2
    reset_stream_sentence_tts()
    _ = feed_stream_tts  # imported for API surface check


def test_stream_tts_agent_helpers() -> None:
    from dana.agentic import (
        agent_id_from_label,
        get_stream_tts_agent,
        set_stream_tts_agent,
    )

    assert agent_id_from_label("MoA_Reasoner") == "moa"
    assert agent_id_from_label("Vision_Agent") == "vision"
    assert agent_id_from_label("ReAct_Agent") == "broker"
    set_stream_tts_agent("deepseek")
    assert get_stream_tts_agent() == "deepseek"
    assert get_active_tts_agent() == "deepseek"


def test_orb_pulse_animation_scales_and_colors() -> None:
    import customtkinter as ctk

    from dana.ui.assistive_orb import (
        AssistiveTouchOrb,
        _ICON_SIZE_MAX,
        _ICON_SIZE_MIN,
    )

    root = ctk.CTk()
    root.withdraw()
    set_active_tts_agent("moa")
    orb = AssistiveTouchOrb(
        root,
        agent_getter=lambda: "moa",
        dictation_getter=lambda: False,
        mode_getter=lambda: "chat",
    )
    root.update_idletasks()
    root.update()

    assert hasattr(orb, "pulse_animation")
    assert orb._canvas.winfo_exists()
    # Drive a few animation frames and ensure phase advances.
    phase0 = orb._pulse_phase
    orb.pulse_animation()
    root.update_idletasks()
    assert orb._pulse_phase != phase0
    # Color bound to MoA red persona.
    assert orb._accent().lower() == PERSONA_COLORS["moa"].lower()
    mid = (_ICON_SIZE_MIN + _ICON_SIZE_MAX) / 2.0
    amp = (_ICON_SIZE_MAX - _ICON_SIZE_MIN) / 2.0
    size = mid + amp * math.sin(orb._pulse_phase)
    assert _ICON_SIZE_MIN <= size <= _ICON_SIZE_MAX
    items = orb._canvas.find_withtag("icon")
    assert items
    assert orb._canvas.type(items[0]) in {"image", "polygon"}
    assert orb._logo_mode in {"png", "polygon"}

    set_active_tts_agent("vision")
    orb._active_agent = "vision"
    assert orb._accent().lower() == PERSONA_COLORS["vision"].lower()

    orb.destroy()
    root.destroy()
