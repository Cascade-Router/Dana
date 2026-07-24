"""Stage 8.1 — Multi-voice mixer ducking + Jason Andon announce."""

from __future__ import annotations

import time
from pathlib import Path

from donna.audio.multi_voice_tts import (
    JASON_ANDON_LINE,
    synthesize_speech,
    write_tone_wav,
)
from donna.ui import audio_mixer


def test_synthesize_speech_voice_id(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_AUDIO_DRY_RUN", "1")
    # Force silence/placeholder path (no pygame device required).
    out_d = synthesize_speech("Hello receptionist.", voice_id="donna", out_path=tmp_path / "d.wav")
    out_j = synthesize_speech(JASON_ANDON_LINE, voice_id="jason", out_path=tmp_path / "j.wav")
    assert out_d.is_file() and out_d.stat().st_size > 44
    assert out_j.is_file() and out_j.stat().st_size > 44


def test_jason_ducks_donna_then_restores(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_AUDIO_DRY_RUN", "1")
    audio_mixer.clear_dry_events()
    # Reset mixer init flag for dry path.
    audio_mixer._INIT_DONE = False  # noqa: SLF001
    audio_mixer._MIXER_OK = False  # noqa: SLF001

    donna_wav = write_tone_wav(tmp_path / "donna_long.wav", duration_s=3.0, freq_hz=440.0)
    jason_wav = write_tone_wav(tmp_path / "jason_line.wav", duration_s=0.8, freq_hz=220.0)

    audio_mixer.ensure_mixer()
    audio_mixer.play_donna(donna_wav, block=False)
    time.sleep(0.35)
    assert abs(audio_mixer.get_donna_volume() - 1.0) < 0.05

    # Trigger Jason override while Donna is "speaking".
    audio_mixer.play_jason(jason_wav, block=True)
    events = audio_mixer.dry_events()
    kinds = [e["event"] for e in events]
    assert "duck" in kinds
    assert "restore" in kinds
    duck_i = kinds.index("duck")
    restore_i = kinds.index("restore")
    assert duck_i < restore_i
    assert events[duck_i]["volume"] == 0.2
    assert events[restore_i]["volume"] == 1.0
    assert abs(audio_mixer.get_donna_volume() - 1.0) < 0.05


def test_recovery_mode_triggers_jason_voice(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_AUDIO_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    from donna.management.jason_supervisor import recovery_mode
    from donna.memory.blackboard import init_blackboard, set_sensor_state

    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_sensor_state("latest_visual_context", "empty desk", db_path=db)

    called: list[dict] = []

    def _announce(*, block: bool = False):  # noqa: ANN001
        called.append({"block": block})
        return {"ok": True, "line": JASON_ANDON_LINE, "path": str(tmp_path / "j.wav")}

    monkeypatch.setattr(
        "donna.management.jason_supervisor.announce_jason_andon_override",
        _announce,
    )
    result = recovery_mode(
        failed_action_id=1,
        failed_tool="navigate_and_click",
        error_context="target box not found",
        failed_arguments={"query": "Target"},
        session_id="voice-test",
        db_path=db,
    )
    assert result.get("ok") is True
    assert called, "recovery_mode should announce Jason voice"
    assert result.get("voice", {}).get("ok") is True
