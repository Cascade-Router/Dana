"""Unit tests for Phase-2 production features + multi-forge routing."""

from __future__ import annotations

from dana.swarm.multi_forge import looks_like_multi_forge, split_forge_goals
from dana.tools.broker import _TOOL_FORGE_HINT_RE, get_broker
from dana.telemetry import note_tool_event, write_dashboard
from dana.paths import DASHBOARD_PATH
from dana.audio.vad_consumer import SILERO_SPEECH_THRESHOLD, SILERO_WINDOW_SAMPLES
from dana.core_agent import VAD_FRAME_SAMPLES, VAD_SILENCE_MS


MASS = (
    "Donna, build three different tools back-to-back: one that tells me the time, "
    "one that generates a random number, and one that lists files in the sandbox."
)


def test_forge_hint_matches_batch() -> None:
    assert _TOOL_FORGE_HINT_RE.search(MASS)
    call = get_broker().parse_utterance(MASS)
    assert call is not None
    assert call.tool_id == "architect_new_tool", call
    print("[PASS] broker routes mass-forge to architect_new_tool")


def test_split_three_goals() -> None:
    assert looks_like_multi_forge(MASS)
    goals = split_forge_goals(MASS)
    assert len(goals) == 3, goals
    assert any("time" in g.lower() for g in goals)
    assert any("random" in g.lower() for g in goals)
    assert any("sandbox" in g.lower() or "files" in g.lower() for g in goals)
    print("[PASS] split_forge_goals → 3")


def test_silero_vad_frame_config() -> None:
    assert SILERO_WINDOW_SAMPLES == 512
    assert VAD_FRAME_SAMPLES == SILERO_WINDOW_SAMPLES
    assert SILERO_SPEECH_THRESHOLD == 0.5
    assert VAD_SILENCE_MS == 1500
    print("[PASS] Silero VAD frame / threshold config")


def test_dashboard_write() -> None:
    note_tool_event("forge:demo_tool")
    path = write_dashboard(status="Healthy", pid=12345)
    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "System Status" in text
    assert "12345" in text
    assert "PENDING" in text
    assert path.endswith("dashboard.md")
    print("[PASS] dashboard.md write")


if __name__ == "__main__":
    test_forge_hint_matches_batch()
    test_split_three_goals()
    test_silero_vad_frame_config()
    test_dashboard_write()
    print("OK")
