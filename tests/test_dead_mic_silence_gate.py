"""Dead-mic RMS floor — skip OpenWakeWord predict on synthetic silence."""

from __future__ import annotations

import numpy as np

from dana.core_agent import (
    DEAD_MIC_RMS_FLOOR,
    audio_buffer_rms,
    prioritize_text_input,
    should_skip_wake_predict,
    vad_abort_event,
    vad_capture_active,
)


def test_dead_mic_floor_constant() -> None:
    assert DEAD_MIC_RMS_FLOOR == 0.0001


def test_audio_buffer_rms_silence_and_empty() -> None:
    silence = np.zeros(1280, dtype=np.float32)
    assert audio_buffer_rms(silence) == 0.0
    assert audio_buffer_rms(None) == 0.0
    assert audio_buffer_rms(np.array([], dtype=np.float32)) == 0.0


def test_audio_buffer_rms_known_tone() -> None:
    # Constant |x|=0.01 → RMS == 0.01
    buf = np.full(1600, 0.01, dtype=np.float32)
    assert abs(audio_buffer_rms(buf) - 0.01) < 1e-6


def test_should_skip_wake_predict_below_and_above_floor() -> None:
    # Explicit absolute floor (dynamic gate is calibrated at session boot).
    assert should_skip_wake_predict(0.0, floor=DEAD_MIC_RMS_FLOOR) is True
    assert should_skip_wake_predict(
        DEAD_MIC_RMS_FLOOR * 0.5, floor=DEAD_MIC_RMS_FLOOR
    ) is True
    assert should_skip_wake_predict(
        DEAD_MIC_RMS_FLOOR, floor=DEAD_MIC_RMS_FLOOR
    ) is False
    assert should_skip_wake_predict(0.01, floor=DEAD_MIC_RMS_FLOOR) is False
    # Near-silent virtual-mic noise still below floor.
    assert should_skip_wake_predict(5e-5, floor=DEAD_MIC_RMS_FLOOR) is True


def test_prioritize_text_input_sets_vad_abort() -> None:
    vad_abort_event.clear()
    vad_capture_active.set()
    try:
        prioritize_text_input(reason="unit_test")
        assert vad_abort_event.is_set()
    finally:
        vad_capture_active.clear()
        vad_abort_event.clear()


if __name__ == "__main__":
    test_dead_mic_floor_constant()
    test_audio_buffer_rms_silence_and_empty()
    test_audio_buffer_rms_known_tone()
    test_should_skip_wake_predict_below_and_above_floor()
    test_prioritize_text_input_sets_vad_abort()
    print("OK")
