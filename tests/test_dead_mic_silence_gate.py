"""Dead-mic RMS floor — skip OpenWakeWord predict on synthetic silence."""

from __future__ import annotations

import numpy as np

from dana.core.constants import DEAD_MIC_RMS_FLOOR
from dana.audio.noise_floor import audio_buffer_rms, should_skip_wake_predict


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


if __name__ == "__main__":
    test_dead_mic_floor_constant()
    test_audio_buffer_rms_silence_and_empty()
    test_audio_buffer_rms_known_tone()
    test_should_skip_wake_predict_below_and_above_floor()
    print("OK")
