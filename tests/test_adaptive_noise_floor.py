"""Adaptive noise floor — synthetic RMS samples (no mic)."""

from __future__ import annotations

from dana.audio.noise_floor import (
    ABSOLUTE_MIN_SPEECH_FLOOR,
    NOISE_FLOOR_MULTIPLIER,
    compute_ambient_baseline,
    compute_dynamic_speech_floor,
)


def test_compute_ambient_baseline_mean() -> None:
    samples = [0.001, 0.002, 0.003, 0.002]
    assert abs(compute_ambient_baseline(samples) - 0.002) < 1e-9


def test_compute_ambient_baseline_empty() -> None:
    assert compute_ambient_baseline([]) == 0.0


def test_dynamic_speech_floor_multiplier() -> None:
    baseline = 0.004
    expected = baseline * NOISE_FLOOR_MULTIPLIER
    assert abs(compute_dynamic_speech_floor(baseline) - expected) < 1e-12


def test_dynamic_speech_floor_absolute_min_on_silence() -> None:
    floor = compute_dynamic_speech_floor(0.0)
    assert floor == ABSOLUTE_MIN_SPEECH_FLOOR
    # Tiny ambient still clamps to absolute min when ambient * 1.5 is lower.
    tiny = ABSOLUTE_MIN_SPEECH_FLOOR / 10.0
    assert compute_dynamic_speech_floor(tiny) == ABSOLUTE_MIN_SPEECH_FLOOR


def test_dynamic_speech_floor_custom_multiplier() -> None:
    assert abs(compute_dynamic_speech_floor(0.01, multiplier=2.0) - 0.02) < 1e-12


if __name__ == "__main__":
    test_compute_ambient_baseline_mean()
    test_compute_ambient_baseline_empty()
    test_dynamic_speech_floor_multiplier()
    test_dynamic_speech_floor_absolute_min_on_silence()
    test_dynamic_speech_floor_custom_multiplier()
    print("OK")
