"""Dynamic adaptive noise floor for VAD / Whisper speech gating.

Pure helpers (no mic I/O) so unit tests can drive synthetic RMS samples.
``calibrate_noise_floor`` samples the live mic (via ``dana.audio.mic_input``)
and stores the baseline used by ``compute_dynamic_speech_floor``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dana.core.constants import DEAD_MIC_RMS_FLOOR, SAMPLE_RATE

# Note: dana.core.shared_state itself re-exports names from this module
# (Phase 1), so importing it at module level here would be circular --
# calibrate_noise_floor imports it lazily instead.

# Speech floor = max(absolute_min, ambient_baseline * multiplier).
NOISE_FLOOR_MULTIPLIER = 1.5
# Absolute minimum so total silence / dead virtual mics still gate.
ABSOLUTE_MIN_SPEECH_FLOOR = 0.0001

# Last mic ambient probe — drives adaptive VAD / barge-in floors for quiet headsets.
# Reassigned here AND by dana.audio.mic_input.ensure_live_mic (cross-module writer);
# both sides must go through this module's attribute, never a bare-name copy.
_mic_ambient_rms: float = 0.0
# Dynamic speech gate from calibrate_noise_floor (ambient * 1.5, abs min).
_dynamic_speech_floor: float = 0.0015


def compute_ambient_baseline(rms_samples: Sequence[float]) -> float:
    """Mean ambient RMS from window samples (empty → 0.0)."""
    if not rms_samples:
        return 0.0
    arr = np.asarray(list(rms_samples), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def compute_dynamic_speech_floor(
    ambient_rms_baseline: float,
    *,
    multiplier: float = NOISE_FLOOR_MULTIPLIER,
    absolute_min: float = ABSOLUTE_MIN_SPEECH_FLOOR,
) -> float:
    """Derive speech gate from calibrated ambient baseline.

    ``dynamic_speech_floor = max(absolute_min, ambient_rms_baseline * multiplier)``
    """
    baseline = max(0.0, float(ambient_rms_baseline))
    return max(float(absolute_min), baseline * float(multiplier))


def audio_buffer_rms(samples: np.ndarray | None) -> float:
    """RMS of a float PCM buffer; ``0.0`` for empty / None."""
    if samples is None:
        return 0.0
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def get_dynamic_speech_floor() -> float:
    """Current adaptive speech / Whisper RMS gate (post-calibration)."""
    return float(_dynamic_speech_floor)


def should_skip_wake_predict(
    rms: float,
    *,
    floor: float | None = None,
) -> bool:
    """True when chunk RMS is below the speech floor (skip OpenWakeWord predict)."""
    gate = float(get_dynamic_speech_floor() if floor is None else floor)
    return float(rms) < gate


def calibrate_noise_floor(duration_sec: float = 3.0) -> tuple[float, float]:
    """Sample mic ambient RMS before wake-word arming; set dynamic speech floor.

    Captures short windows over ``duration_sec``, averages ambient ``rms_raw``,
    then sets ``dynamic_speech_floor = max(ABSOLUTE_MIN, ambient * 1.5)``.

    Returns ``(ambient_rms_baseline, dynamic_speech_floor)``.
    """
    global _mic_ambient_rms, _dynamic_speech_floor
    from dana.audio.mic_input import probe_mic_rms
    from dana.core import shared_state as state
    from dana.logging import log

    window_s = 0.25
    n_windows = max(1, int(round(float(duration_sec) / window_s)))
    rms_samples: list[float] = []
    device_idx = state.AUDIO_INPUT_DEVICE
    rate = int(state.AUDIO_INPUT_RATE or SAMPLE_RATE)
    for _ in range(n_windows):
        try:
            rms_samples.append(
                float(probe_mic_rms(device_idx, rate, seconds=window_s))
            )
        except Exception as exc:  # noqa: BLE001
            log("Audio", f"WARNING: noise-floor window probe failed: {exc}")
            rms_samples.append(0.0)
    baseline = compute_ambient_baseline(rms_samples)
    floor = compute_dynamic_speech_floor(
        baseline,
        multiplier=NOISE_FLOOR_MULTIPLIER,
        absolute_min=max(ABSOLUTE_MIN_SPEECH_FLOOR, DEAD_MIC_RMS_FLOOR),
    )
    _mic_ambient_rms = float(baseline)
    _dynamic_speech_floor = float(floor)
    log(
        "Audio",
        f"Noise floor calibrated: ambient_rms_baseline={baseline:.6f}, "
        f"dynamic_speech_floor={floor:.6f} "
        f"(multiplier={NOISE_FLOOR_MULTIPLIER}, windows={n_windows})",
    )
    return float(baseline), float(floor)
