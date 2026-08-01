"""Dynamic adaptive noise floor for VAD / Whisper speech gating.

Pure helpers (no mic I/O) so unit tests can drive synthetic RMS samples.
``calibrate_noise_floor`` in ``core_agent`` samples the live mic and stores
the baseline used by ``compute_dynamic_speech_floor``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Speech floor = max(absolute_min, ambient_baseline * multiplier).
NOISE_FLOOR_MULTIPLIER = 1.5
# Absolute minimum so total silence / dead virtual mics still gate.
ABSOLUTE_MIN_SPEECH_FLOOR = 0.0001


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
