"""Silero VAD speech detection + TTS barge-in onset hooks.

Replaces legacy WebRTC VAD. Mic frames are float32 @ 16 kHz; Silero expects
512-sample windows. Probability > 0.38 counts as speech (quiet-mic tuned).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from dana.audio.tts_worker import get_tts_worker

_log = logging.getLogger("dana.audio.vad_consumer")

# Silero v4/v5 window @ 16 kHz (32 ms).
SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW_SAMPLES = 512
# Headphone / quiet conversational speech — 0.5 was too strict after gain.
SILERO_SPEECH_THRESHOLD = 0.38

# Quiet-mic path: boost frames for Silero only (capture buffer stays raw).
# Target matches Whisper normalization; max gain is higher so ~0.0001 RMS speech
# can still cross speech_prob > 0.38 without inventing onset on dead silence.
VAD_TARGET_RMS = 0.05
VAD_GAIN_RMS_CEIL = 0.005
VAD_MAX_GAIN = 256.0
# Absolute floor for gain eligibility (below calibrated speech floor when needed).
VAD_MIN_RMS_FOR_GAIN = 0.00005

_model_lock = threading.Lock()
_silero_model: Any | None = None
_silero_load_error: str | None = None


def trigger_tts_barge_in(*, reason: str = "vad_onset") -> int:
    """Invoke when valid speech / wake-word is detected while TTS is playing.

    Safe to call from the barge-in watcher or ``record_utterance`` VAD path.
    Returns the number of flushed TTS spool items.
    """
    return get_tts_worker().interrupt(reason=reason, set_listening=True)


def get_silero_vad() -> Any:
    """Lazy-load Silero VAD once (CPU). Raises RuntimeError if unavailable."""
    global _silero_model, _silero_load_error
    with _model_lock:
        if _silero_model is not None:
            return _silero_model
        if _silero_load_error is not None:
            raise RuntimeError(_silero_load_error)
        try:
            import torch

            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            model.eval()
            # Tiny model — keep on CPU for predictable audio-thread latency.
            try:
                model = model.to("cpu")
            except Exception:  # noqa: BLE001
                pass
            _silero_model = model
            _log.info("Silero VAD loaded (cpu, window=%d @ %d Hz)", SILERO_WINDOW_SAMPLES, SILERO_SAMPLE_RATE)
            return _silero_model
        except Exception as exc:  # noqa: BLE001
            _silero_load_error = f"Silero VAD load failed: {exc}"
            _log.error(_silero_load_error)
            raise RuntimeError(_silero_load_error) from exc


def reset_silero_states() -> None:
    """Clear Silero streaming state at the start of each utterance."""
    try:
        model = get_silero_vad()
        reset = getattr(model, "reset_states", None)
        if callable(reset):
            reset()
    except Exception as exc:  # noqa: BLE001
        _log.debug("Silero reset_states skipped: %s", exc)


def frame_rms(samples: np.ndarray) -> float:
    """RMS of a float PCM frame (never zero — epsilon for log/ratio safety)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 1e-9
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)


def prepare_frame_for_silero(
    samples: np.ndarray,
    *,
    noise_floor: float = 0.0,
    target_rms: float = VAD_TARGET_RMS,
    gain_ceil: float = VAD_GAIN_RMS_CEIL,
    max_gain: float = VAD_MAX_GAIN,
) -> tuple[np.ndarray, float, float]:
    """Gain-normalize a quiet mic frame before Silero evaluation.

    Applies dynamic gain when ``noise_floor < rms < gain_ceil`` so Silero sees
    speech-scale energy. Below the noise floor (or already loud enough) the
    frame is returned unchanged. Peak-clipped to [-1, 1] after boost.

    Returns ``(frame_for_vad, rms_raw, gain_applied)``.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
    rms = frame_rms(x)
    # Absolute quiet-mic floor (0.00005). Cap calibrated noise_floor so
    # ~0.00014 cannot block soft conversational speech around ~0.0001.
    _ = noise_floor  # retained for API compatibility / future ambient gating
    floor = float(VAD_MIN_RMS_FOR_GAIN)
    if rms < floor or rms >= float(gain_ceil):
        return x, rms, 1.0
    gain = min(float(target_rms) / rms, float(max_gain))
    if gain <= 1.01:
        return x, rms, 1.0
    x *= float(gain)
    np.clip(x, -1.0, 1.0, out=x)
    return x, rms, float(gain)


def _to_silero_tensor(samples: np.ndarray) -> Any:
    """Normalize float PCM to a 512-sample CPU float32 tensor in [-1, 1]."""
    import torch

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        x = np.zeros(SILERO_WINDOW_SAMPLES, dtype=np.float32)
    elif x.size < SILERO_WINDOW_SAMPLES:
        pad = np.zeros(SILERO_WINDOW_SAMPLES, dtype=np.float32)
        pad[: x.size] = x
        x = pad
    else:
        x = x[:SILERO_WINDOW_SAMPLES]
    # Mic path is already float32 ~[-1, 1]; clamp in case of overshoot.
    np.clip(x, -1.0, 1.0, out=x)
    return torch.from_numpy(x)


def speech_probability(
    samples: np.ndarray,
    *,
    sample_rate: int = SILERO_SAMPLE_RATE,
) -> float:
    """Return Silero speech probability for one window (0.0–1.0)."""
    import torch

    model = get_silero_vad()
    tensor = _to_silero_tensor(samples)
    with torch.inference_mode():
        prob = model(tensor, int(sample_rate))
    try:
        return float(prob.item())
    except Exception:  # noqa: BLE001
        return float(prob)


def is_speech_frame(
    samples: np.ndarray,
    *,
    sample_rate: int = SILERO_SAMPLE_RATE,
    threshold: float = SILERO_SPEECH_THRESHOLD,
) -> bool:
    """True when Silero speech probability exceeds ``threshold`` (default 0.5)."""
    try:
        return speech_probability(samples, sample_rate=sample_rate) > float(threshold)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Silero is_speech_frame failed: %s", exc)
        return False


def pcm_int16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to float32 in [-1, 1]."""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
