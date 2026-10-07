"""Piper text-to-speech: the voice model to use, a cached loader, and WAV
synthesis (used by ``multi_voice_tts``)."""

from __future__ import annotations

import os
import time
import wave
from typing import Any

from dana.logging import log, log_debug
from dana.paths import TTS_MODELS_DIR as _TTS_DIR

TTS_MODELS_DIR = str(_TTS_DIR)

# Default voice: en_US-hfc_female-medium (CC BY-NC-SA 4.0 — see docs/LEGAL_AND_IP.md).
# Override via DANA_PIPER_VOICE (e.g. en_US-ljspeech-high for public-domain weights).
PIPER_VOICE_ID = (
    os.environ.get("DANA_PIPER_VOICE", "en_US-hfc_female-medium").strip()
    or "en_US-hfc_female-medium"
)
PIPER_EN_ONNX = os.path.join(TTS_MODELS_DIR, f"{PIPER_VOICE_ID}.onnx")
DEFAULT_PIPER_ONNX = PIPER_EN_ONNX

# length_scale < 1.0 speeds speech (VITS). Default 0.75 for snappier replies.
try:
    PIPER_LENGTH_SCALE = float(os.environ.get("DANA_PIPER_LENGTH_SCALE", "0.75"))
except ValueError:
    PIPER_LENGTH_SCALE = 0.75
PIPER_LENGTH_SCALE = max(0.5, min(2.0, PIPER_LENGTH_SCALE))

_piper_voice_cache: dict[str, Any] = {}


def get_piper_voice(model_path: str) -> Any:
    """Load (and cache) a PiperVoice for the given .onnx path (lazy onnx import)."""
    voice = _piper_voice_cache.get(model_path)
    if voice is not None:
        return voice
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Piper model missing: {model_path}")
    from piper import PiperVoice

    log("Audio", f"Loading Piper voice: {os.path.basename(model_path)}")
    t_load = time.perf_counter()
    voice = PiperVoice.load(model_path)
    _piper_voice_cache[model_path] = voice
    try:
        from dana.perf import log_perf

        log_perf(
            "piper_voice_load",
            (time.perf_counter() - t_load) * 1000.0,
            model=os.path.basename(model_path),
        )
    except Exception:  # noqa: BLE001
        pass
    return voice


def synthesize_to_file(voice: Any, text: str, path: str) -> bool:
    """Write Piper speech to a WAV path.

    Collects audio from ``voice.synthesize`` first so empty/failed TTS never
    opens a half-initialized ``wave`` writer (``# channels not specified``).

    Returns:
        True when a valid WAV was written; False when TTS produced no audio
        (caller should skip playback).
    """
    from piper.config import SynthesisConfig

    from dana.audio.tts_manager import sanitize_text_for_tts

    # Defaults used when the voice omits format metadata.
    channels = 1
    sampwidth = 2
    framerate = 22050
    try:
        cfg_rate = int(getattr(getattr(voice, "config", None), "sample_rate", 0) or 0)
        if cfg_rate > 0:
            framerate = cfg_rate
    except Exception:  # noqa: BLE001
        pass

    utterance = sanitize_text_for_tts(text or "")
    if not utterance:
        # Empty / markdown-only input — skip without warning spam.
        return False

    chunks: list[Any] = []
    piper_bytes = 0
    t_piper0 = time.perf_counter()
    ttfb_logged = False
    syn_config = SynthesisConfig(length_scale=float(PIPER_LENGTH_SCALE))
    try:
        for chunk in voice.synthesize(utterance, syn_config=syn_config):
            if chunk is None:
                continue
            try:
                raw = chunk.audio_int16_bytes
            except Exception:  # noqa: BLE001
                raw = b""
            if not raw:
                continue
            if not ttfb_logged:
                ttfb_logged = True
                try:
                    from dana.perf import log_perf

                    log_perf(
                        "piper_ttfb",
                        (time.perf_counter() - t_piper0) * 1000.0,
                        chars=len(utterance),
                    )
                except Exception:  # noqa: BLE001
                    pass
            piper_bytes += len(raw)
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    if not chunks:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False

    log_debug(
        "Audio",
        f"Piper synthesize chunks={len(chunks)} bytes={piper_bytes} "
        f"chars={len(utterance)} length_scale={PIPER_LENGTH_SCALE} "
        f"dt_ms={(time.perf_counter() - t_piper0) * 1000.0:.1f}",
    )

    first = chunks[0]
    try:
        channels = int(getattr(first, "sample_channels", None) or channels)
        sampwidth = int(getattr(first, "sample_width", None) or sampwidth)
        framerate = int(getattr(first, "sample_rate", None) or framerate)
    except Exception:  # noqa: BLE001
        pass
    if channels < 1:
        channels = 1
    if sampwidth < 1:
        sampwidth = 2
    if framerate < 1:
        framerate = 22050

    try:
        with wave.open(path, "wb") as wav_file:
            # Explicit format BEFORE any frames (required by wave module).
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            for chunk in chunks:
                try:
                    frame_bytes = chunk.audio_int16_bytes
                except Exception:  # noqa: BLE001
                    frame_bytes = b""
                if frame_bytes:
                    wav_file.writeframes(frame_bytes)
    except Exception as exc:  # noqa: BLE001
        log(
            "Audio",
            f"WARNING: TTS returned empty audio data, skipping synthesis ({exc})",
        )
        return False

    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 44:
            log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
            return False
    except OSError:
        log("Audio", "WARNING: TTS returned empty audio data, skipping synthesis")
        return False
    return True


__all__ = ("DEFAULT_PIPER_ONNX", "PIPER_LENGTH_SCALE", "get_piper_voice", "synthesize_to_file")
