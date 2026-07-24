"""Stage 8.1 — Lightweight multi-voice TTS (donna / jason).

``donna`` prefers Piper (existing female voice) when available.
``jason`` always uses OS ``pyttsx3`` (male Windows voice) — zero GPU/VRAM.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Literal

VoiceId = Literal["donna", "jason"]

JASON_ANDON_LINE = (
    "Operator failure detected. Taking control of the visual state."
)


def _log(msg: str) -> None:
    try:
        from donna.logging import log

        log("MultiVoiceTTS", msg)
    except Exception:  # noqa: BLE001
        print(f"[MultiVoiceTTS] {msg}", flush=True)


def synthesize_speech(
    text: str,
    *,
    voice_id: VoiceId | str = "donna",
    out_path: str | Path | None = None,
) -> Path:
    """Synthesize ``text`` for ``voice_id`` and return a WAV path.

    Jason never loads CUDA models (VRAM hard-cap friendly).
    """
    body = (text or "").strip() or "..."
    vid = (voice_id or "donna").strip().lower()
    if vid not in {"donna", "jason"}:
        vid = "donna"

    if out_path is None:
        fd, name = tempfile.mkstemp(prefix=f"donna_{vid}_", suffix=".wav")
        os.close(fd)
        dest = Path(name)
    else:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    if vid == "jason":
        ok = _synthesize_pyttsx3(body, dest, prefer_male=True)
        if not ok:
            _write_silence_wav(dest, duration_s=1.2)
            _log("jason pyttsx3 unavailable — wrote silence placeholder")
        return dest

    # donna — Piper first, then pyttsx3 female fallback.
    if _synthesize_piper(body, dest):
        return dest
    if _synthesize_pyttsx3(body, dest, prefer_male=False):
        return dest
    _write_silence_wav(dest, duration_s=1.0)
    _log("donna TTS fallback — wrote silence placeholder")
    return dest


def synthesize_jason_andon_line(*, out_path: str | Path | None = None) -> Path:
    """CTO Andon recovery utterance."""
    return synthesize_speech(
        JASON_ANDON_LINE,
        voice_id="jason",
        out_path=out_path,
    )


def _synthesize_piper(text: str, dest: Path) -> bool:
    try:
        from donna.core_agent import (
            DEFAULT_PIPER_ONNX,
            get_piper_voice,
            synthesize_to_file,
        )

        model = str(DEFAULT_PIPER_ONNX)
        if not Path(model).is_file():
            return False
        voice = get_piper_voice(model)
        return bool(synthesize_to_file(voice, text, str(dest)))
    except Exception as exc:  # noqa: BLE001
        _log(f"piper synth skipped: {exc}")
        return False


def _synthesize_pyttsx3(text: str, dest: Path, *, prefer_male: bool) -> bool:
    """Offline OS TTS (SAPI5 on Windows) — no VRAM."""
    try:
        import pyttsx3
    except Exception as exc:  # noqa: BLE001
        _log(f"pyttsx3 import failed: {exc}")
        return False
    try:
        engine = pyttsx3.init()
        voices = list(engine.getProperty("voices") or [])
        chosen = None
        for v in voices:
            name = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
            if prefer_male and any(
                k in name for k in ("david", "male", "mark", "james", "richard")
            ):
                chosen = v
                break
            if not prefer_male and any(
                k in name for k in ("zira", "female", "hazel", "susan")
            ):
                chosen = v
                break
        if chosen is None and voices:
            # Prefer last voice for male when unmarked; first for female.
            chosen = voices[-1] if prefer_male else voices[0]
        if chosen is not None:
            engine.setProperty("voice", chosen.id)
        engine.setProperty("rate", 175 if prefer_male else 165)
        # save_to_file is async until runAndWait.
        engine.save_to_file(text, str(dest))
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass
        return dest.is_file() and dest.stat().st_size > 44
    except Exception as exc:  # noqa: BLE001
        _log(f"pyttsx3 synth failed: {exc}")
        return False


def _write_silence_wav(path: Path, *, duration_s: float = 1.0, rate: int = 22050) -> None:
    n_frames = max(1, int(rate * float(duration_s)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def write_tone_wav(
    path: str | Path,
    *,
    duration_s: float = 1.0,
    freq_hz: float = 440.0,
    rate: int = 22050,
    amplitude: float = 0.2,
) -> Path:
    """Utility for dry ducking demos (no TTS dependency)."""
    import math
    import struct

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(rate * float(duration_s)))
    frames = bytearray()
    for i in range(n):
        sample = amplitude * math.sin(2.0 * math.pi * freq_hz * (i / rate))
        frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return dest
