"""TTS cadence / sample-rate regression checks for vision OCR reads."""

from __future__ import annotations

import numpy as np

from donna.core_agent import (
    PIPER_LENGTH_SCALE,
    _resample_pcm,
    sanitize_text_for_tts,
)


def test_piper_length_scale_slower_than_realtime_default() -> None:
    assert PIPER_LENGTH_SCALE >= 1.15


def test_sanitize_inserts_pauses_for_ocr_newlines() -> None:
    raw = "Submit\nError\nTraceback\nFileNotFound"
    out = sanitize_text_for_tts(raw)
    assert out.count(".") >= 2
    assert "Submit" in out and "Traceback" in out


def test_resample_22050_to_44100_doubles_samples() -> None:
    # 0.2s of audio at 22050 → ~4410 samples; at 44100 → ~8820
    n = 4410
    src = np.sin(np.linspace(0, 8 * np.pi, n, dtype=np.float32))
    out = _resample_pcm(src, 22050, 44100)
    assert abs(out.size - 8820) <= 2
