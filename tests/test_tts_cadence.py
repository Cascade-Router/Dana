"""TTS cadence / sample-rate regression checks for vision OCR reads."""

from __future__ import annotations

from dana.audio.tts_worker import (
    DEFAULT_PIPER_ONNX,
    PIPER_EN_ONNX,
    PIPER_LENGTH_SCALE,
    PIPER_VOICE_ID,
)
from dana.audio.tts_manager import sanitize_text_for_tts


def test_piper_length_scale_faster_than_realtime_default() -> None:
    assert PIPER_LENGTH_SCALE == 0.75


def test_default_piper_voice_is_hfc_female_medium() -> None:
    assert PIPER_VOICE_ID == "en_US-hfc_female-medium"
    assert PIPER_EN_ONNX.endswith("en_US-hfc_female-medium.onnx")
    assert DEFAULT_PIPER_ONNX == PIPER_EN_ONNX


def test_sanitize_inserts_pauses_for_ocr_newlines() -> None:
    raw = "Submit\nError\nTraceback\nFileNotFound"
    out = sanitize_text_for_tts(raw)
    assert out.count(".") >= 2
    assert "Submit" in out and "Traceback" in out
