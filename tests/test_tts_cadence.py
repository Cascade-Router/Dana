"""TTS cadence / sample-rate regression checks for vision OCR reads."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dana.core_agent import (
    DEFAULT_PIPER_ONNX,
    PIPER_EN_ONNX,
    PIPER_LENGTH_SCALE,
    PIPER_VOICE_ID,
    _resample_pcm,
    download_piper_models,
    sanitize_text_for_tts,
)


def test_piper_length_scale_faster_than_realtime_default() -> None:
    assert PIPER_LENGTH_SCALE == 0.75


def test_default_piper_voice_is_hfc_female_medium() -> None:
    assert PIPER_VOICE_ID == "en_US-hfc_female-medium"
    assert PIPER_EN_ONNX.endswith("en_US-hfc_female-medium.onnx")
    assert DEFAULT_PIPER_ONNX == PIPER_EN_ONNX


def test_download_piper_falls_back_to_legacy_when_network_fails(
    tmp_path: Path, monkeypatch
) -> None:
    # download_piper_models's `global` reassignments (PIPER_EN_ONNX, etc.) bind
    # to dana.audio.tts_worker's own namespace, not core_agent's façade
    # re-export -- patch the real module so the function under test actually
    # observes these values.
    import dana.audio.tts_worker as ttsw

    models = tmp_path / "tts_models"
    models.mkdir()
    legacy_onnx = models / "en_US-ljspeech-high.onnx"
    legacy_json = models / "en_US-ljspeech-high.onnx.json"
    legacy_onnx.write_bytes(b"legacy-onnx")
    legacy_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ttsw, "TTS_MODELS_DIR", str(models))
    monkeypatch.setattr(ttsw, "PIPER_VOICE_ID", "en_US-hfc_female-medium")
    monkeypatch.setattr(
        ttsw, "PIPER_EN_ONNX", str(models / "en_US-hfc_female-medium.onnx")
    )
    monkeypatch.setattr(
        ttsw, "PIPER_EN_JSON", str(models / "en_US-hfc_female-medium.onnx.json")
    )
    monkeypatch.setattr(ttsw, "DEFAULT_PIPER_ONNX", ttsw.PIPER_EN_ONNX)
    monkeypatch.setattr(ttsw, "_PIPER_LEGACY_ONNX", str(legacy_onnx))
    monkeypatch.setattr(ttsw, "_PIPER_LEGACY_JSON", str(legacy_json))

    def _boom(url: str, dest: str) -> None:
        raise OSError("network disabled in test")

    with patch.object(ttsw, "_download_file", side_effect=_boom):
        download_piper_models()

    assert ttsw.PIPER_EN_ONNX == str(legacy_onnx)
    assert os.path.basename(ttsw.PIPER_EN_ONNX) == "en_US-ljspeech-high.onnx"


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
