"""Hermetic checks for mss + pytesseract screen OCR actuator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dana.tools.vision import analyze_visual_context


def test_analyze_visual_context_returns_screen_text() -> None:
    fake_shot = MagicMock()
    fake_shot.size = (2, 2)
    fake_shot.bgra = b"\x00\x00\x00\xff" * 4

    fake_sct = MagicMock()
    fake_sct.monitors = [{}, {"left": 0, "top": 0, "width": 2, "height": 2}]
    fake_sct.grab.return_value = fake_shot
    fake_sct.__enter__ = MagicMock(return_value=fake_sct)
    fake_sct.__exit__ = MagicMock(return_value=False)

    with (
        patch("mss.MSS", return_value=fake_sct),
        patch("pytesseract.image_to_string", return_value="Hello Screen\n"),
    ):
        out = analyze_visual_context()

    # Current contract is a natural-language wrapper (not the old <screen_text>
    # XML tag); anchor on the deterministic prefix since a "Recent visual
    # history: ..." trailer may follow depending on episodic-store state.
    assert out.startswith("Current screen: On-screen text/HUD: Hello Screen.")


def test_analyze_visual_context_missing_tesseract_binary() -> None:
    from pytesseract import TesseractNotFoundError

    fake_shot = MagicMock()
    fake_shot.size = (2, 2)
    fake_shot.bgra = b"\x00\x00\x00\xff" * 4

    fake_sct = MagicMock()
    fake_sct.monitors = [{}, {"left": 0, "top": 0, "width": 2, "height": 2}]
    fake_sct.grab.return_value = fake_shot
    fake_sct.__enter__ = MagicMock(return_value=fake_sct)
    fake_sct.__exit__ = MagicMock(return_value=False)

    with (
        patch("mss.MSS", return_value=fake_sct),
        patch(
            "pytesseract.image_to_string",
            side_effect=TesseractNotFoundError(),
        ),
    ):
        out = analyze_visual_context()

    assert out == "SYSTEM_ERROR: Tesseract OCR binary not found on host OS."
