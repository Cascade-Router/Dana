"""Regression coverage for dana.tools.os_control's window-capture fix:
capture_window_png_bytes must try PrintWindow (immune to Z-order/occlusion,
no focus-steal) before falling back to an on-screen region grab, and must
treat a blank/black PrintWindow result as a failure rather than trusting it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dana.tools import os_control


def test_capture_via_printwindow_rejects_non_positive_dimensions() -> None:
    assert os_control._capture_window_via_printwindow(123, 0, 100) is None
    assert os_control._capture_window_via_printwindow(123, 100, 0) is None
    assert os_control._capture_window_via_printwindow(123, -5, 100) is None


def test_capture_via_printwindow_returns_none_when_print_window_reports_failure() -> None:
    with patch("ctypes.windll.user32.PrintWindow", return_value=0):
        assert os_control._capture_window_via_printwindow(123, 100, 100) is None


def test_capture_via_printwindow_rejects_a_blank_result() -> None:
    """A PrintWindow call that "succeeds" (non-zero return) but paints
    nothing typically comes back as a single flat color — this must be
    detected and rejected rather than trusted as a real screenshot."""
    import numpy as np
    from PIL import Image

    blank = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))

    # Exercise the real function's blank-detection by mocking the GDI layer
    # to return a flat black bitmap via GetBitmapBits.
    fake_bitmap = MagicMock()
    fake_bitmap.GetInfo.return_value = {"bmWidth": 50, "bmHeight": 50}
    fake_bitmap.GetBitmapBits.return_value = blank.convert("RGBA").tobytes()

    fake_save_dc = MagicMock()
    fake_mem_dc = MagicMock()
    fake_mem_dc.CreateCompatibleDC.return_value = fake_save_dc

    with (
        patch("win32gui.GetWindowDC", return_value=111),
        patch("win32ui.CreateDCFromHandle", return_value=fake_mem_dc),
        patch("win32ui.CreateBitmap", return_value=fake_bitmap),
        patch("ctypes.windll.user32.PrintWindow", return_value=1),
        patch("win32gui.DeleteObject"),
        patch("win32gui.ReleaseDC"),
    ):
        assert os_control._capture_window_via_printwindow(123, 50, 50) is None


def test_capture_window_png_bytes_falls_back_to_mss_when_printwindow_yields_nothing() -> None:
    with (
        patch.object(os_control, "get_window_rect", return_value=(0, 0, 10, 10)),
        patch.object(os_control, "_capture_window_via_printwindow", return_value=None),
    ):
        import mss

        with patch.object(mss, "mss") as mock_mss:
            fake_shot = MagicMock(size=(10, 10), bgra=bytes(10 * 10 * 4))
            mock_mss.return_value.__enter__.return_value.grab.return_value = fake_shot
            result = os_control.capture_window_png_bytes(123)
    assert isinstance(result, bytes)
    assert len(result) > 0
