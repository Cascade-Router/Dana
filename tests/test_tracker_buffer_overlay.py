"""Unit tests for Tracker rolling buffer + ROI overlay helpers."""

from __future__ import annotations

import time

import numpy as np

from dana import tracker as tr
from dana.vision.overlay import RoiOverlay


def test_rolling_buffer_maxlen_and_cadence() -> None:
    tr.clear_frame_buffer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert tr.push_frame(frame, source="screen", force=True) is True
    assert tr.buffer_len() == 1
    assert tr.should_push_frame(interval_s=tr.FRAME_BUFFER_INTERVAL_S) is False
    assert tr.push_frame(frame, source="screen", force=False) is False
    for i in range(6):
        colored = np.full((480, 640, 3), i, dtype=np.uint8)
        assert tr.push_frame(colored, source="screen", force=True) is True
    assert tr.buffer_len() == 5
    ctx = tr.get_temporal_context()
    assert ctx["count"] == 5
    latest = tr.get_latest_buffered_frame()
    assert latest is not None
    assert latest.shape == (480, 640, 3)
    print("[PASS] rolling buffer maxlen=5 + cadence gate")


def test_map_box_to_screen() -> None:
    box = tr.map_box_to_screen(
        [160, 120, 320, 240],
        frame_wh=(640, 480),
        monitor={"left": 0, "top": 0, "width": 1920, "height": 1080},
    )
    assert box == (480, 270, 960, 540)
    print("[PASS] map_box_to_screen")


def test_roi_overlay_update_clear() -> None:
    ov = RoiOverlay()
    ov.start()
    assert ov._ready.wait(timeout=3.0)
    ov.update_roi((100, 100, 300, 250), "cup (test)")
    time.sleep(0.2)
    ov.clear_roi()
    time.sleep(0.1)
    ov.stop()
    print("[PASS] ROI overlay update/clear on dedicated thread")
