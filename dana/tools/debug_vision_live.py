"""Live vision diagnostic monitor for Donna's typed perception topics.

Polls ``memory/blackboard.db`` every 500ms and prints objects vs OCR age,
latency, sample text, and bounding-box count. Optional frame dump with boxes.

Examples::

    python -m dana.tools.debug_vision_live
    python -m dana.tools.debug_vision_live --save-frames
    python -m dana.tools.debug_vision_live --save-frames --fast
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dana.memory.blackboard import (
    BLACKBOARD_DB_PATH,
    PERCEPTION_OBJECTS_KEY,
    PERCEPTION_OCR_KEY,
    init_blackboard,
    publish_perception_frame_ref,
    read_perception_objects,
    read_perception_ocr,
    set_sensor_state,
)
from dana.operators.nav_and_click import parse_target_boxes

# Sensor key honored by vision_poller when present (seconds).
VISION_CAPTURE_INTERVAL_KEY = "vision_capture_interval_s"
POLL_INTERVAL_S = 0.5
STALE_WARN_S = 5.0
SAMPLE_CHARS = 120
DEBUG_OUTPUT_DIR = Path("debug_output")
LATEST_FRAME_PATH = DEBUG_OUTPUT_DIR / "latest_frame.png"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sample(text: str, n: int = SAMPLE_CHARS) -> str:
    body = " ".join((text or "").split())
    if len(body) <= n:
        return body
    return body[: max(0, n - 1)] + "…"


def query_perception(*, db_path: Path | str | None = None) -> dict[str, Any]:
    """Read typed objects + OCR topics (never conflate the two)."""
    path = init_blackboard(db_path or BLACKBOARD_DB_PATH)
    objects = read_perception_objects(db_path=path) or {}
    ocr = read_perception_ocr(db_path=path) or {}
    ocr_text = str(ocr.get("text") or "")
    boxes = parse_target_boxes(ocr_text) if ocr_text else []
    return {
        "db_path": str(path),
        "objects": objects,
        "ocr": ocr,
        "boxes": boxes,
        "box_count": len(boxes),
    }


def format_status_line(row: dict[str, Any]) -> str:
    obj = row.get("objects") or {}
    ocr = row.get("ocr") or {}
    obj_age = obj.get("age_seconds")
    ocr_age = ocr.get("age_seconds")
    obj_age_s = f"{float(obj_age):.1f}s" if obj_age is not None else "n/a"
    ocr_age_s = f"{float(ocr_age):.1f}s" if ocr_age is not None else "n/a"
    obj_meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    ocr_meta = ocr.get("meta") if isinstance(ocr.get("meta"), dict) else {}
    obj_lat = obj_meta.get("latency_ms")
    ocr_lat = ocr_meta.get("latency_ms")
    obj_lat_s = f"{float(obj_lat):.1f}ms" if obj_lat is not None else "n/a"
    ocr_lat_s = f"{float(ocr_lat):.1f}ms" if ocr_lat is not None else "n/a"
    return (
        f"[VISION LIVE] "
        f"objects_age={obj_age_s} objects_lat={obj_lat_s} "
        f"objects_chars={len(str(obj.get('text') or ''))} "
        f'objects_sample="{_sample(str(obj.get("text") or ""))}" | '
        f"ocr_age={ocr_age_s} ocr_lat={ocr_lat_s} "
        f"ocr_chars={len(str(ocr.get('text') or ''))} "
        f"boxes={int(row.get('box_count') or 0)} "
        f'ocr_sample="{_sample(str(ocr.get("text") or ""))}"'
    )


def request_fast_interval(*, db_path: Path | str | None = None) -> None:
    """Ask the Vision Poller to use a 1.0s capture interval temporarily."""
    set_sensor_state(
        VISION_CAPTURE_INTERVAL_KEY,
        "1.0",
        meta={"publisher": "debug_vision_live", "requested_s": 1.0},
        db_path=db_path,
    )
    _log("[VISION LIVE] --fast: requested 1.0s capture interval via Blackboard")


def capture_raw_primary_frame() -> tuple[Any, dict[str, int]] | tuple[None, None]:
    """Grab primary monitor at native resolution (BGR) for debug overlays."""
    try:
        import cv2
        import mss
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        _log(f"[VISION LIVE] capture deps missing: {exc}")
        return None, None
    try:
        with (getattr(mss, "MSS", None) or mss.mss)() as sct:
            if len(sct.monitors) < 2:
                return None, None
            mon = sct.monitors[1]
            shot = sct.grab(mon)
            frame = np.asarray(shot, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif not (frame.ndim == 3 and frame.shape[2] == 3):
                return None, None
            return frame, {
                "left": int(mon.get("left") or 0),
                "top": int(mon.get("top") or 0),
                "width": int(mon.get("width") or frame.shape[1]),
                "height": int(mon.get("height") or frame.shape[0]),
            }
    except Exception as exc:  # noqa: BLE001
        _log(f"[VISION LIVE] screen capture failed: {exc}")
        return None, None


def save_annotated_frame(row: dict[str, Any], *, dest: Path = LATEST_FRAME_PATH) -> bool:
    """Save raw primary frame with red bounding boxes over OCR regions."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        _log(f"[VISION LIVE] OpenCV missing: {exc}")
        return False

    frame, mon = capture_raw_primary_frame()
    if frame is None or mon is None:
        return False

    left = int(mon["left"])
    top = int(mon["top"])
    h, w = int(frame.shape[0]), int(frame.shape[1])
    boxes = list(row.get("boxes") or [])
    for box in boxes:
        x1 = int(getattr(box, "x1", 0) - left)
        y1 = int(getattr(box, "y1", 0) - top)
        x2 = int(getattr(box, "x2", 0) - left)
        y2 = int(getattr(box, "y2", 0) - top)
        if max(x1, x2) < 0 or max(y1, y2) < 0 or min(x1, x2) >= w or min(y1, y2) >= h:
            x1 = int(getattr(box, "x1", 0))
            y1 = int(getattr(box, "y1", 0))
            x2 = int(getattr(box, "x2", 0))
            y2 = int(getattr(box, "y2", 0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = str(getattr(box, "label", "") or "")[:40]
        if label:
            cv2.putText(
                frame,
                label,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # Banner: which topics are live.
    obj = row.get("objects") or {}
    ocr = row.get("ocr") or {}
    banner = (
        f"objects={'yes' if obj.get('text') else 'no'} "
        f"ocr={'yes' if ocr.get('text') else 'no'} "
        f"boxes={int(row.get('box_count') or 0)}"
    )
    cv2.putText(
        frame,
        banner,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(dest), frame))
    if ok:
        try:
            publish_perception_frame_ref(
                str(dest.resolve()),
                producer="debug_vision_live",
            )
        except Exception:  # noqa: BLE001
            pass
    return ok


def run_monitor(
    *,
    db_path: Path | str | None = None,
    save_frames: bool = False,
    fast: bool = False,
    poll_s: float = POLL_INTERVAL_S,
    max_ticks: int | None = None,
) -> int:
    path = init_blackboard(db_path or BLACKBOARD_DB_PATH)
    _log(
        f"[VISION LIVE] connected db={path} poll={poll_s * 1000:.0f}ms "
        f"topics={PERCEPTION_OBJECTS_KEY},{PERCEPTION_OCR_KEY}"
    )
    if fast:
        request_fast_interval(db_path=path)
    if save_frames:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _log(f"[VISION LIVE] --save-frames -> {LATEST_FRAME_PATH.resolve()}")

    ticks = 0
    while True:
        row = query_perception(db_path=path)
        obj_age = (row.get("objects") or {}).get("age_seconds")
        ocr_age = (row.get("ocr") or {}).get("age_seconds")
        for label, age in (("objects", obj_age), ("ocr", ocr_age)):
            if age is not None and float(age) > STALE_WARN_S:
                _log(
                    f"[VISION WARN] {label} age {float(age):.1f}s exceeds "
                    f"{STALE_WARN_S:.0f} seconds"
                )
        _log(format_status_line(row))
        if save_frames:
            try:
                if not save_annotated_frame(row):
                    _log("[VISION LIVE] save-frames failed (no capture)")
            except Exception as exc:  # noqa: BLE001
                _log(f"[VISION LIVE] save-frames error: {exc}")

        ticks += 1
        if max_ticks is not None and ticks >= int(max_ticks):
            return 0
        time.sleep(max(0.05, float(poll_s)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live Donna vision diagnostic (typed objects vs OCR)."
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Write debug_output/latest_frame.png with red OCR bounding boxes",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Request a temporary 1.0s Vision Poller capture interval",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="",
        help="Optional Blackboard DB path (default: workspace memory/blackboard.db)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Stop after N polls (0 = run forever)",
    )
    args = parser.parse_args(argv)
    db = Path(args.db).resolve() if str(args.db or "").strip() else None
    max_ticks = int(args.max_ticks) if int(args.max_ticks or 0) > 0 else None
    try:
        return run_monitor(
            db_path=db,
            save_frames=bool(args.save_frames),
            fast=bool(args.fast),
            max_ticks=max_ticks,
        )
    except KeyboardInterrupt:
        _log("[VISION LIVE] stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
