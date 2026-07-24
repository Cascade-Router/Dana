"""Stage 4.1 — Asynchronous vision sensor daemon (blackboard publisher).

Standalone process: capture screen → YOLO extract → upsert
``latest_visual_context`` on the SQLite Blackboard, then sleep.

Run:
    python -m donna.middleware.vision_poller
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np
import psutil

from donna.middleware.resource_cap import (
    apply_cpu_half_affinity,
    apply_torch_vram_half_cap,
)

# Stage 7.1 — pin this daemon to the first 50% of logical cores.
try:
    apply_cpu_half_affinity()
except Exception:  # noqa: BLE001
    pass

# Stage 4.4 QoS — yield CPU to foreground Chat/Audio and user apps.
try:
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:  # noqa: BLE001
    pass

# Stage 4.4 / 7.1 — cap PyTorch CPU threads + hard 50% VRAM fraction.
try:
    import torch

    torch.set_num_threads(2)
    apply_torch_vram_half_cap(0)
except Exception:  # noqa: BLE001
    pass

from donna.memory.blackboard import (
    LATEST_VISUAL_CONTEXT_KEY,
    set_sensor_state,
)
from donna.telemetry import log_sensor_vision
from donna.vision_tools import analyze_visual_context, capture_screen_frame

DEFAULT_INTERVAL_S = 30.0
DEFAULT_PIXEL_DIFF_THRESHOLD = 0.02
_PREV_GRAY: np.ndarray | None = None


def _significant_change(
    frame: np.ndarray,
    *,
    threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
) -> bool:
    """Return True if frame differs enough from the previous sample to warrant YOLO."""
    global _PREV_GRAY
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        return True
    if _PREV_GRAY is None or _PREV_GRAY.shape != gray.shape:
        _PREV_GRAY = gray
        return True
    diff = float(np.mean(cv2.absdiff(_PREV_GRAY, gray))) / 255.0
    _PREV_GRAY = gray
    return diff >= float(threshold)


def extract_visual_context(*, source: str = "screen") -> str:
    """Run existing visual extraction (YOLO via ``analyze_visual_context``)."""
    kind = str(source or "screen").strip().lower() or "screen"
    if kind in {"webcam", "camera", "video"}:
        return analyze_visual_context(source="webcam")
    return analyze_visual_context(source="screen")


def publish_visual_context(
    semantic_text: str,
    *,
    latency_ms: float,
    skipped: bool = False,
) -> None:
    """Write/overwrite ``latest_visual_context`` and emit ``[SENSOR_VISION]``."""
    meta: dict[str, Any] = {
        "latency_ms": float(latency_ms),
        "skipped_inference": bool(skipped),
        "publisher": "vision_poller",
    }
    set_sensor_state(LATEST_VISUAL_CONTEXT_KEY, semantic_text or "", meta=meta)
    log_sensor_vision(
        f"latest_visual_context updated chars={len(semantic_text or '')}",
        latency_ms=float(latency_ms),
        payload={
            "key": LATEST_VISUAL_CONTEXT_KEY,
            "chars": len(semantic_text or ""),
            "skipped_inference": bool(skipped),
        },
    )


def poll_once(
    *,
    source: str = "screen",
    pixel_diff_threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
    force: bool = False,
) -> dict[str, Any]:
    """One capture → optional YOLO → blackboard publish. Returns loop stats."""
    t0 = time.perf_counter()
    frame = None
    if str(source or "screen").strip().lower() in {"webcam", "camera", "video"}:
        from donna.vision_tools import capture_webcam_frame

        frame = capture_webcam_frame()
    else:
        frame = capture_screen_frame()

    if frame is None:
        text = f"[Vision Output] Detected: nothing (no {source} frame)."
        latency_ms = (time.perf_counter() - t0) * 1000.0
        publish_visual_context(text, latency_ms=latency_ms, skipped=True)
        return {"ok": False, "latency_ms": latency_ms, "skipped": True, "text": text}

    if not force and not _significant_change(
        frame, threshold=pixel_diff_threshold
    ):
        # Screen quiet — still refresh the key lightly so Chat sees freshness,
        # but skip GPU inference. Re-read prior value if we have one.
        from donna.memory.blackboard import read_visual_state

        prior = read_visual_state()
        text = prior or "[Vision Output] Detected: (unchanged; inference skipped)."
        latency_ms = (time.perf_counter() - t0) * 1000.0
        publish_visual_context(text, latency_ms=latency_ms, skipped=True)
        return {"ok": True, "latency_ms": latency_ms, "skipped": True, "text": text}

    text = extract_visual_context(source=source)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    publish_visual_context(text, latency_ms=latency_ms, skipped=False)
    return {"ok": True, "latency_ms": latency_ms, "skipped": False, "text": text}


def run_forever(
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    source: str = "screen",
    pixel_diff_threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
) -> None:
    """Robust ``while True`` publisher loop with exception isolation."""
    interval = max(1.0, float(interval_s))
    print(
        f"[vision_poller] starting interval={interval}s source={source} "
        f"pixel_diff={pixel_diff_threshold}",
        flush=True,
    )
    while True:
        try:
            stats = poll_once(
                source=source,
                pixel_diff_threshold=pixel_diff_threshold,
            )
            print(
                f"[vision_poller] published skipped={stats.get('skipped')} "
                f"latency_ms={stats.get('latency_ms'):.1f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[vision_poller] ERROR: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Donna Stage 4.1 vision sensor daemon (blackboard publisher)."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"Seconds between captures (default {DEFAULT_INTERVAL_S})",
    )
    parser.add_argument(
        "--source",
        choices=("screen", "webcam"),
        default="screen",
        help="Capture source",
    )
    parser.add_argument(
        "--pixel-diff",
        type=float,
        default=DEFAULT_PIXEL_DIFF_THRESHOLD,
        help="Mean abs-diff threshold (0–1) to skip YOLO when screen is quiet",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll then exit (for tests/smoke)",
    )
    args = parser.parse_args(argv)
    if args.once:
        stats = poll_once(
            source=args.source,
            pixel_diff_threshold=float(args.pixel_diff),
            force=True,
        )
        print(stats.get("text", ""), flush=True)
        return 0 if stats.get("ok") else 1
    run_forever(
        interval_s=float(args.interval),
        source=args.source,
        pixel_diff_threshold=float(args.pixel_diff),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
