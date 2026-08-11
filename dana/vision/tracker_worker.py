"""YOLO vision tracker thread + spatial-memory helpers.

Extracted verbatim from ``dana.core_agent`` (Phase 7 of the core_agent.py
decomposition; see docs/architecture/phase7_core_agent_decomposition.md).
``tracker_worker`` is one of the four daemon threads ``dana.core.app_runtime
.agent_loop()`` spawns; the rest of this module is helpers it (and
``dana.core.agent_loop``'s ``conversation_worker``) use to parse YOLO
detections into spatial-anchor labels ("bottle (top-left)") and remember
recently-seen labels for a few seconds after they leave frame.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from spatial_context import SPATIAL_AGGREGATOR

import dana.core.shared_state as state
from dana.agentic import get_dana_mode
from dana.core.constants import (
    FRAME_SIZE,
    TRACKER_BUFFER_INTERVAL_S,
    TRACKER_SLEEP_SEC,
    YOLO_CONF,
    YOLO_WEIGHTS,
)
from dana.core.shared_state import (
    SPATIAL_MEMORY_SEC,
    active_vision_lock,
    camera_tool,
    latest_dets_lock,
    latest_frame_lock,
    spatial_memory,
    spatial_memory_lock,
    stop_event,
)
from dana.logging import log, log_debug
from dana.paths import _nt_hide_console_if_mp_child

def yolo_device_arg(device) -> str | int:
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return "mps"
    return "cpu"
def spatial_zone(cx: float, cy: float, frame_w: int = FRAME_SIZE[0], frame_h: int = FRAME_SIZE[1]) -> str:
    """Map a point to a 3x3 spatial label for 640x480 (or given) frames."""
    # X-axis: Left (< 213), Center (213-426), Right (> 426) on 640-wide frames.
    x_left = frame_w / 3.0
    x_right = 2.0 * frame_w / 3.0
    # Y-axis: Top (< 160), Center (160-320), Bottom (> 320) on 480-tall frames.
    y_top = frame_h / 3.0
    y_bottom = 2.0 * frame_h / 3.0

    if cx < x_left:
        x_pos = "left"
    elif cx > x_right:
        x_pos = "right"
    else:
        x_pos = "center"

    if cy < y_top:
        y_pos = "top"
    elif cy > y_bottom:
        y_pos = "bottom"
    else:
        y_pos = "center"

    if x_pos == "center" and y_pos == "center":
        return "center"
    if x_pos == "center":
        return y_pos
    if y_pos == "center":
        return x_pos
    return f"{y_pos}-{x_pos}"
def parse_yolo_results(results: Any) -> tuple[list[str], list[tuple[np.ndarray, str, float]]]:
    """Return spatial labels like 'bottle (top-left)' plus drawable detections."""
    labels: list[str] = []
    dets: list[tuple[np.ndarray, str, float]] = []
    if not results:
        return labels, dets

    result = results[0]
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return labels, dets

    # Prefer actual frame size from the result if available.
    frame_w, frame_h = FRAME_SIZE
    try:
        shape = getattr(result, "orig_shape", None)
        if shape is not None and len(shape) >= 2:
            frame_h, frame_w = int(shape[0]), int(shape[1])
    except Exception:
        pass

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        name = str(names.get(cls_id, cls_id))
        xyxy = box.xyxy[0].detach().cpu().numpy()
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        zone = spatial_zone(cx, cy, frame_w, frame_h)
        spatial_label = f"{name} ({zone})"
        labels.append(spatial_label)
        dets.append((xyxy, spatial_label, conf))
    return labels, dets
def remember_spatial_labels(labels: list[str]) -> None:
    now = time.monotonic()
    with spatial_memory_lock:
        for label in labels:
            spatial_memory[label] = now
        stale = [k for k, ts in spatial_memory.items() if now - ts > SPATIAL_MEMORY_SEC]
        for key in stale:
            del spatial_memory[key]
def get_spatial_memory_labels() -> list[str]:
    now = time.monotonic()
    with spatial_memory_lock:
        alive = [(label, ts) for label, ts in spatial_memory.items() if now - ts <= SPATIAL_MEMORY_SEC]
        alive.sort(key=lambda row: row[1], reverse=True)
        return [label for label, _ts in alive]
def format_class_list(labels: list[str] | set[str]) -> str:
    """Join spatial anchors; keep same-class objects in different zones."""
    if isinstance(labels, set):
        items = sorted(labels)
    else:
        # Preserve order; dedupe identical full labels only.
        items = list(dict.fromkeys(labels))
    return ", ".join(items) if items else "none detected"
def format_vision_context_for_llm(labels: list[str] | set[str] | str | None) -> str:
    """Natural Visual Context sentence for ReAct injection (empty if none)."""
    from dana.prompts.spatial_synthesis import format_vision_context

    return format_vision_context(labels)
def tracker_worker(device) -> None:
    _nt_hide_console_if_mp_child()

    from dana.tracker import (
        FRAME_BUFFER_INTERVAL_S,
        get_yolo_model,
        map_box_to_screen,
        primary_monitor_geometry,
        push_frame,
        should_push_frame,
        seconds_since_last_push,
        yolo_is_loaded,
    )

    log(
        "Tracker",
        f"Idle (JIT YOLO) — will load {YOLO_WEIGHTS} on Vision mode or first detect.",
    )
    yolo_dev = yolo_device_arg(device)
    buf_interval = float(TRACKER_BUFFER_INTERVAL_S or FRAME_BUFFER_INTERVAL_S)
    log(
        "Tracker",
        f"Rolling buffer every {buf_interval:.1f}s from "
        f"active_vision_tool.get_frame() (maxlen=60 thumbnails).",
    )

    frames = 0
    while not stop_event.is_set():
        # Wait out the ~1s cadence before grabbing (avoids busy mss polling).
        if not should_push_frame(interval_s=buf_interval):
            rem = buf_interval - seconds_since_last_push()
            if rem == float("inf"):
                rem = 0.0
            time.sleep(max(0.05, min(TRACKER_SLEEP_SEC, max(0.0, rem))))
            continue

        with active_vision_lock:
            tool = state.active_vision_tool
        tool_name = "camera" if tool is camera_tool else "screen"

        try:
            frame = tool.get_frame()
        except Exception as exc:  # noqa: BLE001
            log("Tracker", f"WARNING: {tool_name} get_frame failed ({exc})")
            time.sleep(0.05)
            continue

        if frame is None:
            time.sleep(0.05)
            continue

        with latest_frame_lock:
            state.latest_frame = frame

        try:
            mode = get_dana_mode()
        except Exception:  # noqa: BLE001
            mode = "chat"

        run_yolo = mode == "vision" or yolo_is_loaded()
        monitor = primary_monitor_geometry() if tool_name == "screen" else None

        dets: list = []
        labels: list[str] = []
        if run_yolo:
            try:
                yolo = get_yolo_model(YOLO_WEIGHTS)
                results = yolo.predict(
                    source=frame,
                    conf=YOLO_CONF,
                    device=yolo_dev,
                    verbose=False,
                )
                _, dets = parse_yolo_results(results)
            except Exception as exc:  # noqa: BLE001
                log("Tracker", f"WARNING: YOLO predict failed: {exc}")
                dets = []

            with latest_dets_lock:
                state.latest_dets = dets

            labels = [name for _, name, _ in dets]
            remember_spatial_labels(labels)
            SPATIAL_AGGREGATOR.set_vision_source(tool_name)
            SPATIAL_AGGREGATOR.update_from_dets(
                dets, frame_shape=getattr(frame, "shape", None)
            )

            if dets and tool_name == "screen":
                try:
                    from dana.vision.overlay import update_roi

                    best = max(dets, key=lambda d: float(d[2]))
                    xyxy, label, _conf = best
                    shape = getattr(frame, "shape", None)
                    fw = (
                        int(shape[1])
                        if shape is not None and len(shape) >= 2
                        else FRAME_SIZE[0]
                    )
                    fh = (
                        int(shape[0])
                        if shape is not None and len(shape) >= 2
                        else FRAME_SIZE[1]
                    )
                    screen_box = map_box_to_screen(
                        xyxy, frame_wh=(fw, fh), monitor=monitor
                    )
                    if screen_box is not None:
                        update_roi(screen_box, label)
                except Exception as exc:  # noqa: BLE001
                    log_debug("Tracker", f"ROI overlay update skipped ({exc})")

        push_frame(
            frame,
            source=tool_name,
            dets=list(dets),
            monitor=monitor,
            force=True,
        )

        # Phase 3 — paced screen_history extraction (~12s; OCR, not every frame).
        if tool_name == "screen":
            try:
                from dana.tools.vision import maybe_extract_screen_history

                maybe_extract_screen_history()
            except Exception:  # noqa: BLE001
                pass

        frames += 1
        if frames % 15 == 0:
            from dana.tracker import buffer_len

            log_debug(
                "Tracker",
                f"Alive - {frames} samples via {tool_name}; "
                f"buffer={buffer_len()}/60; last=[{format_class_list(labels)}]",
            )

        time.sleep(TRACKER_SLEEP_SEC)

    log("Tracker", "Stopped.")
    try:
        from dana.vision.overlay import clear_roi

        clear_roi()
    except Exception:  # noqa: BLE001
        pass
