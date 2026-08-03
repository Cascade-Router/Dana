"""Spatial context aggregation for deictic / 'what am I looking at?' queries."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionRecord:
    label: str
    zone: str
    conf: float
    area: float
    cx: float
    cy: float
    ts: float = field(default_factory=time.monotonic)


class SpatialContextAggregator:
    """Unifies YOLO detections, vision source, and live transcript into a dense prompt block."""

    def __init__(self, ttl_sec: float = 2.5, max_objects: int = 8) -> None:
        self.ttl_sec = ttl_sec
        self.max_objects = max_objects
        self._lock = threading.RLock()
        self._detections: list[DetectionRecord] = []
        self._vision_source: str = "screen"
        self._transcript_user: str = ""
        self._transcript_assistant: str = ""
        self._ui_state: str = "idle"
        self._frame_wh: tuple[int, int] = (640, 480)

    def set_vision_source(self, source: str) -> None:
        with self._lock:
            self._vision_source = source

    def set_ui_state(self, state: str) -> None:
        with self._lock:
            self._ui_state = state

    def set_frame_size(self, width: int, height: int) -> None:
        with self._lock:
            self._frame_wh = (max(1, width), max(1, height))

    def update_transcript(self, *, user: str | None = None, assistant: str | None = None) -> None:
        with self._lock:
            if user is not None:
                self._transcript_user = user.strip()[:240]
            if assistant is not None:
                self._transcript_assistant = assistant.strip()[:240]

    def update_from_dets(
        self,
        dets: list[tuple[Any, str, float]],
        frame_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Ingest tracker dets: (xyxy, 'label (zone)', conf)."""
        now = time.monotonic()
        width, height = self._frame_wh
        if frame_shape is not None and len(frame_shape) >= 2:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            self._frame_wh = (width, height)

        records: list[DetectionRecord] = []
        frame_area = float(max(1, width * height))
        for xyxy, spatial_label, conf in dets:
            try:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
            except Exception:
                continue
            area = max(0.0, (x2 - x1) * (y2 - y1))
            cx = (x1 + x2) * 0.5 / float(width)
            cy = (y1 + y2) * 0.5 / float(height)
            label = spatial_label
            zone = "center"
            if "(" in spatial_label and spatial_label.endswith(")"):
                base, _, rest = spatial_label.partition("(")
                label = base.strip()
                zone = rest[:-1].strip() or "center"
            records.append(
                DetectionRecord(
                    label=label,
                    zone=zone,
                    conf=float(conf),
                    area=area / frame_area,
                    cx=cx,
                    cy=cy,
                    ts=now,
                )
            )
        with self._lock:
            # Merge with TTL memory (keep recent if flicker).
            kept = [r for r in self._detections if (now - r.ts) <= self.ttl_sec]
            by_key = {(r.label, r.zone): r for r in kept}
            for r in records:
                by_key[(r.label, r.zone)] = r
            merged = sorted(
                by_key.values(),
                key=lambda r: r.area * 0.65 + r.conf * 0.35,
                reverse=True,
            )
            self._detections = merged[: self.max_objects]

    def remember_labels(self, labels: list[str]) -> None:
        """Fallback when only label strings are available (no boxes)."""
        now = time.monotonic()
        with self._lock:
            for spatial_label in labels:
                label = spatial_label
                zone = "center"
                if "(" in spatial_label and spatial_label.endswith(")"):
                    base, _, rest = spatial_label.partition("(")
                    label = base.strip()
                    zone = rest[:-1].strip() or "center"
                self._detections.append(
                    DetectionRecord(
                        label=label,
                        zone=zone,
                        conf=0.5,
                        area=0.05,
                        cx=0.5,
                        cy=0.5,
                        ts=now,
                    )
                )
            self._detections = self._detections[-self.max_objects :]

    def state_vector(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            live = [r for r in self._detections if (now - r.ts) <= self.ttl_sec]
            dominant = live[0] if live else None
            return {
                "vision_source": self._vision_source,
                "ui_state": self._ui_state,
                "object_count": len(live),
                "dominant": None
                if dominant is None
                else {
                    "label": dominant.label,
                    "zone": dominant.zone,
                    "conf": round(dominant.conf, 2),
                    "area": round(dominant.area, 3),
                },
                "objects": [
                    {
                        "l": r.label,
                        "z": r.zone,
                        "c": round(r.conf, 2),
                        "a": round(r.area, 3),
                        "xy": [round(r.cx, 2), round(r.cy, 2)],
                    }
                    for r in live
                ],
                "user_intent": self._transcript_user,
            }

    def synthesize_prompt_block(self) -> str:
        """Token-efficient spatial block for the LLM system prompt."""
        vec = self.state_vector()
        objs = vec["objects"]
        if not objs:
            scene = "none"
        else:
            # Prioritize dominance + proximity to center for deictics ("this"/"that").
            parts = []
            for o in objs:
                dist = abs(o["xy"][0] - 0.5) + abs(o["xy"][1] - 0.5)
                parts.append(f"{o['l']}@{o['z']}(a={o['a']},d={dist:.2f})")
            scene = "; ".join(parts)
        dom = vec["dominant"]
        dom_s = "none" if not dom else f"{dom['label']}@{dom['zone']}"
        intent = vec["user_intent"] or "-"
        return (
            f"vis={vec['vision_source']}|ui={vec['ui_state']}|"
            f"dom={dom_s}|scene=[{scene}]|intent={intent}"
        )

    def label_list(self) -> list[str]:
        vec = self.state_vector()
        return [f"{o['l']} ({o['z']})" for o in vec["objects"]]


# Process-wide aggregator used by tracker + conversation.
SPATIAL_AGGREGATOR = SpatialContextAggregator()
