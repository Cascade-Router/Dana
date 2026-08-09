"""Stage 6.3 — Navigation Operator (closed-loop mouse servo).

Locates a UI target box from typed Blackboard ``perception.ocr`` (Florence),
moves the cursor along a stochastic cubic Bezier path (easeInOutQuad velocity),
and left-clicks via SendInput after a human pause — with Sense-Evaluate-Act
repathing if the target drifts or disappears.
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Label heuristics for slide/web evaluation text fields + generic targets.
_TARGET_HINTS = (
    "enter comments",
    "comments",
    "comment box",
    "evaluation",
    "text field",
    "input field",
    "textarea",
    "target",
)

# Florence / OCR style: Label [x1, y1, x2, y2] — label is the nearest short token(s).
_BBOX_ONLY_RE = re.compile(
    r"\[(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*"
    r"(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\]"
)
_BBOX_PAREN_RE = re.compile(
    r"\((?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*"
    r"(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\)"
)
_BBOX_AT_RE = re.compile(
    r"(?is)(?P<label>[^\n@]{1,80}?)\s*@\s*bbox\s*=\s*"
    r"(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*"
    r"(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)"
)


def _label_before(text: str, end_idx: int) -> str:
    """Grab up to ~4 trailing words before a bbox at ``end_idx``."""
    prefix = text[:end_idx].rstrip(" -:|.,;")
    m = re.search(r"(?i)labeled\s+([A-Za-z][\w-]{0,40})\s*$", prefix)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,2})\s*$",
        prefix,
    )
    if m:
        return m.group(1).strip()
    tokens = re.findall(r"[A-Za-z0-9]+", prefix)
    return " ".join(tokens[-3:]) if tokens else "target"


# Must be defined before parse_target_boxes uses it at runtime.
@dataclass(frozen=True)
class TargetBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, int(self.x2) - int(self.x1))

    @property
    def height(self) -> int:
        return max(1, int(self.y2) - int(self.y1))

    def contains(self, x: float, y: float, *, pad: int = 2) -> bool:
        return (
            self.x1 - pad <= x <= self.x2 + pad
            and self.y1 - pad <= y <= self.y2 + pad
        )

    def inner_random_point(self, *, inner_frac: float = 0.60) -> tuple[int, int]:
        """Sample a click point inside the inner ``inner_frac`` of the box."""
        frac = max(0.1, min(1.0, float(inner_frac)))
        mx = (1.0 - frac) / 2.0
        x = self.x1 + self.width * (mx + random.random() * frac)
        y = self.y1 + self.height * (mx + random.random() * frac)
        return int(round(x)), int(round(y))


def parse_target_boxes(visual_context: str) -> list[TargetBox]:
    """Extract labeled bounding boxes from Florence/OCR blackboard text."""
    text = visual_context or ""
    found: list[TargetBox] = []

    def _add(label: str, x1: int, y1: int, x2: int, y2: int) -> None:
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        found.append(
            TargetBox(label=(label or "").strip() or "target", x1=x1, y1=y1, x2=x2, y2=y2)
        )

    for rx in (_BBOX_ONLY_RE, _BBOX_PAREN_RE):
        for m in rx.finditer(text):
            _add(
                _label_before(text, m.start()),
                int(m.group("x1")),
                int(m.group("y1")),
                int(m.group("x2")),
                int(m.group("y2")),
            )
    for m in _BBOX_AT_RE.finditer(text):
        _add(
            str(m.group("label") or "").strip(),
            int(m.group("x1")),
            int(m.group("y1")),
            int(m.group("x2")),
            int(m.group("y2")),
        )
    return found


def _dry_run() -> bool:
    return os.environ.get("DANA_OS_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def find_target_box(
    visual_context: str,
    query: str,
) -> TargetBox | None:
    """Pick the best TargetBox for ``query`` (e.g. ``Target``, ``Enter Comments``)."""
    boxes = parse_target_boxes(visual_context)
    q = (query or "").strip().lower()
    if not boxes:
        return None
    if not q:
        return boxes[0]

    def _score(box: TargetBox) -> float:
        lab = (box.label or "").lower()
        if q == lab or q in lab or lab in q:
            return 100.0 + len(lab)
        # Heuristic boost for evaluation-field signatures.
        hint_hit = any(h in lab for h in _TARGET_HINTS if h in q or q in h or h in lab)
        if hint_hit and (q in " ".join(_TARGET_HINTS) or any(h in q for h in _TARGET_HINTS)):
            return 50.0
        if any(h in lab for h in _TARGET_HINTS) and any(h in q for h in _TARGET_HINTS):
            return 40.0
        # Token overlap
        qt = set(re.findall(r"[a-z0-9]+", q))
        lt = set(re.findall(r"[a-z0-9]+", lab))
        if not qt or not lt:
            return 0.0
        return 10.0 * len(qt & lt) / max(1, len(qt))

    ranked = sorted(boxes, key=_score, reverse=True)
    best = ranked[0]
    if _score(best) <= 0 and q not in (best.label or "").lower():
        # Fall back: exact substring search in full visual for query near a bbox.
        for box in boxes:
            if q in (box.label or "").lower():
                return box
        return None
    return best


def ease_in_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = (
        (u**3) * p0[0]
        + 3 * (u**2) * t * p1[0]
        + 3 * u * (t**2) * p2[0]
        + (t**3) * p3[0]
    )
    y = (
        (u**3) * p0[1]
        + 3 * (u**2) * t * p1[1]
        + 3 * u * (t**2) * p2[1]
        + (t**3) * p3[1]
    )
    return x, y


def generate_bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    steps: int = 36,
) -> list[tuple[int, int]]:
    """Cubic Bezier with randomized control points + easeInOutQuad sampling."""
    x0, y0 = float(start[0]), float(start[1])
    x3, y3 = float(end[0]), float(end[1])
    dx, dy = x3 - x0, y3 - y0
    dist = math.hypot(dx, dy) or 1.0
    # Perpendicular unit for stochastic "shake".
    px, py = -dy / dist, dx / dist
    shake = max(8.0, min(80.0, dist * 0.18))
    # Control points along the segment with lateral noise (never identical).
    c1t, c2t = random.uniform(0.2, 0.4), random.uniform(0.6, 0.85)
    p1 = (
        x0 + dx * c1t + px * random.uniform(-shake, shake),
        y0 + dy * c1t + py * random.uniform(-shake, shake),
    )
    p2 = (
        x0 + dx * c2t + px * random.uniform(-shake, shake),
        y0 + dy * c2t + py * random.uniform(-shake, shake),
    )
    n = max(8, int(steps))
    pts: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        t = ease_in_out_quad(i / n)
        x, y = cubic_bezier((x0, y0), p1, p2, (x3, y3), t)
        pts.append((int(round(x)), int(round(y))))
    return pts


@dataclass
class NavigationOperator:
    """Closed-loop navigate-and-click servo."""

    read_visual: Callable[[], str] | None = None
    get_cursor: Callable[[], tuple[int, int]] | None = None
    move_cursor: Callable[[int, int], None] | None = None
    click: Callable[[], None] | None = None
    chunk_size: int = 10
    events: list[dict[str, Any]] = field(default_factory=list)
    _virt_cursor: tuple[int, int] = (0, 0)

    def _sense_visual(self) -> str:
        if self.read_visual is not None:
            return str(self.read_visual() or "")
        try:
            from dana.memory import read_perception_ocr_text

            text = read_perception_ocr_text() or ""
            if not text.strip():
                return (
                    "(no OCR boxes: perception.ocr missing or schema mismatch — "
                    "run ocr_with_region before navigate_and_click)"
                )
            return text
        except Exception as exc:  # noqa: BLE001
            return f"(sense_error: {exc})"

    def _cursor(self) -> tuple[int, int]:
        if self.get_cursor is not None:
            return self.get_cursor()
        if _dry_run() and self.move_cursor is None:
            return self._virt_cursor
        if _dry_run():
            return self._virt_cursor
        from dana.tools.os_control import get_cursor_pos

        return get_cursor_pos()

    def _move(self, x: int, y: int) -> None:
        self._virt_cursor = (int(x), int(y))
        if self.move_cursor is not None:
            self.move_cursor(int(x), int(y))
            return
        if _dry_run():
            return
        from dana.tools.os_control import move_cursor_absolute

        move_cursor_absolute(int(x), int(y))

    def _click(self) -> None:
        if self.click is not None:
            self.click()
            return
        if _dry_run():
            return
        from dana.tools.os_control import click_left_sendinput

        click_left_sendinput()

    def navigate_and_click(
        self,
        query: str,
        *,
        visual_context: str | None = None,
        max_loops: int = 8,
    ) -> dict[str, Any]:
        """SEA servo: Bezier move → sense → evaluate → click."""
        self.events.clear()
        visual = visual_context if visual_context is not None else self._sense_visual()
        if "(no OCR boxes:" in (visual or ""):
            return {
                "ok": False,
                "error": (
                    "no OCR boxes: perception.ocr missing or schema mismatch — "
                    "run ocr_with_region before navigate_and_click"
                ),
                "engine": "navigation_operator",
            }
        target = find_target_box(visual, query)
        if target is None:
            # Fail closed: YOLO object prose never yields Florence boxes.
            if (visual or "").lstrip().startswith("[Vision Output]"):
                return {
                    "ok": False,
                    "error": (
                        "no OCR boxes: got YOLO objects prose instead of "
                        "perception.ocr — run ocr_with_region first"
                    ),
                    "engine": "navigation_operator",
                }
            return {
                "ok": False,
                "error": f"target box not found for query={query!r}",
                "engine": "navigation_operator",
            }

        if self.get_cursor is None and (_dry_run() or self.move_cursor is None):
            # Seed virtual cursor away from target for dry-run path demos.
            self._virt_cursor = (
                max(0, target.x1 - 120),
                max(0, target.y1 - 80),
            )
        start = self._cursor()
        end = target.inner_random_point(inner_frac=0.60)
        path = generate_bezier_path(start, end)
        self.events.append(
            {
                "event": "plan",
                "start": start,
                "end": end,
                "target": target.__dict__,
                "path_len": len(path),
            }
        )

        cursor = start
        loops = 0
        idx = 0
        target_now = target
        arrived = False
        while loops < max_loops:
            loops += 1
            # Stage 7.2 — hardware kill switch aborts mid-SEA.
            try:
                from dana.middleware.kill_switch import halt_if_requested

                if halt_if_requested():
                    self.events.append({"event": "halt", "loop": loops})
                    return {
                        "ok": False,
                        "halted": True,
                        "error": "halted by GLOBAL_HALT_EVENT",
                        "engine": "navigation_operator",
                        "cursor": cursor,
                        "events": list(self.events),
                    }
            except Exception:  # noqa: BLE001
                pass

            # Stage 7.4 — yield to physical human input (soft pause).
            try:
                from dana.middleware.human_yield import yield_check

                yield_check(operator="navigation_operator")
            except Exception:  # noqa: BLE001
                pass

            # Act — advance a chunk of the path.
            chunk_end = min(len(path), idx + max(1, int(self.chunk_size)))
            for pt in path[idx:chunk_end]:
                try:
                    from dana.middleware.kill_switch import halt_if_requested

                    if halt_if_requested():
                        self.events.append({"event": "halt", "loop": loops, "cursor": pt})
                        return {
                            "ok": False,
                            "halted": True,
                            "error": "halted by GLOBAL_HALT_EVENT",
                            "engine": "navigation_operator",
                            "cursor": pt,
                            "events": list(self.events),
                        }
                except Exception:  # noqa: BLE001
                    pass
                try:
                    from dana.middleware.human_yield import yield_check

                    yield_check(operator="navigation_operator")
                except Exception:  # noqa: BLE001
                    pass
                self._move(pt[0], pt[1])
                # Velocity already eased; small dwell keeps motion visible/human.
                time.sleep(random.uniform(0.008, 0.022))
                cursor = pt
            idx = chunk_end

            # Sense
            time.sleep(random.uniform(0.02, 0.05))
            if self.get_cursor is not None or not _dry_run():
                cursor = self._cursor()
            visual_now = self._sense_visual()
            sensed = find_target_box(visual_now, query)
            if sensed is not None:
                drifted = (
                    abs(sensed.x1 - target_now.x1) > 8
                    or abs(sensed.y1 - target_now.y1) > 8
                    or abs(sensed.x2 - target_now.x2) > 8
                    or abs(sensed.y2 - target_now.y2) > 8
                )
                target_now = sensed
            elif find_target_box(visual, query) is not None:
                # Poller stale / empty — keep planned target from initial sense.
                drifted = False
                target_now = target
            else:
                self.events.append(
                    {
                        "event": "abort",
                        "reason": "target_disappeared",
                        "visual_preview": (visual_now or "")[:120],
                    }
                )
                return {
                    "ok": False,
                    "error": "target disappeared from visual context",
                    "engine": "navigation_operator",
                    "events": list(self.events),
                }

            arrived = target_now.contains(cursor[0], cursor[1], pad=6)
            self.events.append(
                {
                    "event": "chunk",
                    "loop": loops,
                    "cursor": cursor,
                    "arrived": arrived,
                    "path_idx": idx,
                }
            )
            if arrived:
                break

            # Repath only when the box drifted or the planned path is exhausted.
            if drifted or idx >= len(path):
                end = target_now.inner_random_point(inner_frac=0.60)
                path = generate_bezier_path(cursor, end)
                idx = 0
                self.events.append(
                    {
                        "event": "repath",
                        "end": end,
                        "path_len": len(path),
                        "reason": "drift" if drifted else "path_exhausted",
                    }
                )
                continue
            # Otherwise keep following the current Bezier chunk-by-chunk.

        if not arrived:
            return {
                "ok": False,
                "error": "cursor never arrived inside target box",
                "engine": "navigation_operator",
                "cursor": cursor,
                "target": target_now.__dict__,
                "events": list(self.events),
            }

        # Human pause then click.
        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt", "phase": "pre_click"})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                    "engine": "navigation_operator",
                    "cursor": cursor,
                    "events": list(self.events),
                }
        except Exception:  # noqa: BLE001
            pass
        pause_s = random.uniform(0.150, 0.400)
        time.sleep(pause_s)
        self._click()
        self.events.append({"event": "click", "pause_s": pause_s, "cursor": cursor})

        try:
            from dana.telemetry import log_operator_nav_click_complete

            log_operator_nav_click_complete(
                query=query,
                payload={
                    "cursor": cursor,
                    "target": target.__dict__,
                    "loops": loops,
                    "dry_run": _dry_run(),
                },
            )
        except Exception:  # noqa: BLE001
            pass

        return {
            "ok": True,
            "query": query,
            "cursor": cursor,
            "target": target.__dict__,
            "loops": loops,
            "engine": "navigation_operator",
            "dry_run": _dry_run(),
            "events": list(self.events),
        }


def navigate_and_click(query: str, *, visual_context: str | None = None) -> str:
    """Tool / actuator entry for NavigationOperator."""
    op = NavigationOperator()
    result = op.navigate_and_click(query or "Target", visual_context=visual_context)
    if result.get("halted"):
        return f"HALTED: navigate_and_click — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: navigate_and_click failed: {result.get('error')}"
    tgt = result.get("target") or {}
    return (
        f"OK: navigate_and_click query={query!r} "
        f"clicked=({result.get('cursor')}) "
        f"target=[{tgt.get('x1')},{tgt.get('y1')},{tgt.get('x2')},{tgt.get('y2')}] "
        f"loops={result.get('loops')} dry_run={result.get('dry_run')}"
    )
