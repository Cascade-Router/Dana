"""Florence-2 spatial macro recorder and semantic replay engine.

Reuses ``dana.vision.florence_engine`` / Tracker capture helpers. All OS
side-effects (screenshot, click, type, hotkey) and Florence grounding are
injectable so tests run offline with no GPU and no real clicks.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from dana.macros.schema import MacroSequence, MacroStep

_log = logging.getLogger("dana.macros.engine")

# (screenshot, phrase_prompt) -> grounding dict
GroundingFn = Callable[[Any, str], dict[str, Any]]
ScreenshotFn = Callable[[], Any]
ClickFn = Callable[[int, int], None]
TypeFn = Callable[[str], None]
HotkeyFn = Callable[[str], None]
DoubleClickFn = Callable[[int, int], None]

_ALLOWED_ACTIONS = frozenset(
    {"click", "double_click", "type_text", "key_combination"}
)


def sanitize_macro_id(macro_id: str) -> str:
    """Filesystem-safe macro id (alnum / underscore / hyphen)."""
    raw = (macro_id or "").strip()
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE)
    safe = re.sub(r"_+", "_", safe).strip("._")
    return (safe or "macro")[:128]


def default_macros_dir() -> Path:
    """Package-local macro JSON directory (``dana/macros/``)."""
    return Path(__file__).resolve().parent


def bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[int, int]:
    """Translate axis-aligned bbox → integer click coordinates (center)."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def _as_bgr_frame(screenshot: Any) -> Any:
    """Best-effort convert screenshot input to a BGR numpy array."""
    if screenshot is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return screenshot

    if isinstance(screenshot, np.ndarray):
        return screenshot
    # PIL Image
    if hasattr(screenshot, "convert") and hasattr(screenshot, "size"):
        import cv2

        rgb = np.asarray(screenshot.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return screenshot


def _pick_box_for_prompt(
    labels: list[str],
    boxes: list[tuple[float, float, float, float]],
    prompt: str,
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    q = (prompt or "").strip().lower()
    if not q:
        return boxes[0]
    tokens = [t for t in re.split(r"\W+", q) if len(t) >= 2]
    best_i = 0
    best_score = -1.0
    for i, label in enumerate(labels[: len(boxes)]):
        text = str(label or "").lower()
        if not text:
            score = -0.5
        elif q in text or text in q:
            score = 100.0 + len(text)
        else:
            score = float(sum(1 for t in tokens if t in text))
        if score > best_score:
            best_score = score
            best_i = i
    return boxes[best_i]


def default_grounding_fn(screenshot: Any, prompt: str) -> dict[str, Any]:
    """Florence-2 OCR/region grounding; graceful when Florence/GPU missing."""
    frame = _as_bgr_frame(screenshot)
    if frame is None:
        return {
            "ok": False,
            "error": "no screenshot for grounding",
            "labels": [],
            "boxes_xyxy": [],
            "picked_xyxy": None,
            "image_wh": (0, 0),
        }
    try:
        from dana.vision.florence_engine import (
            norm_box_to_screen,
            run_ocr_with_region,
        )
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"Florence unavailable: {exc}",
            "labels": [],
            "boxes_xyxy": [],
            "picked_xyxy": None,
            "image_wh": (0, 0),
        }

    try:
        result = run_ocr_with_region(frame)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Florence inference failed: {exc}",
            "labels": [],
            "boxes_xyxy": [],
            "picked_xyxy": None,
            "image_wh": (0, 0),
        }

    if not result.get("ok"):
        return {
            "ok": False,
            "error": str(result.get("error") or "Florence OCR failed"),
            "labels": [],
            "boxes_xyxy": [],
            "picked_xyxy": None,
            "image_wh": tuple(result.get("image_wh") or (0, 0)),
        }

    labels = [str(x) for x in (result.get("labels") or [])]
    raw_boxes = list(result.get("boxes_xyxy_norm") or [])
    image_wh = tuple(result.get("image_wh") or (int(frame.shape[1]), int(frame.shape[0])))
    screen_boxes: list[tuple[float, float, float, float]] = []
    for box in raw_boxes:
        if not box or len(box) < 4:
            continue
        xyxy = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        mapped = norm_box_to_screen(xyxy, image_wh=(int(image_wh[0]), int(image_wh[1])))
        if mapped is not None:
            screen_boxes.append(
                (float(mapped[0]), float(mapped[1]), float(mapped[2]), float(mapped[3]))
            )
        else:
            # Already in frame pixels from post_process(..., image_size=orig_wh).
            screen_boxes.append(xyxy)

    picked = _pick_box_for_prompt(labels, screen_boxes, prompt)
    matched_label = ""
    if picked is not None and screen_boxes:
        try:
            matched_label = labels[screen_boxes.index(picked)]
        except ValueError:
            matched_label = prompt

    return {
        "ok": True,
        "error": "",
        "labels": labels,
        "boxes_xyxy": screen_boxes,
        "picked_xyxy": picked,
        "matched_label": matched_label,
        "image_wh": image_wh,
    }


def _default_screenshot_fn() -> Any:
    try:
        from dana.vision_tools import capture_screen_frame

        return capture_screen_frame()
    except Exception as exc:  # noqa: BLE001
        _log.debug("screenshot_fn failed: %s", exc)
        return None


def _default_click_fn(x: int, y: int) -> None:
    from dana.tools.os_control import click_left_sendinput, move_cursor_absolute

    move_cursor_absolute(int(x), int(y))
    click_left_sendinput()


def _default_double_click_fn(x: int, y: int) -> None:
    import time

    from dana.tools.os_control import click_left_sendinput, move_cursor_absolute

    move_cursor_absolute(int(x), int(y))
    click_left_sendinput()
    time.sleep(0.05)
    click_left_sendinput()


def _default_type_fn(text: str) -> None:
    from dana.tools.os_control import execute_os_keystrokes

    execute_os_keystrokes(text or "")


def _default_hotkey_fn(chord: str) -> None:
    from dana.tools.os_control import execute_os_keystrokes

    execute_os_keystrokes("", hotkey=(chord or "").strip())


class MacroEngine:
    """Record Florence-grounded UI steps and replay them semantically."""

    def __init__(
        self,
        *,
        macros_dir: Path | str | None = None,
        grounding_fn: GroundingFn | None = None,
        screenshot_fn: ScreenshotFn | None = None,
        click_fn: ClickFn | None = None,
        type_fn: TypeFn | None = None,
        hotkey_fn: HotkeyFn | None = None,
        double_click_fn: DoubleClickFn | None = None,
    ) -> None:
        self.macros_dir = Path(macros_dir) if macros_dir else default_macros_dir()
        self.grounding_fn: GroundingFn = grounding_fn or default_grounding_fn
        self.screenshot_fn: ScreenshotFn = screenshot_fn or _default_screenshot_fn
        self.click_fn: ClickFn = click_fn or _default_click_fn
        self.type_fn: TypeFn = type_fn or _default_type_fn
        self.hotkey_fn: HotkeyFn = hotkey_fn or _default_hotkey_fn
        self.double_click_fn: DoubleClickFn = double_click_fn or _default_double_click_fn
        self._draft_id: str = ""
        self._draft_description: str = ""
        self._draft_steps: list[MacroStep] = []

    def begin_recording(self, macro_id: str, description: str = "") -> None:
        """Start a new in-memory draft sequence."""
        self._draft_id = sanitize_macro_id(macro_id)
        self._draft_description = str(description or "")
        self._draft_steps = []

    def record_step(
        self,
        label: str,
        action_type: str,
        value: Optional[str],
        screenshot: Any,
    ) -> MacroStep:
        """Ground ``label`` on ``screenshot`` via Florence and append a MacroStep."""
        action = str(action_type or "").strip().lower()
        if action not in _ALLOWED_ACTIONS:
            raise ValueError(
                f"unsupported action_type={action_type!r}; "
                f"expected one of {sorted(_ALLOWED_ACTIONS)}"
            )
        target = str(label or "").strip() or "target"
        prompt = target  # Florence-2 phrase-grounding prompt

        grounding = self.grounding_fn(screenshot, prompt)
        if grounding.get("ok") and grounding.get("matched_label"):
            matched = str(grounding.get("matched_label") or "").strip()
            if matched:
                prompt = matched

        step = MacroStep(
            target_label=target,
            action_type=action,
            action_value=value,
            visual_context_prompt=prompt,
        )
        self._draft_steps.append(step)
        return step

    def finish_recording(self) -> MacroSequence:
        """Seal the draft into a MacroSequence (does not auto-save)."""
        mid = self._draft_id or "macro"
        seq = MacroSequence(
            macro_id=mid,
            description=self._draft_description,
            steps=list(self._draft_steps),
        )
        return seq

    def macro_path(self, macro_id: str) -> Path:
        safe = sanitize_macro_id(macro_id)
        return self.macros_dir / f"{safe}.json"

    def save_macro(self, macro_sequence: MacroSequence) -> Path:
        """Persist sequence to ``dana/macros/<macro_id>.json`` (sanitized id)."""
        seq = (
            macro_sequence
            if isinstance(macro_sequence, MacroSequence)
            else MacroSequence.model_validate(macro_sequence)
        )
        safe_id = sanitize_macro_id(seq.macro_id)
        seq = seq.model_copy(update={"macro_id": safe_id})
        self.macros_dir.mkdir(parents=True, exist_ok=True)
        path = self.macro_path(safe_id)
        path.write_text(
            seq.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load_macro(self, macro_id: str) -> MacroSequence:
        path = self.macro_path(macro_id)
        if not path.is_file():
            raise FileNotFoundError(f"macro not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return MacroSequence.model_validate(data)

    def replay_macro(self, macro_id: str) -> dict[str, Any]:
        """Load macro, re-ground each step on a fresh screenshot, execute actions."""
        seq = self.load_macro(macro_id)
        executed: list[dict[str, Any]] = []
        for i, step in enumerate(seq.steps):
            shot = self.screenshot_fn()
            grounding = self.grounding_fn(shot, step.visual_context_prompt)
            picked = grounding.get("picked_xyxy")
            if not grounding.get("ok") or picked is None:
                return {
                    "ok": False,
                    "macro_id": seq.macro_id,
                    "error": (
                        f"step {i} ({step.target_label!r}): "
                        f"{grounding.get('error') or 'target ROI not found'}"
                    ),
                    "executed": executed,
                }
            xyxy = (
                float(picked[0]),
                float(picked[1]),
                float(picked[2]),
                float(picked[3]),
            )
            cx, cy = bbox_center(xyxy)
            action = step.action_type
            if action == "click":
                self.click_fn(cx, cy)
            elif action == "double_click":
                self.double_click_fn(cx, cy)
            elif action == "type_text":
                # Focus target then type.
                self.click_fn(cx, cy)
                self.type_fn(str(step.action_value or ""))
            elif action == "key_combination":
                self.hotkey_fn(str(step.action_value or ""))
            else:
                return {
                    "ok": False,
                    "macro_id": seq.macro_id,
                    "error": f"step {i}: unsupported action_type={action!r}",
                    "executed": executed,
                }
            executed.append(
                {
                    "index": i,
                    "target_label": step.target_label,
                    "action_type": action,
                    "coords": [cx, cy],
                    "bbox": list(xyxy),
                }
            )
        return {
            "ok": True,
            "macro_id": seq.macro_id,
            "steps": len(executed),
            "executed": executed,
            "error": "",
        }
