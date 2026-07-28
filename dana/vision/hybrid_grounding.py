"""Hybrid Win32 UIA + 2-stage Florence-2 crop-and-zoom UI grounding.

Pipeline
--------
1. Native UIA hit → return immediately (``[0, 1000]^4``).
2. Coarse Florence phrase grounding on the full image.
3. If the coarse box width **or** height is ``< 30`` in 1000-scale space,
   crop an ROI with 15% padding, upscale 2× (Lanczos/bicubic), run fine
   Florence on the zoomed crop, and project coordinates back to global
   ``[0, 1000]^4``.
4. Return the final normalized bbox (or ``None``).

All OS / model side-effects are injectable for offline tests.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional, Sequence

import numpy as np

from dana.vision.uia_provider import Win32UIAProvider

_log = logging.getLogger("dana.vision.hybrid")

NormBBox = list[float]

# Threshold in Florence 0–1000 space: either edge below this triggers zoom.
ZOOM_EDGE_THRESHOLD = 30.0
ROI_PADDING = 0.15
ZOOM_SCALE = 2

FlorenceGroundFn = Callable[[Any, str], Optional[NormBBox]]
CropZoomFn = Callable[
    [Any, NormBBox],
    tuple[Any, tuple[float, float, float, float]],
]
# crop_zoom returns (zoomed_image, crop_rect_px) where crop_rect is
# (x0, y0, x1, y1) in original image pixels.


def _as_bgr(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    if hasattr(image, "convert") and hasattr(image, "size"):
        try:
            import cv2

            rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            return np.asarray(image.convert("RGB"))
    return None


def _image_wh(image: Any) -> tuple[int, int]:
    frame = _as_bgr(image)
    if frame is not None and getattr(frame, "ndim", 0) >= 2:
        return int(frame.shape[1]), int(frame.shape[0])
    if hasattr(image, "size"):
        w, h = image.size
        return int(w), int(h)
    return 0, 0


def _clamp_box(box: Sequence[float]) -> NormBBox:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    x2 = max(0.0, min(1000.0, x2))
    y2 = max(0.0, min(1000.0, y2))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _box_needs_zoom(box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return w < ZOOM_EDGE_THRESHOLD or h < ZOOM_EDGE_THRESHOLD


def _pixels_to_norm1000(
    xyxy: Sequence[float],
    *,
    image_wh: tuple[int, int],
) -> NormBBox | None:
    iw, ih = int(image_wh[0]), int(image_wh[1])
    if iw <= 0 or ih <= 0:
        return None
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    span = max(abs(x1), abs(y1), abs(x2), abs(y2))
    # Already Florence-normalized.
    if span <= 1000.5:
        return _clamp_box((x1, y1, x2, y2))
    return _clamp_box(
        (
            x1 / iw * 1000.0,
            y1 / ih * 1000.0,
            x2 / iw * 1000.0,
            y2 / ih * 1000.0,
        )
    )


def _pick_label_box(
    labels: Sequence[str],
    boxes: Sequence[Sequence[float]],
    prompt: str,
) -> NormBBox | None:
    if not boxes:
        return None
    q = (prompt or "").strip().lower()
    if not q:
        box = boxes[0]
        return [float(v) for v in box[:4]]
    tokens = [t for t in re.split(r"\W+", q) if len(t) >= 2]
    best_i = 0
    best_score = -1.0
    for i, label in enumerate(list(labels)[: len(boxes)]):
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
    return [float(v) for v in boxes[best_i][:4]]


def default_florence_ground_fn(image: Any, target_label: str) -> Optional[NormBBox]:
    """Phrase-ground via existing ``run_ocr_with_region`` + label match.

    Returns a ``[0, 1000]`` bbox or ``None``. Does not invent a second Florence
    stack — reuses ``dana.vision.florence_engine``.
    """
    frame = _as_bgr(image)
    if frame is None:
        return None
    try:
        from dana.vision.florence_engine import run_ocr_with_region
    except ImportError as exc:
        _log.debug("Florence unavailable: %s", exc)
        return None

    try:
        result = run_ocr_with_region(frame)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Florence ground failed: %s", exc)
        return None
    if not result.get("ok"):
        return None

    labels = [str(x) for x in (result.get("labels") or [])]
    raw_boxes = list(result.get("boxes_xyxy_norm") or [])
    image_wh = tuple(
        result.get("image_wh") or (int(frame.shape[1]), int(frame.shape[0]))
    )
    norm_boxes: list[NormBBox] = []
    for box in raw_boxes:
        if not box or len(box) < 4:
            continue
        converted = _pixels_to_norm1000(box, image_wh=(int(image_wh[0]), int(image_wh[1])))
        if converted is not None:
            norm_boxes.append(converted)
    picked = _pick_label_box(labels, norm_boxes, target_label)
    return picked


def default_crop_zoom(
    image: Any,
    coarse_box_1000: NormBBox,
) -> tuple[Any, tuple[float, float, float, float]]:
    """Crop ROI with 15% padding and upscale 2× (Lanczos / bicubic).

    Returns ``(zoomed_bgr, (x0, y0, x1, y1))`` in original image pixels.
    """
    frame = _as_bgr(image)
    if frame is None:
        raise ValueError("crop_zoom requires an image array")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    x1, y1, x2, y2 = (float(v) for v in coarse_box_1000[:4])
    px1 = x1 / 1000.0 * w
    py1 = y1 / 1000.0 * h
    px2 = x2 / 1000.0 * w
    py2 = y2 / 1000.0 * h
    bw = max(1.0, px2 - px1)
    bh = max(1.0, py2 - py1)
    pad_x = bw * ROI_PADDING
    pad_y = bh * ROI_PADDING
    x0 = max(0.0, px1 - pad_x)
    y0 = max(0.0, py1 - pad_y)
    x1c = min(float(w), px2 + pad_x)
    y1c = min(float(h), py2 + pad_y)
    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = int(round(x1c)), int(round(y1c))
    if ix1 <= ix0:
        ix1 = min(w, ix0 + 1)
    if iy1 <= iy0:
        iy1 = min(h, iy0 + 1)
    crop = frame[iy0:iy1, ix0:ix1]
    cw = max(1, crop.shape[1])
    ch = max(1, crop.shape[0])
    zw, zh = cw * ZOOM_SCALE, ch * ZOOM_SCALE

    try:
        import cv2

        zoomed = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_CUBIC)
    except Exception:  # noqa: BLE001
        from PIL import Image

        if crop.ndim == 2:
            pil = Image.fromarray(crop)
        else:
            # BGR → RGB for PIL
            pil = Image.fromarray(crop[:, :, ::-1])
        pil = pil.resize((zw, zh), resample=Image.Resampling.LANCZOS)
        arr = np.asarray(pil)
        if arr.ndim == 3:
            zoomed = arr[:, :, ::-1].copy()
        else:
            zoomed = arr
    return zoomed, (float(ix0), float(iy0), float(ix1), float(iy1))


def project_roi_box_to_global(
    fine_box_1000: Sequence[float],
    *,
    crop_rect_px: tuple[float, float, float, float],
    image_wh: tuple[int, int],
    zoom_scale: float = ZOOM_SCALE,
) -> NormBBox:
    """Map a fine ``[0,1000]`` box on the zoomed ROI back to global ``[0,1000]``."""
    iw, ih = int(image_wh[0]), int(image_wh[1])
    cx0, cy0, cx1, cy1 = (float(v) for v in crop_rect_px)
    crop_w = max(1e-6, cx1 - cx0)
    crop_h = max(1e-6, cy1 - cy0)
    # Fine box is relative to the *zoomed* image; undo the upscale first.
    fx1, fy1, fx2, fy2 = (float(v) for v in fine_box_1000[:4])
    # Local pixel on original crop (not zoomed):
    lx1 = (fx1 / 1000.0) * crop_w
    ly1 = (fy1 / 1000.0) * crop_h
    lx2 = (fx2 / 1000.0) * crop_w
    ly2 = (fy2 / 1000.0) * crop_h
    # zoom_scale cancels: fine is normalized to zoomed size ≡ crop size in 0–1000.
    _ = zoom_scale
    gx1 = (cx0 + lx1) / max(1, iw) * 1000.0
    gy1 = (cy0 + ly1) / max(1, ih) * 1000.0
    gx2 = (cx0 + lx2) / max(1, iw) * 1000.0
    gy2 = (cy0 + ly2) / max(1, ih) * 1000.0
    return _clamp_box((gx1, gy1, gx2, gy2))


class HybridVisionGrounding:
    """UIA-first, Florence crop-and-zoom fallback UI localizer."""

    def __init__(
        self,
        *,
        uia_provider: Win32UIAProvider | None = None,
        florence_ground_fn: FlorenceGroundFn | None = None,
        crop_zoom_fn: CropZoomFn | None = None,
    ) -> None:
        self.uia_provider = uia_provider or Win32UIAProvider()
        self.florence_ground_fn = florence_ground_fn or default_florence_ground_fn
        self.crop_zoom_fn = crop_zoom_fn or default_crop_zoom
        self.last_stage: str = ""  # "uia" | "coarse" | "zoom" | "miss"

    def locate_ui_element(
        self,
        image: Any,
        target_label: str,
    ) -> Optional[NormBBox]:
        """Locate ``target_label``; return global ``[0,1000]`` xyxy or ``None``."""
        label = str(target_label or "").strip()
        if not label:
            self.last_stage = "miss"
            return None

        # 1) Native UIA — instant return on hit.
        try:
            uia_box = self.uia_provider.find_element_bounds(label)
        except Exception as exc:  # noqa: BLE001
            _log.debug("UIA provider error: %s", exc)
            uia_box = None
        if uia_box is not None and len(uia_box) >= 4:
            self.last_stage = "uia"
            return _clamp_box(uia_box)

        # 2) Coarse Florence on the full frame.
        try:
            coarse = self.florence_ground_fn(image, label)
        except Exception as exc:  # noqa: BLE001
            _log.debug("coarse Florence failed: %s", exc)
            coarse = None
        if coarse is None or len(coarse) < 4:
            self.last_stage = "miss"
            return None
        coarse = _clamp_box(coarse)

        # 3) Small target → crop & zoom → fine Florence → project.
        if not _box_needs_zoom(coarse):
            self.last_stage = "coarse"
            return coarse

        try:
            zoomed, crop_rect = self.crop_zoom_fn(image, coarse)
        except Exception as exc:  # noqa: BLE001
            _log.debug("crop_zoom failed (%s); returning coarse", exc)
            self.last_stage = "coarse"
            return coarse

        try:
            fine = self.florence_ground_fn(zoomed, label)
        except Exception as exc:  # noqa: BLE001
            _log.debug("fine Florence failed: %s", exc)
            fine = None
        if fine is None or len(fine) < 4:
            # Fall back to coarse rather than miss entirely.
            self.last_stage = "coarse"
            return coarse

        iw, ih = _image_wh(image)
        if iw <= 0 or ih <= 0:
            zw, zh = _image_wh(zoomed)
            # Recover original size from crop + zoom.
            cw = max(1.0, (crop_rect[2] - crop_rect[0]))
            ch = max(1.0, (crop_rect[3] - crop_rect[1]))
            iw = int(round(crop_rect[0] + cw)) if zw <= 0 else iw
            ih = int(round(crop_rect[1] + ch)) if zh <= 0 else ih
            if iw <= 0 or ih <= 0:
                self.last_stage = "coarse"
                return coarse

        global_box = project_roi_box_to_global(
            fine,
            crop_rect_px=crop_rect,
            image_wh=(iw, ih),
        )
        self.last_stage = "zoom"
        return global_box
