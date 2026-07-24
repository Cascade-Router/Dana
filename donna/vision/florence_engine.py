"""Florence-2-base engine for OCR + spatial UI grounding (CUDA FP16).

Loads ``microsoft/Florence-2-base`` with the same offline-cache fallback used
for Distil-Whisper: try ``local_files_only=True``, then download on ``OSError``.

Uses ``AutoProcessor`` / ``AutoModelForCausalLM`` with ``trust_remote_code=True``
(as required by Florence-2), plus a small transformers-5.x compatibility shim
for ``forced_bos_token_id`` on the remote language config.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

_log = logging.getLogger("donna.vision.florence")

FLORENCE_ID = "microsoft/Florence-2-base"
TASK_OCR_WITH_REGION = "<OCR_WITH_REGION>"

_lock = threading.Lock()
_processor: Any = None
_model: Any = None
_device: Any = None
_dtype: Any = None
_load_error: str | None = None
_compat_patched = False


def florence_is_loaded() -> bool:
    return _model is not None and _processor is not None


def florence_load_error() -> str | None:
    return _load_error


def _patch_transformers_florence_compat() -> None:
    """Shims for Florence-2 remote code under transformers>=5.

    - ``forced_bos_token_id`` probed before PretrainedConfig finishes init
    - ``additional_special_tokens`` renamed to ``extra_special_tokens`` in v5
    """
    global _compat_patched
    if _compat_patched:
        return
    try:
        from transformers.configuration_utils import PretrainedConfig

        original = PretrainedConfig.__getattribute__

        def _safe_getattr(self, key):  # type: ignore[no-untyped-def]
            try:
                return original(self, key)
            except AttributeError:
                if key in {"forced_bos_token_id", "forced_eos_token_id"}:
                    return None
                raise

        PretrainedConfig.__getattribute__ = _safe_getattr  # type: ignore[method-assign]
    except Exception as exc:  # noqa: BLE001
        _log.debug("Florence config shim skipped: %s", exc)

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):

            def _additional_special_tokens(self):  # type: ignore[no-untyped-def]
                extra = getattr(self, "extra_special_tokens", None)
                if extra is None:
                    return []
                if isinstance(extra, dict):
                    return list(extra.values())
                return list(extra)

            PreTrainedTokenizerBase.additional_special_tokens = property(  # type: ignore[attr-defined]
                _additional_special_tokens
            )
    except Exception as exc:  # noqa: BLE001
        _log.debug("Florence tokenizer shim skipped: %s", exc)

    _compat_patched = True
    _log.debug("Applied Florence-2 / transformers v5 compatibility shims")


def _resolve_device_dtype():
    import torch

    # Stage 7.1 — hard-cap CUDA to 50% VRAM before Florence allocates.
    try:
        from donna.middleware.resource_cap import apply_torch_vram_half_cap

        apply_torch_vram_half_cap(0)
    except Exception:  # noqa: BLE001
        pass

    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.float16
    return torch.device("cpu"), torch.float32


def load_florence(*, local_files_only: bool = True) -> tuple[Any, Any, Any, Any]:
    """Load Florence-2 processor+model on cuda:0 float16 with cache fallback.

    Returns ``(processor, model, device, dtype)``.
    """
    global _processor, _model, _device, _dtype, _load_error

    if _model is not None and _processor is not None:
        return _processor, _model, _device, _dtype

    with _lock:
        if _model is not None and _processor is not None:
            return _processor, _model, _device, _dtype

        import torch

        # Stage 7.1 — immediately after importing torch, lock VRAM to 50%.
        try:
            from donna.middleware.resource_cap import apply_torch_vram_half_cap

            apply_torch_vram_half_cap(0)
        except Exception:  # noqa: BLE001
            pass

        from transformers import AutoModelForCausalLM, AutoProcessor

        _patch_transformers_florence_compat()

        device, dtype = _resolve_device_dtype()
        _log.info(
            "Loading %s (local_files_only=True first, device=%s, dtype=%s, "
            "trust_remote_code=True)...",
            FLORENCE_ID,
            device,
            dtype,
        )
        t0 = time.perf_counter()

        def _load_pair(*, files_only: bool):
            proc = AutoProcessor.from_pretrained(
                FLORENCE_ID,
                trust_remote_code=True,
                local_files_only=files_only,
            )
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    FLORENCE_ID,
                    dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=files_only,
                    attn_implementation="eager",
                )
            except TypeError:
                try:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        FLORENCE_ID,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=files_only,
                        attn_implementation="eager",
                    )
                except TypeError:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        FLORENCE_ID,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=files_only,
                    )
            return proc, mdl.to(device)

        try:
            processor, model = _load_pair(files_only=True)
        except OSError as exc:
            _log.warning(
                "%s not in local cache (%s); falling back to download "
                "(local_files_only=False)...",
                FLORENCE_ID,
                exc,
            )
            processor, model = _load_pair(files_only=False)

        model.eval()
        _tie_florence_language_weights(model)
        _processor = processor
        _model = model
        _device = device
        _dtype = dtype
        _load_error = None
        _log.info(
            "Florence-2 ready in %.1fs on %s (dtype=%s).",
            time.perf_counter() - t0,
            device,
            dtype,
        )
        return _processor, _model, _device, _dtype


def _tie_florence_language_weights(model: Any) -> None:
    """Re-tie shared / embed / lm_head weights (often reported MISSING on load)."""
    try:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    except Exception:  # noqa: BLE001
        pass
    try:
        lm = getattr(model, "language_model", None)
        inner = getattr(lm, "model", None) if lm is not None else None
        shared = getattr(inner, "shared", None) if inner is not None else None
        if shared is None:
            return
        weight = shared.weight
        enc = getattr(inner, "encoder", None)
        dec = getattr(inner, "decoder", None)
        if enc is not None and hasattr(enc, "embed_tokens"):
            enc.embed_tokens.weight = weight
        if dec is not None and hasattr(dec, "embed_tokens"):
            dec.embed_tokens.weight = weight
        if lm is not None and hasattr(lm, "lm_head"):
            lm.lm_head.weight = weight
    except Exception as exc:  # noqa: BLE001
        _log.debug("Florence weight tie skipped: %s", exc)


def reset_florence() -> None:
    """Drop cached Florence weights (tests / forced reload)."""
    global _processor, _model, _device, _dtype, _load_error
    with _lock:
        _processor = None
        _model = None
        _device = None
        _dtype = None
        _load_error = None


def _bgr_to_pil(frame: np.ndarray):
    from PIL import Image
    import cv2

    arr = np.asarray(frame)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _quad_to_xyxy(quad: Any) -> tuple[float, float, float, float] | None:
    """Collapse Florence quad (8 nums or 4 pts) to axis-aligned xyxy."""
    try:
        vals = [float(v) for v in list(quad)]
    except Exception:  # noqa: BLE001
        return None
    if len(vals) >= 8:
        xs = vals[0:8:2]
        ys = vals[1:8:2]
        return min(xs), min(ys), max(xs), max(ys)
    if len(vals) >= 4:
        x1, y1, x2, y2 = vals[:4]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return None


def _normalize_parsed(parsed: Any, task: str) -> dict[str, Any]:
    """Normalize Florence post_process output to {labels, boxes_xyxy_norm}."""
    if not isinstance(parsed, dict):
        return {"labels": [], "boxes_xyxy_norm": [], "raw": parsed}

    payload = parsed.get(task) or parsed.get(task.strip("<>")) or parsed
    if not isinstance(payload, dict):
        for _key, val in parsed.items():
            if isinstance(val, dict) and (
                "labels" in val or "quad_boxes" in val or "bboxes" in val
            ):
                payload = val
                break
        else:
            return {"labels": [], "boxes_xyxy_norm": [], "raw": parsed}

    labels = list(payload.get("labels") or payload.get("rec_texts") or [])
    quads = (
        payload.get("quad_boxes")
        or payload.get("bboxes")
        or payload.get("boxes")
        or []
    )
    boxes: list[tuple[float, float, float, float]] = []
    for q in quads:
        xyxy = _quad_to_xyxy(q)
        if xyxy is not None:
            boxes.append(xyxy)
    if boxes and not labels:
        labels = [""] * len(boxes)
    n = min(len(labels), len(boxes)) if labels and boxes else 0
    if labels and boxes:
        n = min(len(labels), len(boxes))
    elif boxes:
        n = len(boxes)
        labels = [""] * n
    elif labels:
        n = 0
    return {
        "labels": [str(labels[i]) if i < len(labels) else "" for i in range(n)],
        "boxes_xyxy_norm": [
            boxes[i] if i < len(boxes) else (0.0, 0.0, 0.0, 0.0) for i in range(n)
        ],
        "raw": parsed,
    }


def norm_box_to_screen(
    xyxy_norm: tuple[float, float, float, float],
    *,
    image_wh: tuple[int, int],
    monitor: dict[str, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Map Florence 0–1000 (or pixel) box → absolute screen pixels."""
    try:
        from donna.tracker import map_box_to_screen, primary_monitor_geometry
    except Exception:  # noqa: BLE001
        map_box_to_screen = None  # type: ignore[assignment]
        primary_monitor_geometry = None  # type: ignore[assignment]

    x1, y1, x2, y2 = (float(v) for v in xyxy_norm)
    iw, ih = int(image_wh[0]), int(image_wh[1])
    if iw <= 0 or ih <= 0:
        return None

    # Detect 0–1000 normalized coords (Florence default) vs already-pixel coords.
    span = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if span <= 1000.5:
        fx1 = x1 / 1000.0 * iw
        fy1 = y1 / 1000.0 * ih
        fx2 = x2 / 1000.0 * iw
        fy2 = y2 / 1000.0 * ih
    else:
        fx1, fy1, fx2, fy2 = x1, y1, x2, y2

    mon = monitor
    if mon is None and primary_monitor_geometry is not None:
        try:
            mon = primary_monitor_geometry()
        except Exception:  # noqa: BLE001
            mon = None

    if map_box_to_screen is not None:
        mapped = map_box_to_screen(
            (fx1, fy1, fx2, fy2),
            frame_wh=(iw, ih),
            monitor=mon,
        )
        if mapped is not None:
            return mapped

    left = int((mon or {}).get("left", 0))
    top = int((mon or {}).get("top", 0))
    return (
        int(round(left + fx1)),
        int(round(top + fy1)),
        int(round(left + fx2)),
        int(round(top + fy2)),
    )


def run_ocr_with_region(
    frame: np.ndarray,
    *,
    task: str = TASK_OCR_WITH_REGION,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Run ``<OCR_WITH_REGION>`` on a BGR frame; return labels + norm boxes."""
    global _load_error
    try:
        processor, model, device, dtype = load_florence(local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        return {
            "ok": False,
            "error": f"Florence-2 load failed: {exc}",
            "labels": [],
            "boxes_xyxy_norm": [],
            "regions": [],
        }

    import torch

    image = _bgr_to_pil(frame)
    orig_wh = (int(frame.shape[1]), int(frame.shape[0]))
    # Florence vision tower expects square feature maps; processor size is 768².
    if image.size != (768, 768):
        image = image.resize((768, 768))
    prompt = task if task.startswith("<") else f"<{task.strip('<>')}>"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    with torch.no_grad():
        generated_ids = model.generate(
            **moved,
            max_new_tokens=int(max_new_tokens),
            num_beams=3,
            do_sample=False,
            use_cache=False,  # transformers 5 EncoderDecoderCache incompat with remote generate
        )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    try:
        # Scale loc tokens into original frame pixels for overlay mapping.
        parsed = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=orig_wh,
        )
    except Exception:  # noqa: BLE001
        parsed = {"raw_text": generated_text}

    normalized = _normalize_parsed(parsed, prompt)
    regions = []
    for label, box in zip(normalized["labels"], normalized["boxes_xyxy_norm"]):
        regions.append(
            {
                "text": str(label),
                "xyxy_norm": [float(v) for v in box],
            }
        )
    return {
        "ok": True,
        "error": "",
        "labels": normalized["labels"],
        "boxes_xyxy_norm": normalized["boxes_xyxy_norm"],
        "regions": regions,
        "image_wh": orig_wh,
        "raw_text": generated_text,
    }
