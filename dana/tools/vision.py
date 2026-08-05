"""Screen OCR actuator + Phase 3 visual history summaries."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
except ImportError:
    pass

# Periodic extraction cadence (tracker may call maybe_extract_screen_history).
SCREEN_HISTORY_INTERVAL_S = 12.0
SCREEN_HISTORY_TTL_S = 180
_LAST_EXTRACT_MONO = 0.0
_EXTRACT_LOCK = threading.Lock()

_WS_RE = re.compile(r"\s+")
_LINE_NOISE_RE = re.compile(r"^[\W_\d]+$")


def _clean_ocr_lines(text: str, *, limit: int = 12) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = _WS_RE.sub(" ", (raw or "").strip())
        if len(line) < 3:
            continue
        if _LINE_NOISE_RE.match(line):
            continue
        if line.lower() in {"ok", "cancel", "file", "edit", "view", "help"}:
            continue
        if line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _ocr_primary_monitor() -> str:
    try:
        import mss
        import pytesseract
        from PIL import Image
        from pytesseract import TesseractNotFoundError
    except ImportError as exc:  # noqa: BLE001
        return f"SYSTEM_ERROR: missing dependency ({exc})."

    try:
        img = None
        # Prefer latest full frame from tracker when available (point-in-time).
        try:
            from dana.tracker import get_latest_full_frame
            import cv2

            full = get_latest_full_frame()
            if full is not None:
                rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            with mss.MSS() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                shot = sct.grab(mon)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return str(pytesseract.image_to_string(img) or "")
    except TesseractNotFoundError:
        return "SYSTEM_ERROR: Tesseract OCR binary not found on host OS."
    except Exception as exc:  # noqa: BLE001
        return f"SYSTEM_ERROR: OCR failed ({exc})."


def _det_labels_from_buffer() -> list[str]:
    labels: list[str] = []
    try:
        from dana.tracker import get_latest_sample

        sample = get_latest_sample()
        if sample is None:
            return labels
        for item in sample.dets or ():
            # dets are (xyxy, name, conf) tuples from tracker_worker
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[1] or "").strip()
            elif isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = ""
            if name and name not in labels:
                labels.append(name)
            if len(labels) >= 8:
                break
    except Exception:  # noqa: BLE001
        pass
    return labels


def summarize_visual_history(*, seconds: float = 30.0) -> str:
    """Parse recent buffer + current OCR into a concise NL observation.

    Stores a short-TTL ``screen_history`` fact in EpisodicMemoryStore.
    """
    ocr_raw = _ocr_primary_monitor()
    if ocr_raw.startswith("SYSTEM_ERROR:"):
        return ocr_raw

    lines = _clean_ocr_lines(ocr_raw)
    labels = _det_labels_from_buffer()
    seq_n = 0
    try:
        from dana.tracker import get_recent_frame_sequence

        seq_n = len(get_recent_frame_sequence(seconds=seconds))
    except Exception:  # noqa: BLE001
        seq_n = 0

    parts: list[str] = []
    if labels:
        parts.append("Visible objects: " + ", ".join(labels) + ".")
    if lines:
        # Prefer HUD-like short lines and notification-ish longer lines.
        hud = [ln for ln in lines if len(ln) <= 48][:6]
        notes = [ln for ln in lines if len(ln) > 48][:4]
        if hud:
            parts.append("On-screen text/HUD: " + "; ".join(hud) + ".")
        if notes:
            parts.append("Notable text: " + " | ".join(notes) + ".")
    if not parts:
        parts.append("No clear HUD text or objects were readable on screen.")
    if seq_n > 1:
        parts.append(f"Temporal context: {seq_n} buffered frames (~{int(seconds)}s).")

    summary = " ".join(parts).strip()

    try:
        from dana.memory.store import get_episodic_store

        store = get_episodic_store()
        key = f"screen_hist_{int(time.time())}"
        store.add_fact(
            "screen_history",
            key,
            {
                "summary": summary,
                "labels": labels,
                "ocr_lines": lines[:10],
                "buffer_frames": seq_n,
            },
            confidence_score=0.7,
            ttl_seconds=SCREEN_HISTORY_TTL_S,
        )
        # Opportunistic prune of expired short-TTL visual telemetry.
        store.prune_expired_entries()
    except Exception:  # noqa: BLE001
        pass

    return summary


def maybe_extract_screen_history(*, force: bool = False) -> str | None:
    """Rate-limited background helper (≈ every 10–15s). Returns summary or None."""
    global _LAST_EXTRACT_MONO
    with _EXTRACT_LOCK:
        now = time.monotonic()
        if not force and (now - _LAST_EXTRACT_MONO) < SCREEN_HISTORY_INTERVAL_S:
            return None
        _LAST_EXTRACT_MONO = now
    try:
        return summarize_visual_history(seconds=30.0)
    except Exception:  # noqa: BLE001
        return None


def _recent_screen_history_blurb(*, limit: int = 3) -> str:
    try:
        from dana.memory.store import get_episodic_store

        facts = get_episodic_store().list_facts(include_expired=False)
        hist = [f for f in facts if f.get("category") == "screen_history"]
        hist.sort(key=lambda f: float(f.get("timestamp") or 0.0), reverse=True)
        blurbs: list[str] = []
        for fact in hist[:limit]:
            raw = fact.get("value")
            if isinstance(raw, str):
                try:
                    import json

                    parsed: Any = json.loads(raw)
                except Exception:  # noqa: BLE001
                    parsed = raw
            else:
                parsed = raw
            if isinstance(parsed, dict):
                s = str(parsed.get("summary") or "").strip()
            else:
                s = str(parsed or "").strip()
            if s:
                blurbs.append(s)
        return " ".join(blurbs)
    except Exception:  # noqa: BLE001
        return ""


def analyze_visual_context() -> str:
    """Return a natural-language screen summary (not raw ``<screen_text>`` XML).

    Emits UI telemetry ``STATE_CHANGE`` status=executing before capture.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="analyze_visual_context")
    except Exception:  # noqa: BLE001
        pass

    current = summarize_visual_history(seconds=30.0)
    if current.startswith("SYSTEM_ERROR:"):
        return current
    prior = _recent_screen_history_blurb(limit=2)
    if prior and prior != current:
        return (
            f"Current screen: {current} "
            f"Recent visual history: {prior}"
        )
    return f"Current screen: {current}"
