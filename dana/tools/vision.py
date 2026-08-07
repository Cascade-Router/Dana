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


def click_ui_element(target_description: str) -> str:
    """Locate ``target_description`` on screen and safely click its centroid.

    Pipeline: capture a screen frame -> hybrid UIA/Florence grounding
    (``dana.graph.nodes.vision.locate_ui_element``) -> convert the returned
    ``[0,1000]``-normalized box to real screen pixels and click its inset
    centroid via ``dana.tools.mouse_actuator.MouseActuator`` (rate-limited,
    failsafe-bounded, kill-switch aware, ``DONNA_OS_DRY_RUN``-safe).

    Returns a single observation string for the LLM: ``"SUCCESS: ..."`` on a
    completed click, ``"ERROR: ..."`` when the element can't be located or
    the click is blocked, or ``"HALTED: ..."`` if the global kill switch
    fired mid-click.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="click_ui_element")
    except Exception:  # noqa: BLE001
        pass

    label = str(target_description or "").strip()
    if not label:
        return "ERROR: click_ui_element requires a non-empty target_description"

    try:
        from dana.vision_tools import capture_screen_frame

        image = capture_screen_frame()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: click_ui_element failed to capture screen: {exc}"
    if image is None:
        return "ERROR: click_ui_element could not capture a screen frame"

    try:
        from dana.graph.nodes.vision import locate_ui_element

        bbox_1000 = locate_ui_element(image, label)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: click_ui_element vision lookup failed: {exc}"
    if bbox_1000 is None:
        return f"ERROR: Could not locate {label!r} on screen"

    try:
        from dana.tools.os_control import get_screen_size

        screen_w, screen_h = get_screen_size()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: click_ui_element could not read screen size: {exc}"

    from dana.tools.mouse_actuator import MouseActuator

    result = MouseActuator().click_bbox(
        bbox_1000,
        source_resolution=(1000.0, 1000.0),
        target_resolution=(float(screen_w), float(screen_h)),
    )
    if result.get("halted"):
        return f"HALTED: click_ui_element — {result.get('error')}"
    if not result.get("ok"):
        return (
            f"ERROR: click_ui_element failed to click {label!r}: "
            f"{result.get('error')}"
        )
    x, y = result["point"]
    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return f"SUCCESS: Clicked {label!r} at ({x}, {y}){dry_note}"


def type_text_in_element(target_description: str, text: str) -> str:
    """Locate ``target_description``, click to focus it, then type ``text``.

    Pipeline: the same screen-capture + hybrid grounding lookup as
    ``click_ui_element`` -> ``MouseActuator.click_bbox`` to focus the
    element -> ``KeyboardActuator.type_text`` to type into it. Fails closed
    at any stage: locating, focusing, and typing are each validated, and a
    failure at any step aborts before the next (a click that misses never
    proceeds to type into the wrong place).

    Returns a single observation string for the LLM: ``"SUCCESS: ..."`` on
    a completed click+type, ``"ERROR: ..."`` when locating/focusing/typing
    fails, or ``"HALTED: ..."`` if the global kill switch fired mid-action.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="type_text_in_element")
    except Exception:  # noqa: BLE001
        pass

    label = str(target_description or "").strip()
    if not label:
        return "ERROR: type_text_in_element requires a non-empty target_description"
    body = text if isinstance(text, str) else str(text or "")
    if not body.strip():
        return "ERROR: type_text_in_element requires non-empty text"

    try:
        from dana.vision_tools import capture_screen_frame

        image = capture_screen_frame()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: type_text_in_element failed to capture screen: {exc}"
    if image is None:
        return "ERROR: type_text_in_element could not capture a screen frame"

    try:
        from dana.graph.nodes.vision import locate_ui_element

        bbox_1000 = locate_ui_element(image, label)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: type_text_in_element vision lookup failed: {exc}"
    if bbox_1000 is None:
        return f"ERROR: Could not locate {label!r} on screen"

    try:
        from dana.tools.os_control import get_screen_size

        screen_w, screen_h = get_screen_size()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: type_text_in_element could not read screen size: {exc}"

    from dana.tools.mouse_actuator import MouseActuator

    click_result = MouseActuator().click_bbox(
        bbox_1000,
        source_resolution=(1000.0, 1000.0),
        target_resolution=(float(screen_w), float(screen_h)),
    )
    if click_result.get("halted"):
        return f"HALTED: type_text_in_element — {click_result.get('error')}"
    if not click_result.get("ok"):
        return (
            f"ERROR: type_text_in_element failed to focus {label!r}: "
            f"{click_result.get('error')}"
        )

    # Real click: let the OS register focus before the first keystroke.
    if not click_result.get("dry_run"):
        time.sleep(0.15)

    from dana.tools.keyboard_actuator import KeyboardActuator

    type_result = KeyboardActuator().type_text(body)
    if type_result.get("halted"):
        return f"HALTED: type_text_in_element — {type_result.get('error')}"
    if not type_result.get("ok"):
        return (
            f"ERROR: type_text_in_element clicked {label!r} but failed to type: "
            f"{type_result.get('error')}"
        )

    dry_note = (
        " (dry_run)"
        if (click_result.get("dry_run") or type_result.get("dry_run"))
        else ""
    )
    return f"SUCCESS: Clicked {label!r} and typed text.{dry_note}"


_SCROLL_AMOUNT_TICKS: dict[str, int] = {"small": 2, "medium": 5, "large": 12}


def scroll_screen(direction: str, amount: str = "medium") -> str:
    """Scroll the screen to reveal off-screen UI elements.

    LLMs are unreliable at picking a raw wheel-tick count, so this accepts a
    semantic ``amount`` ("small"/"medium"/"large") and maps it to a sensible
    tick count for ``dana.tools.scroll_actuator.ScrollActuator``. Pure
    actuation — no vision lookup involved, so it works even when a specific
    target hasn't been located yet (e.g. "scroll down to see more").

    Returns ``"SUCCESS: ..."`` on a completed scroll, ``"ERROR: ..."`` for
    an unknown direction/amount or a blocked scroll (rate limit), or
    ``"HALTED: ..."`` if the global kill switch fired mid-scroll.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="scroll_screen")
    except Exception:  # noqa: BLE001
        pass

    key = str(amount or "medium").strip().lower()
    ticks = _SCROLL_AMOUNT_TICKS.get(key)
    if ticks is None:
        return (
            f"ERROR: scroll_screen unknown amount {amount!r}; expected one "
            f"of {sorted(_SCROLL_AMOUNT_TICKS)}"
        )

    from dana.tools.scroll_actuator import ScrollActuator

    result = ScrollActuator().scroll(direction, ticks)
    if result.get("halted"):
        return f"HALTED: scroll_screen — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: scroll_screen failed: {result.get('error')}"

    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return f"SUCCESS: Scrolled {result.get('direction')} by a {key} amount.{dry_note}"


def drag_ui_element(source_description: str, destination_description: str) -> str:
    """Locate two UI elements by description and drag from the first to the second.

    Pipeline: capture a screen frame -> hybrid UIA/Florence grounding
    (``dana.graph.nodes.vision.locate_ui_element``) run once for the drag
    source and once for the destination -> convert both returned
    ``[0,1000]``-normalized boxes to real screen pixels and drag from the
    source's inset centroid to the destination's via
    ``dana.tools.drag_actuator.DragActuator`` (rate-limited, failsafe-bounded,
    kill-switch aware, ``DONNA_OS_DRY_RUN``-safe).

    Fails closed: if either element can't be located, the drag never starts.

    Returns a single observation string for the LLM: ``"SUCCESS: ..."`` on a
    completed drag, ``"ERROR: ..."`` when a lookup or the drag itself fails,
    or ``"HALTED: ..."`` if the global kill switch fired mid-drag.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="drag_ui_element")
    except Exception:  # noqa: BLE001
        pass

    source_label = str(source_description or "").strip()
    dest_label = str(destination_description or "").strip()
    if not source_label:
        return "ERROR: drag_ui_element requires a non-empty source_description"
    if not dest_label:
        return "ERROR: drag_ui_element requires a non-empty destination_description"

    try:
        from dana.vision_tools import capture_screen_frame

        image = capture_screen_frame()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: drag_ui_element failed to capture screen: {exc}"
    if image is None:
        return "ERROR: drag_ui_element could not capture a screen frame"

    from dana.graph.nodes.vision import locate_ui_element

    try:
        source_bbox_1000 = locate_ui_element(image, source_label)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: drag_ui_element vision lookup failed for source {source_label!r}: {exc}"
    if source_bbox_1000 is None:
        return f"ERROR: Could not locate {source_label!r} on screen"

    try:
        dest_bbox_1000 = locate_ui_element(image, dest_label)
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: drag_ui_element vision lookup failed for destination "
            f"{dest_label!r}: {exc}"
        )
    if dest_bbox_1000 is None:
        return f"ERROR: Could not locate {dest_label!r} on screen"

    try:
        from dana.tools.os_control import get_screen_size

        screen_w, screen_h = get_screen_size()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: drag_ui_element could not read screen size: {exc}"

    from dana.tools.drag_actuator import DragActuator

    result = DragActuator().drag_bbox(
        source_bbox_1000,
        dest_bbox_1000,
        source_resolution=(1000.0, 1000.0),
        target_resolution=(float(screen_w), float(screen_h)),
    )
    if result.get("halted"):
        return f"HALTED: drag_ui_element — {result.get('error')}"
    if not result.get("ok"):
        return (
            f"ERROR: drag_ui_element failed to drag {source_label!r} to "
            f"{dest_label!r}: {result.get('error')}"
        )
    sx, sy = result["source_point"]
    dxp, dyp = result["dest_point"]
    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return (
        f"SUCCESS: Dragged {source_label!r} from ({sx}, {sy}) to "
        f"{dest_label!r} at ({dxp}, {dyp}){dry_note}"
    )


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
