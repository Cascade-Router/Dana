"""Screen OCR actuator: primary-monitor capture via mss → Pillow → pytesseract."""

from __future__ import annotations

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
except ImportError:
    pass


def analyze_visual_context() -> str:
    """Capture the primary monitor and return OCR text as ``<screen_text>…``.

    Emits UI telemetry ``STATE_CHANGE`` status=executing before capture.
    Returns a fixed system error string when the Tesseract binary is missing.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="analyze_visual_context")
    except Exception:  # noqa: BLE001
        pass

    try:
        import mss
        import pytesseract
        from PIL import Image
        from pytesseract import TesseractNotFoundError
    except ImportError as exc:  # noqa: BLE001
        return f"SYSTEM_ERROR: missing dependency ({exc})."

    try:
        # mss.MSS is the supported constructor; mss.mss() is deprecated.
        with mss.MSS() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        text = pytesseract.image_to_string(img)
    except TesseractNotFoundError:
        return "SYSTEM_ERROR: Tesseract OCR binary not found on host OS."

    return f"<screen_text>{(text or '').strip()}</screen_text>"
