"""Desktop Omni-Vision — lets the agent "see" the user's actual desktop
(the primary monitor), not just its own sandboxed workspace images
(``dana.plugins.vision.image_analysis``) or a CAD viewport
(``dana.tools.cad_vision``). Backs the "os_tools" capability domain's
``analyze_desktop_screen`` tool (``dana.core.react_dispatch``).

Privacy note: capturing the user's actual screen is fundamentally
different from reading a sandboxed file this agent already wrote — it can
see whatever else is currently on screen (another app, a private
document, a password manager). This is why ``analyze_desktop_screen`` has
no ``"read_only": true`` in its tools.json declaration — even though
nothing gets written to disk, ``dana.core.react_dispatch.is_mutating_tool``'s
fail-closed schema check still gates it: the HITL approval gate is the
privacy control here,
the exact mechanism ``dana.plugins.os.process_manager.run_python_script``
already relies on for ITS highest-risk operation.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from dana.core.model_provider import ModelProvider, cloud_fallback_enabled, cloud_provider_name

# Longest edge, in pixels, a captured screenshot is downscaled to before
# it's ever base64-encoded or sent to a VLM — a raw multi-monitor/4K
# capture can be tens of megabytes; this keeps the payload well inside
# typical VLM upload limits and avoids a multi-second encode/upload
# latency spike on every single capture. Only ever shrinks (never
# upscales) a smaller monitor's capture.
_MAX_WIDTH = 1920

# An effectively-unbounded height so Image.thumbnail's aspect-preserving
# box constraint is driven by width alone, matching "scale down to a max
# WIDTH" — not the (width, height) dual-bound most existing thumbnail()
# callers in this codebase use (e.g. dana.tools.os_control's 1280x720).
_UNBOUNDED_HEIGHT = 10_000

# JPEG, not PNG (dana.tools.os_control.capture_screen_png_bytes's format)
# — a lossy encode of a full desktop capture is a fraction of PNG's size
# at a quality still more than sufficient for a VLM to read text/UI chrome
# back off it.
_JPEG_QUALITY = 85


def _capture_primary_monitor_jpeg_b64() -> str:
    """Grabs the primary monitor via ``mss``, downscales to ``_MAX_WIDTH``
    via Pillow (aspect-ratio preserved, never upscales), and returns a
    base64-encoded JPEG string.

    Raises on any capture failure (no display, capture permission denied,
    ``mss``/Pillow import failure) — ``analyze_desktop_screen`` below
    catches broadly and reports ``{"ok": False, ...}`` rather than letting
    a screen-capture failure crash the whole ReAct turn.
    """
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        shot = sct.grab(monitor)

    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    img.thumbnail((_MAX_WIDTH, _UNBOUNDED_HEIGHT))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _candidate_providers() -> list[str]:
    """Local-first order — an independent, deliberately small copy of the
    same policy ``dana.plugins.vision.image_analysis._candidate_providers``
    already applies to its own VLM calls (see that module's docstring for
    why it isn't shared via import).
    """
    providers = ["ollama"]
    if cloud_fallback_enabled():
        cloud = cloud_provider_name()
        providers.append("openai" if cloud in {"gemini", "google", "anthropic"} else cloud)
    return providers


def analyze_desktop_screen(query: str, *, api_keys: dict[str, str] | None = None) -> dict[str, Any]:
    """Captures the user's primary monitor and asks the VLM ``query``
    about it via ``ModelProvider.complete_vision`` — the exact same vision
    bridge/local-first-fallback pattern
    ``dana.plugins.vision.image_analysis.analyze_workspace_image`` already
    uses for a sandboxed file, just with a live screen capture as the
    image source instead of a path on disk. Never raises — every failure
    mode (capture error, all VLM providers failing) comes back as
    ``{"ok": False, "error": ...}``.

    ``api_keys`` is the calling session's BYOK dict
    (``dana.api.server``'s ``session["api_keys"]``), threaded down via
    ``dana.core.react_dispatch``'s ``dispatch_tool_call`` the same way
    ``analyze_workspace_image``'s is (both are in ``_TOOLS_NEEDING_API_KEYS``).

    This is a HIGH-PRIVACY action, not merely a mutating one — capturing
    the user's actual desktop, not a sandboxed artifact — so this tool
    deliberately declares no ``"read_only": true`` in tools.json:
    ``dana.core.react_dispatch.is_mutating_tool``'s fail-closed check means
    it will never dispatch without an explicit human approval click,
    regardless of how harmless any individual capture actually is.
    """
    try:
        image_b64 = _capture_primary_monitor_jpeg_b64()
    except Exception as exc:  # noqa: BLE001 — capture failure must never crash the turn
        return {"ok": False, "error": f"screen capture failed: {exc}"}

    q = (query or "").strip() or "Describe what is shown on the screen."

    provider_client = ModelProvider(api_keys=api_keys)
    attempts: list[str] = []
    for candidate in _candidate_providers():
        try:
            description = provider_client.complete_vision(q, image_b64, mime_type="image/jpeg", provider=candidate)
        except Exception as exc:  # noqa: BLE001 — try the next candidate provider
            attempts.append(f"{candidate}: {exc}")
            continue
        if description.strip():
            return {"ok": True, "query": q, "description": description.strip()}
        attempts.append(f"{candidate}: empty response")

    return {"ok": False, "error": "all VLM providers failed", "attempts": attempts}


__all__ = ("analyze_desktop_screen",)
