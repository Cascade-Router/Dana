"""CAD Blueprint capture + Visual QA for the AutoCAD Co-Pilot.

AutoCAD's COM API can tell us what it *thinks* it drew, but not what a
human actually sees on screen (wrong layer visibility, a failed
``SendCommand``, a stale viewport). These tools close that loop with a VLM:
screenshot the AutoCAD viewport, ask a vision model to read the geometry
back off pixels, and diff that against the spec that drove
``dana.operators.autocad_engine``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from dana.core.model_provider import ModelProvider, cloud_fallback_enabled, cloud_provider_name
from dana.paths import CAPTURES_DIR
from dana.tools.os_control import (
    capture_screen_png_bytes,
    capture_window_png_bytes,
    get_active_windows,
)

_CAD_WINDOW_HINTS = ("autocad", "acad", "freecad")

_BLUEPRINT_PROMPT = (
    "You are reading an AutoCAD viewport screenshot. Identify every visible "
    "geometric entity (lines, circles, polylines, solids), each entity's "
    "approximate coordinates or dimensions as shown by any visible "
    "annotations/rulers, its layer if labeled, and any dimension/text "
    "callouts. Respond with ONLY a JSON object of this shape: "
    '{"entities": [{"type": str, "layer": str|null, "coords_or_dims": str, '
    '"notes": str}], "summary": str}. No prose outside the JSON.'
)


def _find_cad_window() -> dict[str, Any] | None:
    for win in get_active_windows():
        title = str(win.get("title") or "").lower()
        if any(hint in title for hint in _CAD_WINDOW_HINTS):
            return win
    return None


def capture_cad_viewport(*, save_copy: bool = True) -> dict[str, Any]:
    """Screenshot targeted at the AutoCAD/FreeCAD window's own bounding box.

    Zero-focus workspace: this NEVER calls ``set_foreground_window`` or any
    other focus-stealing API. ``get_window_rect`` + a region-scoped ``mss``
    grab work whether the window is active, in the background, or sitting
    on a secondary monitor while a fullscreen app owns the primary one —
    the only requirement is that nothing else is drawn on top of it. Falls
    back to a full-primary-monitor capture only when no CAD window exists
    at all (nothing to target).
    """
    window = _find_cad_window()

    try:
        if window is not None:
            png = capture_window_png_bytes(int(window["hwnd"]))
        else:
            png = capture_screen_png_bytes()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"capture failed: {exc}"}

    path: str | None = None
    if save_copy:
        try:
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            out = CAPTURES_DIR / "last_cad_viewport.png"
            out.write_bytes(png)
            path = str(out)
        except Exception:  # noqa: BLE001
            path = None

    return {
        "ok": True,
        "png_bytes": png,
        "path": path,
        "window_found": window is not None,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction — VLMs love wrapping JSON in prose/fences."""
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fence.group(1) if fence else stripped
    parsed = _try_parse_json_object(candidate)
    if parsed is not None:
        return parsed
    brace = re.search(r"\{.*\}", candidate, re.DOTALL)
    return _try_parse_json_object(brace.group(0)) if brace else None


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_to_base64(image_path_or_base64: str) -> str | None:
    value = (image_path_or_base64 or "").strip()
    if not value:
        return None
    # Long base64-alphabet strings are treated as inline image data.
    if len(value) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value):
        return re.sub(r"\s+", "", value)
    try:
        path = Path(value)
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        pass
    return None


def _candidate_vlm_providers(explicit: str | None) -> list[str]:
    """Local-first provider order: Ollama VLM, then cloud only if allowed."""
    if explicit:
        return [explicit.strip().lower()]
    providers = ["ollama"]
    if cloud_fallback_enabled():
        cloud = cloud_provider_name()
        providers.append("openai" if cloud in {"gemini", "google", "anthropic"} else cloud)
    return providers


def analyze_cad_blueprint(
    image_path_or_base64: str, *, provider: str | None = None, api_key: str | None = None
) -> str:
    """VLM read of a CAD screenshot into structured JSON (entities/layers/dims).

    Tries the local Ollama vision model (Qwen2.5-VL-class, zero cost, zero
    data egress) first, falling back to a cloud OpenAI-compatible VLM
    (GPT-4o-class) only when local analysis errors or returns no parseable
    JSON, and only if ``DANA_ALLOW_CLOUD_FALLBACK`` is set — mirrors the
    local-first policy already used by ``dana.core.model_provider``.

    ``api_key`` is a BYOK override (from the frontend's SecretsMenu, threaded
    down through ``dana.core.react_dispatch.build_visual_inspection_result``)
    for whichever cloud provider ends up being tried — ``_candidate_vlm_providers``
    maps every cloud fallback except an explicit non-OpenAI-schema override to
    ``"openai"``, so this is stored under that key; ``ModelProvider`` falls
    back to ``OPENAI_API_KEY`` on its own when this is ``None``.
    """
    image_b64 = _resolve_to_base64(image_path_or_base64)
    if image_b64 is None:
        return json.dumps({"ok": False, "error": "could not read image_path_or_base64"})

    provider_client = ModelProvider(api_keys={"openai": api_key} if api_key else None)
    attempts: list[str] = []
    for candidate in _candidate_vlm_providers(provider):
        try:
            text = provider_client.complete_vision(_BLUEPRINT_PROMPT, image_b64, provider=candidate)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{candidate}: {exc}")
            continue
        parsed = _extract_json_object(text)
        if parsed is not None:
            parsed.setdefault("ok", True)
            parsed["provider"] = candidate
            return json.dumps(parsed)
        attempts.append(f"{candidate}: non-JSON response")
    return json.dumps({"ok": False, "error": "all VLM providers failed", "attempts": attempts})


def verify_cad_rendering(expected_spec_json: str) -> str:
    """Capture the live viewport and check it against ``expected_spec_json``.

    ``expected_spec_json`` uses the same ``{"entities": [...]}`` shape
    ``analyze_cad_blueprint`` returns. This is a coarse visual QA pass
    (entity-type presence, not pixel-exact geometry) — VLM coordinate reads
    off a screenshot are approximate by nature, so treat a low
    ``match_ratio`` as "investigate", not "definitely wrong".
    """
    try:
        expected = json.loads(expected_spec_json or "{}")
    except (json.JSONDecodeError, ValueError):
        return json.dumps({"ok": False, "error": "expected_spec_json is not valid JSON"})

    capture = capture_cad_viewport()
    if not capture.get("ok"):
        return json.dumps({"ok": False, "error": capture.get("error")})

    image_b64 = base64.b64encode(capture["png_bytes"]).decode("ascii")
    try:
        analysis = json.loads(analyze_cad_blueprint(image_b64))
    except (json.JSONDecodeError, ValueError):
        analysis = {"ok": False, "error": "analysis returned non-JSON"}

    if not analysis.get("ok"):
        return json.dumps({"ok": False, "error": "viewport analysis failed", "detail": analysis})

    expected_types = [str(e.get("type") or "").lower() for e in (expected.get("entities") or [])]
    found_types = [str(e.get("type") or "").lower() for e in (analysis.get("entities") or [])]
    matched = sum(1 for t in expected_types if t in found_types)

    return json.dumps(
        {
            "ok": True,
            "expected_count": len(expected_types),
            "found_count": len(found_types),
            "matched_types": matched,
            "match_ratio": (matched / len(expected_types)) if expected_types else None,
            "analysis": analysis,
            "path": capture.get("path"),
        }
    )


__all__ = ("analyze_cad_blueprint", "capture_cad_viewport", "verify_cad_rendering")
