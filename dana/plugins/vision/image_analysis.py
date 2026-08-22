"""Workspace image vision — lets the agent "see" its own sandboxed image
artifacts (e.g. a matplotlib chart written by run_python_script's os_tools
capability) via the existing ModelProvider vision bridge, dispatched
directly from the normal ReAct tool-calling path. Unlike
take_canvas_screenshot's suspend/resume round-trip (which needs the LIVE
R3F canvas in the Tauri frontend), there's nothing to wait for here — the
image is already a file on disk, so this dispatches synchronously like any
other tool.
"""

from __future__ import annotations

import base64
from typing import Any

from dana.core.model_provider import ModelProvider, cloud_fallback_enabled, cloud_provider_name
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

# suffix -> MIME type; also doubles as the allowlist of supported image types.
_ALLOWED_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _candidate_providers() -> list[str]:
    """Local-first order — the same policy dana.tools.cad_vision.
    analyze_cad_blueprint already applies to its own VLM calls. This is a
    deliberately small (~5-line), independent copy of that policy, not a
    reimplementation of the actual VLM call itself — ModelProvider.
    complete_vision below is the one and only thing that talks to a model;
    cad_vision.py's own candidate-list helper is private to its blueprint-
    reading flow, so it's not imported from here.
    """
    providers = ["ollama"]
    if cloud_fallback_enabled():
        cloud = cloud_provider_name()
        providers.append("openai" if cloud in {"gemini", "google", "anthropic"} else cloud)
    return providers


def analyze_workspace_image(
    file_path: str, query: str, *, api_keys: dict[str, str] | None = None
) -> dict[str, Any]:
    """Reads a sandboxed image, base64-encodes it, and asks the VLM
    ``query`` about it via ``ModelProvider.complete_vision`` — reusing that
    existing vision bridge exactly (same HTTP/schema logic
    analyze_cad_blueprint's own VLM calls go through), not a
    reimplementation. Read-only; never mutates anything, never raises —
    every failure mode comes back as ``{"ok": False, "error": ...}``.

    ``api_keys`` is the calling session's BYOK dict (dana.api.server's
    session["api_keys"]) — threaded down via dana.core.react_dispatch's
    dispatch_tool_call (extended just for this tool), since this runs on
    the ordinary ReAct dispatch path, which doesn't otherwise carry session
    state to a tool handler the way the take_canvas_screenshot suspend
    path already does for build_visual_inspection_result.
    """
    try:
        target = resolve_sandboxed_path(file_path)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}

    suffix = target.suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return {
            "ok": False,
            "error": f"only {sorted(_ALLOWED_SUFFIXES)} images are supported, got: {file_path!r}",
        }
    if not target.exists():
        return {"ok": False, "error": f"image does not exist: {file_path!r}"}
    if not target.is_file():
        return {"ok": False, "error": f"path is not a file: {file_path!r}"}

    try:
        raw_bytes = target.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"could not read image: {exc}"}

    image_b64 = base64.b64encode(raw_bytes).decode("ascii")
    mime_type = _ALLOWED_SUFFIXES[suffix]
    q = (query or "").strip() or "Describe what is shown in this image."

    provider_client = ModelProvider(api_keys=api_keys)
    attempts: list[str] = []
    for candidate in _candidate_providers():
        try:
            description = provider_client.complete_vision(q, image_b64, mime_type=mime_type, provider=candidate)
        except Exception as exc:  # noqa: BLE001 — try the next candidate provider
            attempts.append(f"{candidate}: {exc}")
            continue
        if description.strip():
            return {"ok": True, "path": file_path, "query": q, "description": description.strip()}
        attempts.append(f"{candidate}: empty response")

    return {"ok": False, "error": "all VLM providers failed", "attempts": attempts}


__all__ = ("analyze_workspace_image",)
