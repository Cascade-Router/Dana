"""Parametric mesh generation + blueprint -> geometry-spec parsing for gr.Model3D.

Mesh generation is real (trimesh). Blueprint analysis calls a real multimodal
model (Anthropic Claude) when ANTHROPIC_API_KEY is configured as a Space
secret; otherwise it falls back to a deterministic, clearly-labeled heuristic
mock so the tab still demonstrates the JSON contract Dana's real
`analyze_cad_blueprint` tool (dana/tools/cad_vision.py) returns.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from typing import Any

import numpy as np
import trimesh
from PIL import Image

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

SUPPORTED_SHAPES = ("box", "cylinder", "box_with_hole")

DEFAULT_SPEC: dict[str, Any] = {
    "shape": "box_with_hole",
    "dims_mm": {"length": 60.0, "width": 40.0, "height": 20.0, "hole_radius": 8.0},
    "source": "default",
}


def _mesh_from_spec(spec: dict[str, Any]) -> trimesh.Trimesh:
    shape = spec.get("shape", "box")
    dims = spec.get("dims_mm", {})

    if shape == "cylinder":
        radius = float(dims.get("radius", 10))
        height = float(dims.get("height", 30))
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif shape == "box_with_hole":
        length = float(dims.get("length", 60))
        width = float(dims.get("width", 40))
        height = float(dims.get("height", 20))
        hole_radius = float(dims.get("hole_radius", 8))
        box = trimesh.creation.box(extents=[length, width, height])
        cylinder = trimesh.creation.cylinder(radius=hole_radius, height=height * 2.2)
        try:
            mesh = box.difference(cylinder)
        except BaseException:
            # boolean engine unavailable in this container — fall back to the box alone
            mesh = box
    else:  # "box"
        length = float(dims.get("length", 40))
        width = float(dims.get("width", 25))
        height = float(dims.get("height", 15))
        mesh = trimesh.creation.box(extents=[length, width, height])

    mesh.apply_translation(-mesh.centroid)
    return mesh


def spec_to_stl(spec: dict[str, Any]) -> str:
    """Generates an STL file on disk for the given geometry spec and returns its path."""
    mesh = _mesh_from_spec(spec)
    fd, path = tempfile.mkstemp(suffix=".stl", prefix="dana_cad_")
    os.close(fd)
    mesh.export(path)
    return path


def default_preview_stl() -> str:
    return spec_to_stl(DEFAULT_SPEC)


def _heuristic_mock_spec(image: Image.Image) -> dict[str, Any]:
    """Deterministic, size-driven mock — no ML involved, just a stand-in for the real call."""
    width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)

    if aspect > 1.4:
        shape = "box_with_hole"
        dims = {"length": round(80 * aspect, 1), "width": 50.0, "height": 18.0, "hole_radius": 9.0}
    elif aspect < 0.7:
        shape = "cylinder"
        dims = {"radius": 14.0, "height": round(90 / max(aspect, 0.3), 1)}
    else:
        shape = "box"
        dims = {"length": 45.0, "width": 45.0, "height": 25.0}

    return {
        "shape": shape,
        "dims_mm": dims,
        "entities_detected": ["outer_profile", "centerline" if shape != "box" else "corner_fillet"],
        "confidence": 0.42,
        "source": "heuristic_mock",
        "note": (
            "No ANTHROPIC_API_KEY configured for this Space — dimensions are a "
            "deterministic heuristic derived from image aspect ratio, not real "
            "vision inference. Set the secret to enable live multimodal analysis."
        ),
    }


_VLM_PROMPT = (
    "You are a CAD blueprint vision analyzer. Look at this engineering drawing or "
    "viewport screenshot and respond with ONLY a JSON object (no prose, no markdown "
    "fences) with this exact shape: "
    '{"shape": one of ["box","cylinder","box_with_hole"], '
    '"dims_mm": {relevant numeric dims for that shape}, '
    '"entities_detected": [short strings], "confidence": 0.0-1.0}. '
    "If the drawing shows something else, pick the closest of the three shapes and "
    "estimate proportional dims_mm. Return nothing except that JSON object."
)


def _call_anthropic_vision(image: Image.Image) -> dict[str, Any] | None:
    import httpx

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 400,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    resp = httpx.post(ANTHROPIC_URL, headers=headers, json=body, timeout=30.0)
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    parsed = json.loads(text)
    parsed["source"] = "anthropic_vision"
    return parsed


def parse_blueprint(image: Image.Image | None) -> dict[str, Any]:
    """Returns a geometry spec dict for the uploaded blueprint/viewport image.

    Tries a real multimodal call when a Space secret is configured; falls back
    to a labeled heuristic mock on missing key, network failure, or malformed
    model output so the tab never hard-fails.
    """
    if image is None:
        return DEFAULT_SPEC

    if ANTHROPIC_API_KEY:
        try:
            spec = _call_anthropic_vision(image)
            if spec.get("shape") in SUPPORTED_SHAPES and spec.get("dims_mm"):
                return spec
        except Exception as exc:
            mock = _heuristic_mock_spec(image)
            mock["note"] = f"Live vision call failed ({exc.__class__.__name__}) — showing heuristic mock instead."
            return mock

    return _heuristic_mock_spec(image)


def spec_to_json(spec: dict[str, Any]) -> str:
    return json.dumps(spec, indent=2)
