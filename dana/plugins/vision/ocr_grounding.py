"""EasyOCR-based deterministic text grounding for CAD reference drawings — a
hard numeric floor under analyze_reference_design's VLM extraction
(dana.plugins.vision.image_analysis), not a replacement for it. OCR reads
exact printed characters off a drawing; it has no idea whether a shape is a
box or a cylinder, or how primitives relate to each other, so this only
ever supplements the VLM's own two passes with numbers it can trust,
never substitutes for the VLM's structural reasoning.

Only useful once the source image actually has printed dimension text on
it — confirmed live earlier in this project's own investigation that
scripts/generate_ortho_tests.py's synthetic drawings were originally blank
line art with nothing printed on them at all (EasyOCR correctly returned
[] against them), before that script's own later dimension-stamping fix.
On any image with no legible text, extract_blueprint_dimensions below
still returns {} correctly — this is expected, not a failure.

Per-view, not a flat bag: confirmed live that a flat, deduplicated list of
every number found across every file gives a VLM no way to tell "60.0" from
the front view apart from "60.0" from the top view, or to know which
number is width vs. height at all — a live 3-view test showed the model
just smearing loose numbers into unrelated/fabricated primitives instead.
Keying by inferred view name (front/top/right/...) at least tells the
prompt WHICH projection each number came from, matching standard
orthographic convention (front -> width/height, top -> width/depth, etc.).
"""

from __future__ import annotations

import re
from pathlib import Path

from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

_reader = None

# Recognized orthographic/standard view names -- checked as whole filename-
# stem tokens (split on _/-/space), case-insensitively. Covers both this
# project's own synthetic naming convention (scripts/generate_ortho_tests.py:
# bracket_front.png, shaft_side.png, ...) and a plausible real user upload
# named the same way (e.g. "front_view.jpg" from the chat UI's own example
# in dana.core.react_dispatch's tool-routing guidance).
_VIEW_KEYWORDS = ("front", "back", "rear", "top", "bottom", "left", "right", "side", "iso", "isometric")

# After the space->decimal repair below, a token is kept only if it looks
# like an actual dimension: a plain decimal number, or a radius/diameter/
# thread-callout prefix followed by digits. A bare "8" or "3" with no
# decimal point and no prefix is exactly the kind of stray OCR fragment
# confirmed live on a real synthetic drawing (EasyOCR misreading a rotated
# label) -- letting it through just hands the VLM more noise to smear into
# a fabricated primitive, which is what a looser filter (any digit at all)
# was confirmed to do.
_DIMENSION_PATTERN = re.compile(r"^(?:\d+\.\d+|[RØMrøm]\s?\d+(?:\.\d+)?)$")


def _infer_view_label(path: Path, index: int) -> str:
    """Best-effort view name from the filename stem, falling back to a
    positional label -- NOT assumed to always succeed. A real chat-upload
    file (arbitrary user-chosen name, e.g. "IMG_1234.jpg") won't match any
    _VIEW_KEYWORDS token, and guessing wrong here (e.g. calling a random
    upload "FRONT") would be worse than an honest positional label: it
    would tell the VLM's prompt a confident-sounding but false thing about
    which projection axis a number belongs to.
    """
    tokens = re.split(r"[_\-\s]+", path.stem.lower())
    for token in tokens:
        if token in _VIEW_KEYWORDS:
            return token.upper()
    return f"VIEW_{index + 1}"


def _repair_and_filter(raw_text: str) -> str | None:
    """Fixes EasyOCR's confirmed-live space-for-decimal-point misread
    ("60 0" -> "60.0" -- observed on this project's own rotated dimension
    labels) and returns None for anything that still doesn't look like a
    real dimension after repair, rather than passing raw OCR noise through.
    """
    cleaned = raw_text.strip()
    repaired = re.sub(r"(\d)\s+(\d)", r"\1.\2", cleaned)
    return repaired if _DIMENSION_PATTERN.match(repaired) else None


class _Unavailable:
    """Sentinel cached in place of a real Reader when easyocr itself isn't
    installed, so _get_reader() only ever tries the (slow, error-prone)
    import once per process instead of retrying it on every call.
    """


def _get_reader():
    """Lazily constructs and caches ONE easyocr.Reader per process.

    Instantiating a Reader loads (and, on first-ever run, downloads) the
    detection + recognition models — confirmed live this takes real,
    non-trivial time. Rebuilding it on every extract_blueprint_dimensions
    call (i.e. every analyze_reference_design call, inside the ReAct loop)
    would make each one pay that cost again; a module-level singleton pays
    it once per process instead.

    ``gpu=False`` explicitly: confirmed live this environment's own torch
    build is CPU-only (``2.13.0+cpu``, no CUDA), and this project's own
    requirements.txt deliberately never pins torch/CUDA wheels here at all
    (see that file's own comment) — requesting GPU unconditionally would be
    a no-op at best on a CPU-only install and a needless surprise if a
    future CUDA install changes that silently.

    ``easyocr`` IS a normal requirements.txt dependency (verified live it
    never touches torch/torchvision there — its own declared requirement
    is a bare, unpinned ``torch`` plus ``torchvision>=0.5``, confirmed via
    ``pip install easyocr --dry-run`` reporting every torch-derived
    package as already satisfied). This ``ImportError`` guard is defensive
    depth, not the primary mechanism keeping it optional — a venv that
    hasn't run a fresh ``pip install -r requirements.txt`` yet, or any
    other environment missing it for whatever reason, still gets no OCR
    grounding rather than an ImportError crashing analyze_reference_design
    entirely.
    """
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError:
            _reader = _Unavailable()
        else:
            _reader = easyocr.Reader(["en"], gpu=False)
    return None if isinstance(_reader, _Unavailable) else _reader


def extract_blueprint_dimensions(file_paths: list[str]) -> dict[str, list[str]]:
    """Runs EasyOCR against each sandboxed source image and returns a
    ``{view_name: [dimension, ...]}`` mapping — one entry per input path,
    keyed by its inferred view name (see _infer_view_label), value sorted
    and deduplicated WITHIN that view (dimensions can legitimately repeat
    across different views, e.g. a shared width, so dedup is per-view only,
    never across the whole call the way an earlier flat-list version did).

    Reads the ORIGINAL per-view files, not analyze_reference_design's own
    stitched composite grid — full-resolution originals avoid any blur or
    detail loss the grid compositing (or a VLM-facing downscale) could
    introduce, and OCR has no need for the views to be combined into one
    image the way the VLM call does.

    Best-effort and never raises: a missing/unreadable/out-of-sandbox path
    just gets an empty list at its own view key, not a fatal error — this
    is a supplementary grounding signal for analyze_reference_design's
    prompt, not a required input, so one bad path must never block the two
    VLM passes it feeds into. Returns {} immediately if easyocr isn't
    installed in this environment (see _get_reader) rather than raising.
    """
    reader = _get_reader()
    if reader is None:
        return {}
    view_dimensions: dict[str, list[str]] = {}
    for index, file_path in enumerate(file_paths):
        view_name = _infer_view_label(Path(file_path), index)
        try:
            target = resolve_sandboxed_path(file_path)
        except PathEscapeError:
            view_dimensions[view_name] = []
            continue
        if not target.is_file():
            view_dimensions[view_name] = []
            continue
        try:
            raw_text = reader.readtext(str(target), detail=0)
        except Exception:  # noqa: BLE001 — OCR is best-effort grounding, never fatal
            view_dimensions[view_name] = []
            continue

        valid_dims: set[str] = set()
        for text in raw_text:
            repaired = _repair_and_filter(text)
            if repaired is not None:
                valid_dims.add(repaired)
        view_dimensions[view_name] = sorted(valid_dims)
    return view_dimensions


__all__ = ("extract_blueprint_dimensions",)
