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
import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from dana.core.model_provider import ModelProvider, cloud_fallback_enabled, cloud_provider_name
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path
from dana.plugins.vision.ocr_grounding import extract_blueprint_dimensions

logger = logging.getLogger(__name__)

# suffix -> MIME type; also doubles as the allowlist of supported image types.
_ALLOWED_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Two-pass CSG extraction, replacing a single combined-schema prompt: asking
# llava:7b for bounding-box + primitives + relationships + joints in ONE
# response (confirmed live) let it complete the first couple sections
# correctly and then fall into an alternating-value repetition loop
# (radius=10mm/radius=5mm forever) trying to enumerate a long primitives
# list, never reaching the later sections at all. Splitting the primitives
# extraction (pass 1) from the relationships/joints extraction (pass 2,
# fed pass 1's own JSON back as context) gives the model two much shorter
# generations instead of one long one, and JSON-with-a-fixed-schema instead
# of free markdown gives analyze_reference_design's caller (create_plan)
# something machine-parseable rather than text it has to re-summarize.
#
# No "bounding_box_mm" in pass 1's own schema anymore -- confirmed live
# across every real run this session that llava:7b reports it as a hard
# {0,0,0} regardless of what else improved (two-pass splitting, OCR
# grounding, label legibility), a persistent model behavior on this
# specific field, not an input-quality problem. analyze_reference_design
# now computes the bounding box deterministically from OCR's own per-view
# numbers instead (see _compute_deterministic_bbox) and never asks the VLM
# to guess it at all.
_CSG_PASS1_PROMPT = (
    "You are an autonomous CAD blueprint analyzer viewing one or more orthographic "
    "projections of a mechanical part (a single view, or a composite grid of "
    "multiple views on a white background — treat each panel as one standard "
    "view, e.g. front/top/side, not a separate object).\n\n"
    "Extract the primary geometric primitives. Cross-reference multiple views to "
    "resolve depth a single view can't show, and read any dashed/dotted "
    "hidden-line detail. These are unscaled reference drawings: extract the "
    "topological primitive shapes and their relative proportions.\n"
    "{ocr_context}\n\n"
    "Output ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "primitives": [{"id": str, "type": "box|cylinder|sphere", "dimensions": '
    '{"length": float, "width": float, "height": float, "radius": float}}]\n'
    "}\n"
    "Stop generation immediately after closing the JSON object."
)

_CSG_PASS2_PROMPT = (
    "You are an autonomous CAD blueprint analyzer viewing the same orthographic "
    "projection(s) again.\n\n"
    "Given these primitives previously extracted from the drawing:\n"
    "{primitives_json}\n\n"
    "Analyze the visual drawing again and determine their spatial relationships "
    "and kinematic joints.\n"
    "Output ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "relationships": [{"parent_id": str, "child_id": str, "attachment_type": '
    '"face_to_face|edge_to_edge|concentric", "offset": {"x": float, "y": float, '
    '"z": float}}],\n'
    '  "joints": [{"parent_id": str, "child_id": str, "type": '
    '"fixed|revolute|prismatic", "axis": [float, float, float]}]\n'
    "}\n"
    "Stop generation immediately after closing the JSON object."
)


def _stitch_images(image_paths: list[Path]) -> str:
    """Composites 2+ orthographic views into one grid image (white
    background, one source image per cell) and returns it as a single
    base64-encoded PNG.

    Passing every view as its own ``image_url`` block to complete_vision
    (dana.core.model_provider's OpenAI-vision bridge, built on
    dana.core.openai_tool_bridge.build_multimodal_messages) is what
    produced a degenerate all-dashes repetition loop confirmed live
    against ``llava:7b`` — a single image already works reliably (see the
    single-file branch in analyze_reference_design below); multiple
    stacked image blocks in one message do not. Stitching first means
    complete_vision always receives exactly one image, the same code path
    that already works, regardless of how many views the caller supplied.

    A 1x2 grid for 2 images, otherwise a 2x2 grid (matching the multi-view
    sets this pipeline actually sees — front/top/side, at most
    MAX_ATTACHMENTS=4 in the chat UI) with any leftover cell left blank.
    Cell size is the max width/height across the input images rather than a
    forced resize, so no view gets stretched or squashed to fit; a source
    image with an alpha channel is composited onto the white cell
    background via its own alpha mask (not just have the channel dropped),
    so a transparent PNG doesn't turn into an unpredictable dark patch on
    the grid.
    """
    images = [Image.open(p) for p in image_paths]
    try:
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)
        count = len(images)
        cols = min(count, 2)
        rows = -(-count // cols)  # ceil division

        canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
        for index, img in enumerate(images):
            col, row = index % cols, index // cols
            x = col * cell_w + (cell_w - img.width) // 2
            y = row * cell_h + (cell_h - img.height) // 2
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                canvas.paste(rgba, (x, y), rgba)
            else:
                canvas.paste(img.convert("RGB"), (x, y))

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        for img in images:
            img.close()


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


def _clean_json_output(text: str) -> str:
    """Strips an Ollama/llava-style ```json fenced code block (or a bare ```
    fence) around a JSON object. llava:7b routinely ignores "Output ONLY
    valid JSON" and wraps the answer in a code fence anyway, so a raw
    ``json.loads`` fails even when the JSON itself is well-formed. Also
    slices to the outermost ``{``/``}`` as a second line of defense against
    any stray prose the model still prepends/appends outside the fence.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        closing = cleaned.rfind("```")
        if closing != -1:
            cleaned = cleaned[:closing]
        cleaned = cleaned.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _extract_json_pass(
    provider_client: ModelProvider, prompt: str, image_b64: str, mime_type: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Runs one VLM pass across every candidate provider (same local-first/
    cloud-fallback order as analyze_workspace_image's own loop), parsing
    each response as JSON via _clean_json_output.

    Retries the next candidate not just on a transport exception or empty
    response but also on a JSON decode failure — llava:7b is exactly the
    model this two-pass split exists to route around, so a malformed
    response from it is this loop's expected failure mode, not a fatal one,
    as long as a cloud fallback (or another local vision model) is
    configured.
    """
    attempts: list[str] = []
    for candidate in _candidate_providers():
        try:
            raw = provider_client.complete_vision(prompt, image_b64, mime_type=mime_type, provider=candidate)
        except Exception as exc:  # noqa: BLE001 — try the next candidate provider
            attempts.append(f"{candidate}: {exc}")
            continue
        if not raw.strip():
            attempts.append(f"{candidate}: empty response")
            continue
        try:
            return json.loads(_clean_json_output(raw)), attempts
        except json.JSONDecodeError as exc:
            logger.warning("analyze_reference_design: VLM JSON parse failure (candidate=%s): %s", candidate, exc)
            attempts.append(f"{candidate}: JSON decode failed: {exc} (raw: {raw[:200]!r})")
            continue
    return None, attempts


def _build_ocr_context(dimensions_by_view: dict[str, list[str]]) -> str:
    """Formats extract_blueprint_dimensions's per-view OCR output into the
    pass-1 prompt's {ocr_context} slot. With real detections, tells the VLM
    to treat them as a hard numeric floor rather than estimating, grouped
    by view name so it has a shot at mapping numbers onto the right axis —
    a flat, ungrouped list was confirmed live to make the model smear loose
    numbers into unrelated/fabricated primitives instead. With none (e.g.
    easyocr unavailable, or the drawing genuinely has no printed dimension
    text, or every view's own list ended up empty — all real, confirmed-live
    cases, not just hypotheticals), falls back to the original guidance not
    to guess exact millimeter tolerances rather than silently omitting the
    instruction.
    """
    ocr_lines = [f"- {view} VIEW dimensions: {', '.join(dims)}" for view, dims in dimensions_by_view.items() if dims]
    if not ocr_lines:
        return "Do not attempt to guess exact millimeter tolerances."
    return (
        "### HARD DIMENSIONAL CONSTRAINTS ###\n"
        "The following dimensions were deterministically extracted via OCR, "
        "grouped by projection view. Map these values to the correct X/Y/Z "
        "axes based on standard orthographic projection rules (e.g. the FRONT "
        "view gives width/height, the TOP view gives width/depth). You MUST "
        "use these exact numerical values for the bounding box and primitives "
        "wherever they plausibly correspond to a dimension you are "
        "extracting:\n" + "\n".join(ocr_lines)
    )


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


def _is_degenerate_dimensions(values: Any) -> bool:
    """True if ``values`` (a bounding-box or primitive-dimensions dict) is
    empty, missing, or every entry is zero/non-numeric.

    VLM JSON is untrusted output, not a validated contract — a stray string
    or nested object in a dimension field is exactly as useless downstream
    as an explicit ``0.0``, so both fail this gate the same way rather than
    crashing analyze_reference_design with an uncaught ValueError/TypeError
    on ``float(v)``.
    """
    if not values:
        return True
    entries = values.values() if isinstance(values, dict) else values
    for v in entries:
        try:
            if float(v) != 0.0:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _compute_deterministic_bbox(dimensions_by_view: dict[str, list[str]]) -> dict[str, float]:
    """Computes the overall bounding box from OCR's own per-view numbers
    directly, in Python — never asks the VLM for it (see _CSG_PASS1_PROMPT's
    own comment on why: a confirmed-live persistent {0,0,0} regardless of
    every other input-quality fix this session).

    Best case — all three of FRONT/TOP/RIGHT present (the standard 3-view
    orthographic set this pipeline's own synthetic test generator produces):
    exploits that each pair of adjacent views shares exactly one axis's
    measurement under standard projection convention —
        width  (X) is shown by both FRONT and TOP  -> shared_x = FRONT ∩ TOP
        height (Z) is shown by both FRONT and RIGHT -> shared_z = FRONT ∩ RIGHT
        depth  (Y) is shown by both TOP and RIGHT   -> shared_y = TOP ∩ RIGHT
    Verified live against the real 3-view L-bracket OCR output
    (FRONT=[50.0,60.0], TOP=[40.0,60.0], RIGHT=[40.0,50.0]) — this reproduces
    the part's true 60x40x50mm bounding box exactly.

    A part whose width happens to numerically equal an unrelated depth/height
    (a real but rare case) could make an intersection pick the wrong shared
    value; this is a best-effort deterministic estimate, not a proof, which
    is exactly why _validate_numerical_integrity still runs after it rather
    than trusting bounding-box computation unconditionally.

    Fallback (view names don't include the full FRONT/TOP/RIGHT triple —
    e.g. a 2-view part, or real user-uploaded files that _infer_view_label
    couldn't recognize a keyword in): the 3 largest distinct numbers seen
    across every view, descending — weaker (no real axis correspondence,
    just magnitude order), but still real, non-zero OCR data rather than a
    guess.
    """
    all_dims: list[float] = []
    for dims in dimensions_by_view.values():
        for d in dims:
            try:
                all_dims.append(float(d))
            except (TypeError, ValueError):
                continue

    if not all_dims:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    unique_dims = sorted(set(all_dims), reverse=True)
    x = unique_dims[0]
    y = unique_dims[1] if len(unique_dims) > 1 else x
    z = unique_dims[2] if len(unique_dims) > 2 else y

    front = {float(v) for v in dimensions_by_view.get("FRONT", [])}
    top = {float(v) for v in dimensions_by_view.get("TOP", [])}
    right = {float(v) for v in dimensions_by_view.get("RIGHT", [])}

    if front and top and right:
        shared_x = front & top
        shared_z = front & right
        shared_y = top & right
        if shared_x and shared_y and shared_z:
            x, y, z = max(shared_x), max(shared_y), max(shared_z)

    return {"x": x, "y": y, "z": z}


def _validate_numerical_integrity(blueprint: dict[str, Any]) -> tuple[bool, str]:
    """Hard gate run before analyze_reference_design ever reports
    ``ok: True`` — a VLM hallucinating STRUCTURALLY valid JSON (parses fine,
    content is wrong/empty) is a confirmed-live failure mode of llava:7b on
    these unscaled synthetic drawings (see this module's two-pass rewrite
    above), and an all-zero bounding box or primitive would silently
    corrupt any downstream scaling/CAD-generation step that trusted it at
    face value instead of erroring loudly here.

    Also checks relationship/joint referential integrity — confirmed live
    that pass 2 can reference a parent_id/child_id that doesn't exist in
    pass 1's own primitives list at all (a hallucinated id), which none of
    the dimension checks above would ever catch since it isn't a dimension
    problem.
    """
    bbox = blueprint.get("bounding_box") or {}
    if _is_degenerate_dimensions(bbox):
        return False, f"Degenerate bounding box detected: {bbox}"

    primitives = blueprint.get("primitives") or []
    if not primitives:
        return False, "No primitives extracted"

    valid_ids: set[str] = set()
    for prim in primitives:
        prim_id = str(prim.get("id", ""))
        if prim_id:
            valid_ids.add(prim_id)
        dims = prim.get("dimensions") or {}
        if _is_degenerate_dimensions(dims):
            return False, f"Degenerate primitive dimensions detected in {prim_id}: {dims}"

    for rel in blueprint.get("relationships") or []:
        if str(rel.get("parent_id")) not in valid_ids or str(rel.get("child_id")) not in valid_ids:
            return False, f"Dangling relationship reference detected: {rel}"

    for joint in blueprint.get("joints") or []:
        if str(joint.get("parent_id")) not in valid_ids or str(joint.get("child_id")) not in valid_ids:
            return False, f"Dangling joint reference detected: {joint}"

    return True, ""


def analyze_reference_design(
    file_paths: list[str], *, api_keys: dict[str, str] | None = None
) -> dict[str, Any]:
    """Reads one or more sandboxed reference images and asks the VLM to reverse
    them into a fixed-schema JSON CAD blueprint (bounding box / primitives /
    spatial relationships / joints), instead of ``analyze_workspace_image``'s
    open-ended ``query`` — the point here is a deterministic, machine-parsed
    shape the next ReAct turn's ``create_plan`` call can read primitives and
    mates off of, not a caption.

    Two VLM passes, not one: asking for the full schema in a single response
    (confirmed live) let llava:7b complete a couple of primitives correctly,
    then fall into an alternating-value repetition loop trying to enumerate
    a long primitives list, never reaching relationships/joints at all. Pass
    1 (``_CSG_PASS1_PROMPT``) extracts just the primitives; pass 2
    (``_CSG_PASS2_PROMPT``) is given pass 1's own JSON back as context and
    asked only for relationships + joints — two much shorter generations
    instead of one long one. Each pass retries across every candidate
    provider independently (see ``_extract_json_pass``).

    The bounding box is NOT asked of the VLM at all (pass 1's own schema has
    no such field) — confirmed live across every real run this session that
    llava:7b reports it as a hard ``{0,0,0}`` regardless of what else
    improved, a persistent model behavior on this one field. It's computed
    deterministically instead, from OCR's own per-view numbers (see
    ``_compute_deterministic_bbox``), which — verified live — reproduces a
    real part's true bounding box exactly when a full FRONT/TOP/RIGHT view
    triple is present.

    ``file_paths`` takes more than one path when the user supplied multiple
    orthographic views (front/top/side) of the same part — a single 2D image
    is depth-ambiguous. Multiple views are composited into one grid image
    (``_stitch_images``) before being sent, rather than passed as separate
    ``image_url`` blocks in the same message — see that function's docstring
    for the confirmed-live degenerate-output failure that caused this; that
    same composite is reused for both passes. Shares every other plumbing/
    safety detail with ``analyze_workspace_image`` (sandboxed path
    resolution, suffix allowlist, local-first VLM fallback, BYOK
    ``api_keys`` threading) — see that function's docstring for why each of
    those exists; read-only, never raises.

    Both passes returning parseable JSON is necessary but not sufficient —
    a VLM can hallucinate a structurally valid but numerically/referentially
    empty result (confirmed live: an invented primitive with all-zero
    dimensions, a relationship/joint referencing a primitive id that
    doesn't exist in pass 1's own output). ``_validate_numerical_integrity``
    runs as a hard gate before this ever reports ``ok: True``, so a caller
    building geometry from ``blueprint`` never has to separately re-check
    for degenerate values or dangling references.

    Pass 1's prompt also gets a deterministic OCR floor
    (``ocr_grounding.extract_blueprint_dimensions``, run against the
    original per-view files): when the drawing actually has legible
    printed dimensions, those exact strings are handed to the VLM as hard
    constraints instead of letting it estimate; when there's nothing
    legible (or easyocr isn't installed), pass 1 falls back to the
    original "don't guess exact millimeter tolerances" guidance.
    """
    if not file_paths:
        return {"ok": False, "error": "at least one file_path is required"}

    targets: list[Path] = []
    for file_path in file_paths:
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

        targets.append(target)

    # Single view: send it as-is, exactly like analyze_workspace_image above
    # (this is the code path already confirmed to return coherent output).
    # Multiple views: stitch into one composite grid rather than passing
    # each as its own image_url block — see _stitch_images for why (a
    # confirmed-live llava:7b degenerate repetition loop on stacked images).
    if len(targets) == 1:
        try:
            raw_bytes = targets[0].read_bytes()
        except OSError as exc:
            return {"ok": False, "error": f"could not read image: {exc}"}
        image_b64 = base64.b64encode(raw_bytes).decode("ascii")
        mime_type = _ALLOWED_SUFFIXES[targets[0].suffix.lower()]
    else:
        try:
            image_b64 = _stitch_images(targets)
        except Exception as exc:  # noqa: BLE001 — a malformed/corrupt source image
            return {"ok": False, "error": f"could not composite views: {exc}"}
        mime_type = "image/png"

    # Deterministic OCR floor under pass 1 -- reads the ORIGINAL per-view
    # files (not the composite built above), since full-resolution
    # originals give EasyOCR the best chance at legible text. Best-effort:
    # never raises, returns {} if easyocr isn't installed or the drawing
    # has no printed text at all (see ocr_grounding.py's own docstring).
    # Computed once and reused for both the pass-1 prompt context AND the
    # bounding box below -- the same OCR numbers ground both.
    dimensions_by_view = extract_blueprint_dimensions(file_paths)
    prompt_1 = _CSG_PASS1_PROMPT.replace("{ocr_context}", _build_ocr_context(dimensions_by_view))

    provider_client = ModelProvider(api_keys=api_keys)

    pass_1_data, attempts_1 = _extract_json_pass(provider_client, prompt_1, image_b64, mime_type)
    if pass_1_data is None:
        return {"ok": False, "error": "pass 1 (bounding box/primitives) failed", "attempts": attempts_1}

    prompt_2 = _CSG_PASS2_PROMPT.replace("{primitives_json}", json.dumps(pass_1_data, indent=2))
    pass_2_data, attempts_2 = _extract_json_pass(provider_client, prompt_2, image_b64, mime_type)
    if pass_2_data is None:
        return {
            "ok": False,
            "error": "pass 2 (spatial relationships/joints) failed",
            "attempts": attempts_2,
            "primitives": pass_1_data,
        }

    blueprint = {
        "bounding_box": _compute_deterministic_bbox(dimensions_by_view),
        "primitives": pass_1_data.get("primitives", []),
        "relationships": pass_2_data.get("relationships", []),
        "joints": pass_2_data.get("joints", []),
    }

    # Hard failure on a structurally-valid-but-numerically-degenerate
    # result (e.g. an all-zero bounding box) — see _validate_numerical_integrity.
    is_valid, error_msg = _validate_numerical_integrity(blueprint)
    if not is_valid:
        logger.warning("analyze_reference_design: blueprint validation failed: %s", error_msg)
        return {"ok": False, "error": error_msg, "blueprint": blueprint}

    return {"ok": True, "paths": file_paths, "blueprint": blueprint}


__all__ = ("analyze_workspace_image", "analyze_reference_design")
