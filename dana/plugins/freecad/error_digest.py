"""Error digestion — turns a raw FreeCAD/kernel failure string into a
structured, LLM-actionable payload instead of a raw stderr/traceback dump.

``dana.plugins.freecad.engine`` already converts every subprocess failure
(a bad exit code, a missing success marker, a timeout) into a plain
``{"ok": False, "error": <str>}`` contract — but that ``error`` string is
often a raw Python/OpenCASCADE traceback (e.g. ``"Part::TopoShape::makeFillet
failed"``) that tells the LLM nothing about WHY the operation failed or what
to try next. This module is a pure, stateless classifier sitting at the
ReAct dispatch choke point (``dana.core.react_dispatch.dispatch_tool_call``)
— not inside the engine itself, so the engine stays a plain file-in/file-out
subprocess wrapper — that pattern-matches known failure signatures and
attaches a human/LLM-readable ``reason`` + ``suggestion`` so the model can
autonomously self-correct on its next turn instead of stalling on an opaque
error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RAW_ERROR_MAX_CHARS = 400


@dataclass(frozen=True)
class _ErrorSignature:
    pattern: re.Pattern[str]
    reason: str
    suggestion: str


# A bare topic keyword ("fillet"/"chamfer") isn't enough on its own to
# classify a message as a kernel failure — dispatch's OWN input-validation
# errors legitimately mention those words too (e.g. "operation must be one
# of fillet, chamfer"). Each pattern below requires the topic word to
# co-occur with an actual failure indicator, via lookaheads, so a clean
# validation message never gets misclassified and overwritten.
_FAILURE_WORD = r"(fail|error|exception|unable|invalid|cannot|can't)"

# Ordered most-specific-first — the first matching signature wins, so a
# fillet-specific message is classified before the generic TopoShape/BRep
# catch-all below it.
_SIGNATURES: tuple[_ErrorSignature, ...] = (
    _ErrorSignature(
        re.compile(rf"(?=.*\bfillet\b)(?=.*{_FAILURE_WORD})", re.I),
        "Topology failure: the fillet radius likely exceeds what the target edge(s) can support.",
        "Retry perform_freecad_edge_operation with a smaller `value` (e.g. half the radius), "
        "or target fewer edges via a face-specific selection instead of the whole object.",
    ),
    _ErrorSignature(
        re.compile(rf"(?=.*\bchamfer\b)(?=.*{_FAILURE_WORD})", re.I),
        "Topology failure: the chamfer distance likely exceeds what the target edge(s) can support.",
        "Retry perform_freecad_edge_operation with a smaller `value`, or target a single face "
        "instead of the whole object.",
    ),
    _ErrorSignature(
        re.compile(r"(self.?intersect|non.?manifold|invalid shape)", re.I),
        "The resulting solid is self-intersecting or non-manifold.",
        "Check for overlapping/degenerate geometry (e.g. two primitives occupying the same space) "
        "before retrying, or inspect_spatial_properties on the inputs to confirm they're valid solids.",
    ),
    _ErrorSignature(
        re.compile(r"(BRep|TopoDS|TopoShape|OCC)", re.I),
        "A FreeCAD/OpenCASCADE kernel operation failed on the given geometry.",
        "Re-check the operand dimensions/placement for this operation — it may be geometrically "
        "infeasible as specified. Try smaller/simpler parameters.",
    ),
    _ErrorSignature(
        # Requires "freecad" to actually co-occur (see dana.plugins.freecad.
        # engine's own "FreeCADCmd timed out after {timeout}s") — a bare
        # "timed out" alone used to match ANY tool's timeout message
        # (dispatch_tool_call/digest_error runs for every tool, not just
        # FreeCAD's), silently overwriting e.g. coder_plugin's
        # run_verification_command timeout with this FreeCAD-specific text.
        re.compile(r"(?=.*freecad)(?=.*timed out)", re.I),
        "The FreeCAD subprocess did not finish within its time budget.",
        "This is not a geometry error — simplify the operation (fewer objects/edges) or simply retry.",
    ),
    _ErrorSignature(
        re.compile(r"not found", re.I),
        "A referenced object or file path could not be located.",
        "Confirm the object name was actually created earlier in this session before referencing it "
        "(object names are case-sensitive).",
    ),
)

_DEFAULT_SUGGESTION = "Adjust the call's parameters based on the reason above before retrying."


def digest_error(tool_id: str, raw_error: str | None) -> dict[str, str]:
    """Classify ``raw_error`` into a structured, LLM-actionable shape.

    Only ADDS structure on top of a recognized FreeCAD/OpenCASCADE kernel
    failure signature (a fillet/chamfer topology error, a non-manifold
    result, a timeout, ...) — a raw stderr/traceback dump the LLM can't act
    on becomes a clear ``reason`` + concrete ``suggestion``. Every OTHER
    failure (a dispatch-level validation message like "requires
    target_object", an "unknown object 'X'" lookup miss, a driver-specific
    limitation note) is ALREADY a clear, specific, human-authored string —
    those are preserved verbatim as ``reason`` rather than replaced with a
    generic placeholder, so no information is ever lost, only enriched.

    Never raises — worst case returns the original text unclassified, so a
    call site can always trust this returns a usable dict.
    """
    text = raw_error or "unknown error"
    truncated = text[:_RAW_ERROR_MAX_CHARS]
    for sig in _SIGNATURES:
        if sig.pattern.search(text):
            return {
                "status": "error",
                "tool_id": tool_id,
                "reason": sig.reason,
                "suggestion": sig.suggestion,
                "raw_error": truncated,
            }
    return {
        "status": "error",
        "tool_id": tool_id,
        "reason": truncated,
        "suggestion": _DEFAULT_SUGGESTION,
        "raw_error": truncated,
    }


__all__ = ("digest_error",)
