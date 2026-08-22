"""Local, dictionary-based lookup of standard mechanical hardware dimensions.

Backs the ``query_engineering_standard`` tool: Dana has the physical tools
to sketch/extrude a mounting plate, but nothing stopped it from hallucinating
a NEMA 17's bolt-hole spacing or an M3 clearance diameter. This is a plain
Python dict — no vector DB, no embeddings, no network call — matching the
"lean, local, dictionary-based" scope for this first pass; a real parts
database can replace ``_STANDARDS`` later without touching the lookup
contract below.
"""

from __future__ import annotations

import re
from typing import Any

# Each entry's "keywords" are plain substrings matched against the
# lowercased query — deliberately not a fuzzy/embedding search, so matches
# are exact and auditable. Add new hardware by adding an entry here; nothing
# else needs to change.
_STANDARDS: dict[str, dict[str, Any]] = {
    "nema17": {
        "title": "NEMA 17 Stepper Motor",
        "keywords": ("nema 17", "nema17", "nema-17", "stepper motor", "stepper"),
        "dimensions": {
            "body_width_mm": 42.3,
            "body_height_mm": 42.3,
            "typical_body_depth_mm": 40.0,
            "shaft_diameter_mm": 5.0,
            "pilot_boss_diameter_mm": 22.0,
            "pilot_boss_depth_mm": 2.0,
            "mounting_hole_count": 4,
            "mounting_hole_pattern": "square, centered on the pilot boss",
            "mounting_hole_spacing_mm": 31.0,
            "mounting_hole_diameter_mm": 3.2,
            "mounting_hole_bolt_size": "M3",
            "default_shaft_length_mm": 24.0,
        },
    },
    "m3_clearance": {
        "title": "M3 Clearance Hole",
        "keywords": ("m3", "m3 clearance", "m3 bolt", "m3 screw"),
        "dimensions": {
            "nominal_diameter_mm": 3.0,
            "close_fit_mm": 3.2,
            "free_fit_mm": 3.4,
            "loose_fit_mm": 3.6,
        },
    },
    "m4_clearance": {
        "title": "M4 Clearance Hole",
        "keywords": ("m4", "m4 clearance", "m4 bolt", "m4 screw"),
        "dimensions": {
            "nominal_diameter_mm": 4.0,
            "close_fit_mm": 4.3,
            "free_fit_mm": 4.5,
            "loose_fit_mm": 4.8,
        },
    },
    "m5_clearance": {
        "title": "M5 Clearance Hole",
        "keywords": ("m5", "m5 clearance", "m5 bolt", "m5 screw"),
        "dimensions": {
            "nominal_diameter_mm": 5.0,
            "close_fit_mm": 5.3,
            "free_fit_mm": 5.5,
            "loose_fit_mm": 5.8,
        },
    },
    "bearing_608": {
        "title": "608 Ball Bearing",
        "keywords": ("608", "608 bearing", "ball bearing"),
        "dimensions": {"bore_diameter_mm": 8.0, "outer_diameter_mm": 22.0, "width_mm": 7.0},
    },
}


def _available_titles() -> list[str]:
    return [entry["title"] for entry in _STANDARDS.values()]


def query_engineering_standard(query: str) -> dict[str, Any]:
    """Look up standard hardware dimensions by keyword match.

    Scores every entry by how many of its keyword substrings appear in the
    (lowercased) query and returns the highest-scoring one. A genuine tie
    returns ALL tied candidates under ``matches`` with ``ambiguous: True``
    rather than silently guessing — engineering dimensions are exactly the
    place a wrong silent guess is worse than asking again. No match at all
    returns ``{"ok": False, ...}`` (which ``dispatch_tool_call`` will run
    through ``digest_error``, same as any other tool failure) listing what
    IS available so the caller can retry with a better query.
    """
    q = (query or "").strip().lower()
    if not q:
        return {"ok": False, "error": "query_engineering_standard requires a non-empty query"}

    scored = [
        (sum(1 for kw in entry["keywords"] if kw in q), key, entry)
        for key, entry in _STANDARDS.items()
    ]
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return {
            "ok": False,
            "error": f"no engineering standard matched query {query!r}",
            "available_standards": _available_titles(),
        }

    top_score = max(s[0] for s in scored)
    top_matches = [s for s in scored if s[0] == top_score]
    if len(top_matches) > 1:
        return {
            "ok": True,
            "query": query,
            "ambiguous": True,
            "matches": [
                {"standard": key, "title": entry["title"], "dimensions": entry["dimensions"]}
                for _, key, entry in top_matches
            ],
        }

    _, key, entry = top_matches[0]
    return {
        "ok": True,
        "query": query,
        "standard": key,
        "title": entry["title"],
        "dimensions": entry["dimensions"],
    }


# --------------------------------------------------------------------------
# Programmatic (non-fuzzy) accessors — insert_standard_part's data source.
#
# query_engineering_standard above is for the LLM's natural-language
# questions; a tool-to-tool call building real geometry (standard_parts.py)
# needs exact, typed lookups instead of free-text keyword scoring, so these
# functions parse/validate their input and raise ValueError on a bad spec
# rather than ever silently guessing a dimension.
# --------------------------------------------------------------------------

# Socket head cap screw proportions (ISO 4762-ish, common hobbyist/hardware
# sizes) — keyed by nominal thread diameter in mm.
_SCREW_GEOMETRY: dict[int, dict[str, float]] = {
    3: {"head_diameter_mm": 5.5, "head_height_mm": 3.0},
    4: {"head_diameter_mm": 7.0, "head_height_mm": 4.0},
    5: {"head_diameter_mm": 8.5, "head_height_mm": 5.0},
}

_SCREW_SPEC_RE = re.compile(r"^M(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)$")


def parse_screw_spec(spec: str) -> dict[str, float]:
    """Parse a screw spec like ``"M3x12"`` (M<diameter>x<length_mm>) into
    its full geometry: nominal shank diameter, length, and head
    diameter/height. Raises ``ValueError`` for a malformed spec or an
    unsupported diameter — callers should surface that as a tool failure,
    not fall back to guessing.
    """
    m = _SCREW_SPEC_RE.match((spec or "").strip())
    if not m:
        raise ValueError(f"invalid screw spec {spec!r} — expected 'M<diameter>x<length>', e.g. 'M3x12'")
    diameter = int(m.group(1))
    length = float(m.group(2))
    geo = _SCREW_GEOMETRY.get(diameter)
    if geo is None:
        raise ValueError(f"no screw geometry for M{diameter} — supported sizes: {sorted(_SCREW_GEOMETRY)}")
    return {"nominal_diameter_mm": float(diameter), "length_mm": length, **geo}


# Deep-groove ball bearing geometry, keyed by bearing designation.
_BEARING_GEOMETRY: dict[str, dict[str, float]] = {
    "608": {"bore_diameter_mm": 8.0, "outer_diameter_mm": 22.0, "width_mm": 7.0},
    "6000": {"bore_diameter_mm": 10.0, "outer_diameter_mm": 26.0, "width_mm": 8.0},
    "6200": {"bore_diameter_mm": 10.0, "outer_diameter_mm": 30.0, "width_mm": 9.0},
}


def get_bearing_geometry(designation: str) -> dict[str, float]:
    """Exact bore/outer-diameter/width for a ball bearing designation (e.g.
    ``"608"``). Raises ``ValueError`` for an unsupported designation."""
    key = (designation or "").strip().lower().replace("-", "").replace(" ", "")
    geo = _BEARING_GEOMETRY.get(key)
    if geo is None:
        raise ValueError(f"no bearing geometry for {designation!r} — supported designations: {sorted(_BEARING_GEOMETRY)}")
    return dict(geo)


def get_nema17_dimensions() -> dict[str, float]:
    """Exact NEMA 17 dimensions as a plain dict — the same numbers
    ``query_engineering_standard("NEMA 17")`` returns, without the
    free-text lookup wrapper."""
    return dict(_STANDARDS["nema17"]["dimensions"])


__all__ = (
    "get_bearing_geometry",
    "get_nema17_dimensions",
    "parse_screw_spec",
    "query_engineering_standard",
)
