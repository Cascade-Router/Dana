"""Token-compression for OpenAI-wire tool schemas — Pillar 2.

The wire CONTRACT (the OpenAI function-calling shape itself —
``type``/``function``/``name``/``parameters``/``properties``/``required``)
can't be renamed, flattened, or re-encoded as YAML: Groq and every other
OpenAI-compatible ``/chat/completions`` endpoint parses exactly that JSON
shape, so any real restructuring would break tool-calling outright. What
actually costs tokens for no benefit is the free-text CONTENT inside that
shape — verbose, repetitive ``description`` prose on both the function and
its parameters (``dana/tools/tools.json``'s ``description_en`` fields were
written for human readability, not token economy). This module condenses
only that prose; every structural key is passed through byte-identical.
"""

from __future__ import annotations

import copy
import json
import os
import re

_WHITESPACE_RE = re.compile(r"\s+")

# Past this length a description is almost always restating the parameter
# name/type in prose rather than adding information the model needs —
# trimmed to the first sentence instead of dropped outright, so a genuinely
# load-bearing caveat ("value must be positive") still survives.
_MAX_DESCRIPTION_CHARS = 140


def minification_enabled() -> bool:
    raw = (os.environ.get("DANA_MINIFY_SCHEMAS") or "").strip().lower()
    return raw not in {"0", "false", "off"}


def _condense(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if len(text) <= _MAX_DESCRIPTION_CHARS:
        return text
    first_period = text.find(". ")
    if 0 < first_period <= _MAX_DESCRIPTION_CHARS:
        return text[: first_period + 1]
    return text[:_MAX_DESCRIPTION_CHARS].rstrip() + "…"


def minify_tool_schema(schema: dict) -> dict:
    """Return a NEW schema dict with every free-text description condensed.

    Never mutates ``schema`` in place, and never touches a structural key
    (``type``, ``name``, ``properties``, ``required``, ``enum``, ``items``)
    — only ``description`` strings, on the function itself and on each
    parameter, get rewritten.
    """
    fn = schema.get("function")
    if not isinstance(fn, dict):
        return schema
    out = copy.deepcopy(schema)
    out_fn = out["function"]
    if isinstance(out_fn.get("description"), str):
        out_fn["description"] = _condense(out_fn["description"])
    params = out_fn.get("parameters")
    props = params.get("properties") if isinstance(params, dict) else None
    if isinstance(props, dict):
        for prop in props.values():
            if isinstance(prop, dict) and isinstance(prop.get("description"), str):
                prop["description"] = _condense(prop["description"])
    return out


def minify_tool_schemas(schemas: list[dict]) -> list[dict]:
    """Batch ``minify_tool_schema``, a no-op passthrough when disabled via
    ``DANA_MINIFY_SCHEMAS=0`` — the escape hatch if a minified description
    ever turns out to regress a specific model's tool-calling accuracy.
    """
    if not minification_enabled():
        return list(schemas)
    return [minify_tool_schema(s) for s in schemas]


def estimate_tokens(schemas: list[dict]) -> int:
    """Cheap word-count-based estimate — good enough for a before/after
    comparison, not a real tokenizer. Same whitespace-split heuristic
    ``dana.memory.compressor._approx_tokens`` already uses elsewhere in this
    codebase, so a log line built from this is comparable across modules.
    """
    words = len(re.findall(r"\S+", json.dumps(schemas)))
    return int(words * 1.3)


__all__ = ("minify_tool_schema", "minify_tool_schemas", "minification_enabled", "estimate_tokens")
