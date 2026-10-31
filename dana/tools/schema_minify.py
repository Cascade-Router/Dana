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
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")

# Past this length a description is almost always restating the parameter
# name/type in prose rather than adding information the model needs —
# trimmed to the first sentence instead of dropped outright, so a genuinely
# load-bearing caveat ("value must be positive") still survives.
_MAX_DESCRIPTION_CHARS = 140


def minification_enabled() -> bool:
    raw = (os.environ.get("DANA_MINIFY_SCHEMAS") or "").strip().lower()
    return raw not in {"0", "false", "off"}


# A ". " that ends one of these isn't a sentence end. Cutting there used to
# leave descriptions like create_plan's "Breaks a complex, long-horizon
# goal (e.g." on the wire.
_ABBREVIATIONS = ("e.g", "i.e", "etc", "vs", "approx")


def _first_sentence_end(text: str) -> int:
    """Index of the period ending ``text``'s first sentence, or -1."""
    start = 0
    while (index := text.find(". ", start)) != -1:
        if not text[:index].lower().endswith(_ABBREVIATIONS):
            return index
        start = index + 2
    return -1


def _condense(text: str, *, max_chars: int = _MAX_DESCRIPTION_CHARS) -> str:
    text = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    first_period = _first_sentence_end(text)
    if 0 < first_period <= max_chars:
        return text[: first_period + 1]
    return text[:max_chars].rstrip() + "…"


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


# Structural keys a parameter (sub-)schema is allowed to keep under
# strip_tool_schema below — everything else (description, examples, default,
# title, ...) is prose/metadata a model doesn't need to CALL the tool
# correctly, only to understand it in human terms. "enum"/"items" are kept
# alongside "type" because they're still part of the parameter's TYPE shape
# (a constrained string, an array of X) rather than free text.
_STRUCTURAL_PARAM_KEYS = ("type", "properties", "items", "required", "enum")

# The Minification Compromise: a stripped schema's root function.description
# is condensed (not dropped) to this length — real-world local-agent runs
# showed a 7B model fixating on one already-loaded tool (e.g. stamping out
# box after box) once the schema carried NOTHING but bare names/types; the
# one-sentence physical-shape context ("a rectangular/cuboid solid" vs "an
# N-sided prism") is exactly what the model needs to tell two same-shaped-
# looking tools apart, and costs only a few tokens per tool. Per-PARAMETER
# descriptions stay fully dropped — this is scoped to the function's own
# description only, the one line a model reads to pick WHICH tool to call
# in the first place.
_STRIPPED_DESCRIPTION_MAX_CHARS = 100


def _strip_parameter_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key in _STRUCTURAL_PARAM_KEYS:
        if key not in node:
            continue
        if key == "properties" and isinstance(node[key], dict):
            out[key] = {name: _strip_parameter_node(prop) for name, prop in node[key].items()}
        elif key == "items":
            out[key] = _strip_parameter_node(node[key])
        else:
            out[key] = node[key]
    return out


def strip_tool_schema(schema: dict) -> dict:
    """Aggressively strip an OpenAI-wire tool schema down to near the bare
    minimum a model needs to CALL it correctly: ``function.name``, a
    condensed (not dropped) one-sentence ``function.description`` — see
    ``_STRIPPED_DESCRIPTION_MAX_CHARS`` for why this one field survives —
    plus the type-shape of its parameters (``type``/``required``/
    per-property ``type``/``enum``/``items``). Every PARAMETER's own
    ``description``, and any other free-text/metadata key, is still
    dropped outright, not condensed.

    This is a strictly more aggressive mode than ``minify_tool_schema``
    above (Pillar 2, which condenses EVERY description — function and
    parameter — to ``_MAX_DESCRIPTION_CHARS`` and never removes the wire
    contract's descriptive intent). It exists for local models under real
    VRAM/context pressure — see ``should_strip_tool_schemas`` — and is
    opt-in, layered on TOP of ``minify_tool_schemas`` in
    ``dana.core.react_dispatch._llm_tools_schema``, not a replacement for
    it: every other provider keeps getting condensed-but-present
    descriptions on both function and parameters.
    """
    fn = schema.get("function")
    if not isinstance(fn, dict):
        return schema
    out_fn: dict[str, Any] = {"name": fn.get("name")}
    description = fn.get("description")
    if isinstance(description, str) and description.strip():
        out_fn["description"] = _condense(description, max_chars=_STRIPPED_DESCRIPTION_MAX_CHARS)
    params = fn.get("parameters")
    if isinstance(params, dict):
        out_fn["parameters"] = _strip_parameter_node(params)
    return {"type": schema.get("type", "function"), "function": out_fn}


def strip_tool_schemas(schemas: list[dict]) -> list[dict]:
    """Batch ``strip_tool_schema``."""
    return [strip_tool_schema(s) for s in schemas]


# --------------------------------------------------------------------------
# Adaptive compaction: for a turn whose REQUIRED tools alone overrun the
# token budget (an executing task that declares several of the heaviest
# tools). Unlike strip_tool_schema it keeps the one kind of parameter prose
# a model can't infer from a name or type -- where a placement puts the
# solid (a box by its corner, a cylinder by its base) -- and it tightens in
# levels, so the caller can stop at the first one that fits.
# --------------------------------------------------------------------------

_CLAUSE_BREAKS = (" (", "; ", " — ", " -- ", ": ")
_CRITICAL_WORDS = ("corner", "base", "origin", "center", "centre")
COMPACTION_LEVELS = (1, 2, 3)


def _first_clause(text: str, max_chars: int) -> str:
    """``text`` cut at its first sentence or clause break, then at a word
    boundary within ``max_chars``."""
    text = _WHITESPACE_RE.sub(" ", text or "").strip()
    end = _first_sentence_end(text)
    if end > 0:
        text = text[:end]
    for brk in _CLAUSE_BREAKS:
        index = text.find(brk)
        if index > 0:
            text = text[:index]
    text = text.rstrip(" .,;")
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars].rstrip(" ,;") + "…"
    return text


def _is_critical(name: str, description: str) -> bool:
    """A parameter whose description says WHERE geometry goes, which its
    name and type alone don't tell a model."""
    lowered = description.lower()
    return name.startswith("placement") or any(word in lowered for word in _CRITICAL_WORDS)


def _compact_node(node: Any, name: str, level: int) -> Any:
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {prop: _compact_node(sub, prop, level) for prop, sub in value.items()}
        elif key == "items":
            out[key] = _compact_node(value, name, level)
        elif key == "description":
            if not isinstance(value, str):
                continue
            if level == 1:
                out[key] = _first_clause(value, 60)
            elif _is_critical(name, value):
                out[key] = _first_clause(value, 60 if level == 2 else 40)
        elif key in _STRUCTURAL_PARAM_KEYS:
            out[key] = value
    return out


def compact_tool_schema(schema: dict, level: int) -> dict:
    """A copy of ``schema`` with its prose tightened to ``level``:

    1. the function description and every parameter description cut to
       their first clause (80/60 characters);
    2. the function description to 60 characters, and only placement-style
       parameter descriptions (see ``_is_critical``) kept, at 60;
    3. as 2, at 40 characters each.

    Names, types, ``required``, ``enum`` and ``items`` survive every level,
    so the tool stays exactly as callable as before."""
    fn = schema.get("function")
    if not isinstance(fn, dict):
        return schema
    out_fn: dict[str, Any] = {"name": fn.get("name")}
    description = fn.get("description")
    if isinstance(description, str) and description.strip():
        out_fn["description"] = _first_clause(description, 80 if level == 1 else 60)
    params = fn.get("parameters")
    if isinstance(params, dict):
        out_fn["parameters"] = _compact_node(params, "", level)
    return {"type": schema.get("type", "function"), "function": out_fn}


def should_strip_tool_schemas(provider: str | None = None) -> bool:
    """Whether ``strip_tool_schema`` (full removal) should run on top of the
    always-on ``minify_tool_schemas`` condensing pass, for this turn's
    resolved tool-calling ``provider`` (e.g. ``dana.core.model_provider.
    tool_calling_provider()``). True if explicitly forced via
    ``DANA_MINIFY_TOOL_SCHEMAS=true``, or if the provider is local Ollama —
    the case this exists for: a 7B-class local model paying real context/VRAM
    cost for every tool's description text, unlike a cloud model with ample
    context headroom.
    """
    if (os.environ.get("DANA_MINIFY_TOOL_SCHEMAS") or "").strip().lower() == "true":
        return True
    return (provider or "").strip().lower() == "ollama"


def estimate_tokens(schemas: list[dict]) -> int:
    """Cheap word-count-based estimate — good enough for a before/after
    comparison, not a real tokenizer. Same whitespace-split heuristic
    ``dana.memory.compressor._approx_tokens`` already uses elsewhere in this
    codebase, so a log line built from this is comparable across modules.
    """
    words = len(re.findall(r"\S+", json.dumps(schemas)))
    return int(words * 1.3)


__all__ = (
    "COMPACTION_LEVELS",
    "compact_tool_schema",
    "minify_tool_schema",
    "minify_tool_schemas",
    "minification_enabled",
    "estimate_tokens",
    "strip_tool_schema",
    "strip_tool_schemas",
    "should_strip_tool_schemas",
)
