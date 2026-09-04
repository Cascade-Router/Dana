"""Sliding context-window pruner for the ReAct loop's ``messages`` history.

Keeping every turn's Base64 image attachment (see ``dana.core.react_dispatch.
build_user_message``) verbatim in ``messages`` means a 10-turn conversation
re-sends every image it ever saw, in full, on every single subsequent LLM
call — burning tokens, adding latency, and eventually overflowing the
model's context window. ``prune_message_history`` strips the PIXEL DATA out
of older image attachments while leaving the surrounding conversation
(including the text the user attached them with) untouched.

``prune_tool_output_history`` addresses the same problem for a different
payload: a multi-step ReAct chain's own ``tool`` role messages (``dana.core.
react_dispatch.build_tool_result_message``'s ``{"role": "tool", "content":
json.dumps(payload)}``). A tool like ``search_codebase``/``execute_code_task``
can return several thousand characters of matches/diff/traceback — by Turn 4
of a chain, the model is re-reading every one of those in full on every
subsequent call, which is exactly what blows up a cloud provider's
Tokens-Per-Minute budget (Groq 429s) well before the model's actual context
window fills up.

``compress_tool_output_history`` is a local-model-only alternative to
``prune_tool_output_history`` (see its own docstring) — used instead of, not
alongside, the character-slice version, when
``dana.core.model_provider.tool_calling_provider() == "ollama"`` — a 7B-class
local model pays real VRAM/context cost per token that a cloud provider's
ample headroom does not. It compresses a stale ``tool`` result's JSON
structure directly (dropping large/low-signal fields, keeping every
file/object-name, numeric-parameter, and outcome field verbatim) rather than
slicing raw characters, and never touches an UNRESOLVED error's payload at
all, however old, since the system prompt's own "stop after the same error
repeats" instruction depends on the model still seeing it.

This module only ever touches the payload handed to ``ModelProvider`` for
one LLM call — never the actual ``messages`` list a session's ReAct loop is
built on (dana.api.server's ``react_state``/``visual_state``, or the
websocket history the frontend renders). Every function here is pure: it
returns a new list/dicts rather than mutating its input, so a caller that
prunes right before an LLM call can keep passing the SAME original
``messages`` list into the next loop iteration with its full tool outputs
intact.
"""

from __future__ import annotations

import json
from typing import Any

# What an older, pruned-away image attachment gets replaced with — kept as
# a text content part (not a broken/placeholder image_url) so the resulting
# message stays a valid OpenAI-wire content array with no dangling
# non-data-uri "url" a stricter upstream provider might reject.
OMITTED_IMAGE_PLACEHOLDER = "[Image omitted from history to save context]"


def _count_image_parts(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def prune_message_history(messages: list[dict[str, Any]], keep_recent_images: int = 2) -> list[dict[str, Any]]:
    """Returns a NEW ``messages`` list with older image attachments replaced
    by ``OMITTED_IMAGE_PLACEHOLDER`` — every content part that isn't an
    ``image_url`` (including the text part(s) in the same multimodal
    message) is passed through completely unchanged.

    ``keep_recent_images`` counts ``image_url`` parts across the WHOLE
    history, most-recent-first, not per-message or per-turn — a single
    message with 3 attachments only leaves budget for
    ``keep_recent_images - 3`` older images elsewhere in the conversation.
    Non-positive/zero prunes every image found.

    Never mutates ``messages`` or any message/content-part dict within it:
    unaffected messages are returned by the same reference, and only
    messages that actually contain a pruned image get a shallow-copied
    ``content`` list. Safe to call with the SAME ``messages`` object every
    turn — the caller's own conversation history is never altered, only the
    fresh list this function returns.
    """
    total_images = _count_image_parts(messages)
    to_prune = max(0, total_images - max(0, keep_recent_images))
    if to_prune == 0:
        return list(messages)

    pruned: list[dict[str, Any]] = []
    seen_images = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            pruned.append(message)
            continue

        new_content: list[dict[str, Any]] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                if seen_images < to_prune:
                    new_content.append({"type": "text", "text": OMITTED_IMAGE_PLACEHOLDER})
                    changed = True
                else:
                    new_content.append(part)
                seen_images += 1
            else:
                new_content.append(part)

        pruned.append({**message, "content": new_content} if changed else message)

    return pruned


# Recent tool executions the model is very likely still actively reasoning
# about (e.g. "did that last search actually find the right file?") stay
# fully intact — only messages OLDER than this many tool-result messages are
# ever eligible for truncation. Raised from 1 back to 3 to keep enough
# recent tool-chain context intact for multi-step CAD sequences (e.g.
# create/boolean/export, each needing to see the previous step's actual
# result). NOTE: this directly trades against the Groq TPM 429 issue that
# a prior change fixed by LOWERING this from 2 to 1 in the first place — if
# Groq is the primary/cascade upstream and 429s reappear, that's this knob,
# and the fix is to lower it again (or lean harder on
# _DEFAULT_TOOL_OUTPUT_TRUNCATE_THRESHOLD/head_chars/tail_chars below rather
# than keep_recent) rather than re-adding a second pruning mechanism.
_DEFAULT_KEEP_RECENT_TOOL_RESULTS = 3
# Below this, a tool result is already cheap enough that truncating it would
# just add noise (the "[Pruned...]" wrapper itself costs characters) for no
# real token savings.
_DEFAULT_TOOL_OUTPUT_TRUNCATE_THRESHOLD = 500
_DEFAULT_TOOL_OUTPUT_HEAD_CHARS = 200
_DEFAULT_TOOL_OUTPUT_TAIL_CHARS = 200
# The tail is kept alongside the head (not just "first N chars...") because
# a tool's own error/verdict is disproportionately likely to be at the END
# of its output (dana.plugins.coder_plugin.engine's own tail-biased error
# truncation documents the same observation for pytest/aider output) — a
# head-only truncation would silently discard exactly the line the model
# most needs to remember a stale tool call actually failed.
_PRUNED_TOOL_OUTPUT_PREFIX = "[Pruned to save context] "


def prune_tool_output_history(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = _DEFAULT_KEEP_RECENT_TOOL_RESULTS,
    truncate_threshold: int = _DEFAULT_TOOL_OUTPUT_TRUNCATE_THRESHOLD,
    head_chars: int = _DEFAULT_TOOL_OUTPUT_HEAD_CHARS,
    tail_chars: int = _DEFAULT_TOOL_OUTPUT_TAIL_CHARS,
) -> list[dict[str, Any]]:
    """Returns a NEW ``messages`` list where every ``role: "tool"`` message
    older than the most recent ``keep_recent`` tool results has its string
    ``content`` truncated to ``head_chars`` + an ellipsis + ``tail_chars``,
    prefixed with ``_PRUNED_TOOL_OUTPUT_PREFIX`` — but ONLY when that
    content is longer than ``truncate_threshold`` to begin with.

    "Recent" is counted by position among ``tool``-role messages in THIS
    history, most-recent-last (the natural chronological order ``messages``
    is already built in) — since this ReAct loop dispatches exactly one tool
    call per LLM turn (see ``dana.core.react_dispatch.next_react_turn``),
    the last ``keep_recent`` tool messages are exactly the last
    ``keep_recent`` tool-execution turns.

    Every OTHER message (the ``assistant`` message announcing the tool call,
    ``system``/``user`` turns, and any tool message that's either recent
    enough or already short) is passed through by the same reference,
    completely untouched — this NEVER removes, reorders, or adds a message,
    only ever rewrites a ``content`` STRING in place on a shallow-copied
    dict. A ``tool_calls``-bearing assistant message is always immediately
    followed by its own matching ``tool`` result message no matter what this
    function does to that result's content — the strict
    call-then-result pairing OpenAI/Groq's API requires is untouched by
    construction, since message COUNT and ORDER are never modified.

    A tool message whose ``content`` isn't a plain string (already
    malformed/never produced by ``build_tool_result_message``) is left
    alone rather than guessed at.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    stale_count = max(0, len(tool_indices) - max(0, keep_recent))
    if stale_count == 0:
        return list(messages)
    stale_indices = set(tool_indices[:stale_count])

    pruned: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        if i not in stale_indices:
            pruned.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= truncate_threshold:
            pruned.append(message)
            continue
        truncated = f"{_PRUNED_TOOL_OUTPUT_PREFIX}{content[:head_chars]}...{content[-tail_chars:]}"
        pruned.append({**message, "content": truncated})
    return pruned


# A local model pays real VRAM/context cost per token, unlike a cloud
# provider with ample headroom — this raises the bar from
# ``prune_tool_output_history``'s blind head/tail character slice to a
# JSON-STRUCTURE-AWARE compression: a stale ``tool`` result gets its large,
# low-signal fields (a 28-entry unlocked_tools list, a verbose description,
# base64 mesh data, ...) dropped entirely, while every field that names a
# file/object, gives a numeric geometric parameter, or reports an unresolved
# failure survives byte-for-byte. Only reachable via
# ``compress_tool_output_history`` below, gated in
# ``dana.core.react_dispatch._call_llm_once`` by
# ``dana.core.model_provider.tool_calling_provider() == "ollama"`` — every
# other provider keeps the existing, cheaper ``prune_tool_output_history``
# behavior unchanged.
_ALWAYS_PRESERVE_KEYS = frozenset(
    {
        # Identity / location — what the model needs to refer back to a
        # specific object or file in a LATER tool call.
        "name",
        "id",
        "tool_id",
        "path",
        "type",
        "domain",
        # Outcome — whether the step actually happened.
        "ok",
        "status",
        "error",
        "reason",
        "suggestion",
        "message",
        # Numeric geometric parameters — exactly the numbers a later step
        # (a boolean op, a mate, a bounding-box comparison) needs verbatim.
        "length",
        "width",
        "height",
        "radius",
        "diameter",
        "x",
        "y",
        "z",
        "placement",
        "position",
        "centroid",
        "normal",
        "bounding_box",
        "dimensions",
        "target_position",
        "target_normal",
        "mesh_url",
        "screenshot_path",
    }
)

# A scalar value under this length is cheap enough to keep even for a key
# NOT in _ALWAYS_PRESERVE_KEYS — only a large blob (a long description, a
# big list) actually costs meaningful tokens, so only those get dropped.
_CHEAP_SCALAR_CHARS = 60
# A dict/list nested under a non-preserved key deeper than this is
# almost always inspection metadata (tool catalogs, capability listings),
# not something a later step in THIS chain will reference by name — dropped
# rather than recursed into further.
_MAX_COMPRESS_DEPTH = 2


def _is_unresolved_error(payload: Any) -> bool:
    """True if a tool-result JSON payload reports a failure — see
    ``compress_tool_output_history``'s own docstring for why these are
    never compressed, only ever passed through untouched."""
    if not isinstance(payload, dict):
        return False
    if payload.get("ok") is False:
        return True
    status = payload.get("status")
    return isinstance(status, str) and status.strip().lower() == "error"


def _compress_json_value(value: Any, *, depth: int) -> Any:
    if isinstance(value, dict):
        if depth >= _MAX_COMPRESS_DEPTH:
            return "[omitted to save context]"
        out: dict[str, Any] = {}
        for key, sub in value.items():
            keep_key = str(key).lower() in _ALWAYS_PRESERVE_KEYS
            if keep_key:
                out[key] = _compress_json_value(sub, depth=depth + 1)
            elif isinstance(sub, (str, int, float, bool)) and len(str(sub)) <= _CHEAP_SCALAR_CHARS:
                out[key] = sub
            # else: a large/nested value under a non-preserved key — dropped.
        return out
    if isinstance(value, list):
        if depth >= _MAX_COMPRESS_DEPTH:
            return "[omitted to save context]"
        return [_compress_json_value(item, depth=depth + 1) for item in value]
    return value


def compress_tool_result_payload(content: str, *, max_chars: int, head_chars: int, tail_chars: int) -> str:
    """Structured compression for one ``tool``-role message's ``content``.

    Tries JSON-structure-aware filtering first (``_compress_json_value`` —
    keeps every ``_ALWAYS_PRESERVE_KEYS`` field verbatim, at any nesting
    depth, dropping only large/irrelevant fields), which also strictly
    preserves an unresolved error's ``reason``/``suggestion`` even though
    ``compress_tool_output_history`` never calls this for a FULLY unresolved
    error payload in the first place (see ``_is_unresolved_error``) — this
    still matters for a payload that is itself nested one level under a
    non-error key. Falls back to the same head+ellipsis+tail character slice
    ``prune_tool_output_history`` already uses whenever ``content`` isn't
    valid JSON, or the structured pass didn't shrink it enough.
    """
    if len(content) <= max_chars:
        return content
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        compressed = _compress_json_value(parsed, depth=0)
        rendered = json.dumps(compressed, separators=(",", ":"), default=str)
        if len(rendered) <= max_chars:
            return rendered
        content = rendered  # still too long — fall through to head/tail on the compressed form
    return f"{_PRUNED_TOOL_OUTPUT_PREFIX}{content[:head_chars]}...{content[-tail_chars:]}"


def compress_tool_output_history(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = 4,
    truncate_threshold: int = _DEFAULT_TOOL_OUTPUT_TRUNCATE_THRESHOLD,
    head_chars: int = _DEFAULT_TOOL_OUTPUT_HEAD_CHARS,
    tail_chars: int = _DEFAULT_TOOL_OUTPUT_TAIL_CHARS,
) -> list[dict[str, Any]]:
    """Local-model variant of ``prune_tool_output_history``: same "only the
    ``tool``-role messages older than the most recent ``keep_recent`` ever
    change, message count/order is NEVER touched" contract, but a stale
    result gets JSON-structure-aware compression (``compress_tool_result_payload``)
    instead of a blind character slice — and an UNRESOLVED error (``"ok":
    false`` / ``"status": "error"``) is never compressed at all, no matter
    how old, since the model's own "stop after the same error repeats"
    instruction (see ``dana.core.react_dispatch._FREECAD_SYSTEM_PROMPT``'s
    "Engineering Rules" #4 / ``_CORE_SYSTEM_PROMPT``'s "Turn-Taking") depends
    on actually still seeing it.

    ``system``/``user``/``assistant`` messages are never inspected or
    altered — only a stale ``tool`` message's own ``content`` string is ever
    rewritten, so every system instruction the prompt carries is preserved
    by construction, not by this function's own logic.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    stale_count = max(0, len(tool_indices) - max(0, keep_recent))
    if stale_count == 0:
        return list(messages)
    stale_indices = set(tool_indices[:stale_count])

    pruned: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        if i not in stale_indices:
            pruned.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= truncate_threshold:
            pruned.append(message)
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            payload = None
        if _is_unresolved_error(payload):
            pruned.append(message)  # unresolved failure — always kept verbatim
            continue
        compressed = compress_tool_result_payload(
            content, max_chars=truncate_threshold, head_chars=head_chars, tail_chars=tail_chars
        )
        pruned.append({**message, "content": compressed} if compressed != content else message)
    return pruned


__all__ = (
    "OMITTED_IMAGE_PLACEHOLDER",
    "prune_message_history",
    "prune_tool_output_history",
    "compress_tool_output_history",
    "compress_tool_result_payload",
)
