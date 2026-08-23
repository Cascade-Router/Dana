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
# ever eligible for truncation. Lowered from 2 to 1 (only the SINGLE most
# recent tool result is now exempt) so a multi-step chain accumulating
# several verbose tool results in a row (e.g. FreeCAD's create/boolean/export
# sequence, each with its own bounding-box/placement/path payload) starts
# trimming stale history from turn 3 instead of turn 4 — part of the same
# fix that keeps a long ReAct chain under Groq's TPM ceiling as the tool
# token budget above.
_DEFAULT_KEEP_RECENT_TOOL_RESULTS = 1
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


__all__ = ("OMITTED_IMAGE_PLACEHOLDER", "prune_message_history", "prune_tool_output_history")
