"""Sliding context-window pruner for the ReAct loop's ``messages`` history.

Keeping every turn's Base64 image attachment (see ``dana.core.react_dispatch.
build_user_message``) verbatim in ``messages`` means a 10-turn conversation
re-sends every image it ever saw, in full, on every single subsequent LLM
call — burning tokens, adding latency, and eventually overflowing the
model's context window. ``prune_message_history`` strips the PIXEL DATA out
of older image attachments while leaving the surrounding conversation
(including the text the user attached them with) untouched.

This module only ever touches the payload handed to ``ModelProvider`` for
one LLM call — never the actual ``messages`` list a session's ReAct loop is
built on (dana.api.server's ``react_state``/``visual_state``, or the
websocket history the frontend renders). Every function here is pure: it
returns a new list/dicts rather than mutating its input, so a caller that
prunes right before an LLM call can keep passing the SAME original
``messages`` list into the next loop iteration with its images intact.
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


__all__ = ("OMITTED_IMAGE_PLACEHOLDER", "prune_message_history")
