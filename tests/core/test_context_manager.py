"""Unit tests for dana.core.context_manager.prune_message_history — the
sliding-context-window pruner that strips Base64 image attachments out of
older ReAct-loop turns before a ``messages`` history is handed to
``ModelProvider``. Pure-function tests: no LLM, no websocket, no fixtures
beyond plain dicts.
"""

from __future__ import annotations

import copy

from dana.core.context_manager import OMITTED_IMAGE_PLACEHOLDER, prune_message_history


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_part(data_uri: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _user_with_image(text: str, data_uri: str) -> dict:
    return {"role": "user", "content": [_text_part(text), _image_part(data_uri)]}


def test_plain_string_content_messages_pass_through_unchanged() -> None:
    messages = [
        {"role": "system", "content": "You are Dana."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    pruned = prune_message_history(messages, keep_recent_images=2)
    assert pruned == messages


def test_no_images_anywhere_is_a_no_op() -> None:
    messages = [
        {"role": "system", "content": "You are Dana."},
        {"role": "user", "content": [_text_part("just text, no image")]},
    ]
    pruned = prune_message_history(messages, keep_recent_images=2)
    assert pruned == messages


def test_images_within_the_keep_budget_are_left_intact() -> None:
    messages = [
        _user_with_image("turn 1", "data:image/png;base64,AAA"),
        _user_with_image("turn 2", "data:image/png;base64,BBB"),
    ]
    pruned = prune_message_history(messages, keep_recent_images=2)
    assert pruned == messages


def test_older_image_beyond_budget_is_replaced_text_preserved() -> None:
    """The core requirement: an image older than the threshold gets its
    base64 data URI replaced with the placeholder, while the text part in
    the SAME message (and every other message) is completely untouched."""
    messages = [
        _user_with_image("what is this part?", "data:image/png;base64,OLDEST"),
        _user_with_image("and this one?", "data:image/png;base64,NEWEST"),
    ]
    pruned = prune_message_history(messages, keep_recent_images=1)

    oldest_content = pruned[0]["content"]
    assert oldest_content[0] == _text_part("what is this part?")  # text untouched
    assert oldest_content[1] == _text_part(OMITTED_IMAGE_PLACEHOLDER)  # image -> placeholder
    assert "OLDEST" not in str(oldest_content)

    newest_content = pruned[1]["content"]
    assert newest_content[1] == _image_part("data:image/png;base64,NEWEST")  # kept verbatim


def test_keep_recent_images_zero_prunes_every_image() -> None:
    messages = [
        _user_with_image("a", "data:image/png;base64,AAA"),
        _user_with_image("b", "data:image/png;base64,BBB"),
    ]
    pruned = prune_message_history(messages, keep_recent_images=0)
    for m in pruned:
        image_parts = [p for p in m["content"] if p["type"] == "image_url"]
        assert image_parts == []


def test_multiple_images_in_one_message_count_individually_toward_budget() -> None:
    multi_image_message = {
        "role": "user",
        "content": [
            _text_part("compare these three views"),
            _image_part("data:image/png;base64,FRONT"),
            _image_part("data:image/png;base64,TOP"),
            _image_part("data:image/png;base64,SIDE"),
        ],
    }
    pruned = prune_message_history([multi_image_message], keep_recent_images=1)
    content = pruned[0]["content"]
    # Only the LAST image (most recent) among the three survives.
    assert content[1] == _text_part(OMITTED_IMAGE_PLACEHOLDER)
    assert content[2] == _text_part(OMITTED_IMAGE_PLACEHOLDER)
    assert content[3] == _image_part("data:image/png;base64,SIDE")


def test_does_not_mutate_the_input_messages_list_or_its_dicts() -> None:
    """Strict requirement: the pruner must only ever affect the payload sent
    to the LLM, never the session's real conversation history — verified
    here by deep-copying before pruning and asserting the original is
    byte-for-byte identical afterward."""
    messages = [
        {"role": "system", "content": "You are Dana."},
        _user_with_image("turn 1", "data:image/png;base64,OLDEST"),
        _user_with_image("turn 2", "data:image/png;base64,NEWEST"),
    ]
    before = copy.deepcopy(messages)

    result = prune_message_history(messages, keep_recent_images=1)

    assert messages == before  # untouched
    assert result != messages  # the returned list actually differs (pruning happened)


def test_tool_and_assistant_messages_are_passed_through_untouched() -> None:
    """Tool-result / assistant messages never carry image content parts in
    this codebase's message shapes (see build_tool_result_message,
    build_assistant_tool_call_message) — the pruner must leave their
    non-list ``content`` (and ``tool_calls``/``tool_call_id`` fields)
    completely alone."""
    messages = [
        _user_with_image("a", "data:image/png;base64,AAA"),
        _user_with_image("b", "data:image/png;base64,BBB"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]
    pruned = prune_message_history(messages, keep_recent_images=1)
    assert pruned[2] == messages[2]
    assert pruned[3] == messages[3]
