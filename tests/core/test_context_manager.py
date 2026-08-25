"""Unit tests for dana.core.context_manager's sliding-context-window
pruners: ``prune_message_history`` strips Base64 image attachments out of
older ReAct-loop turns; ``prune_tool_output_history`` truncates stale
``tool``-role result content the same way. Both run right before a
``messages`` history is handed to ``ModelProvider``. Pure-function tests: no
LLM, no websocket, no fixtures beyond plain dicts.
"""

from __future__ import annotations

import copy

from dana.core.context_manager import (
    OMITTED_IMAGE_PLACEHOLDER,
    prune_message_history,
    prune_tool_output_history,
)


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


# --------------------------------------------------------------------------
# prune_tool_output_history — stale tool-result truncation (Groq TPM relief)
# --------------------------------------------------------------------------


def _tool_cycle(call_id: str, tool_content: str) -> list[dict]:
    """One assistant tool_calls message + its matching tool-result message —
    the exact shape build_assistant_tool_call_message/build_tool_result_message
    produce for a single ReAct turn."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "search_codebase", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": tool_content},
    ]


def _long_output(marker: str, length: int = 1000) -> str:
    body = f"{marker}-" * (length // (len(marker) + 1) + 1)
    return body[:length]


def test_short_tool_output_is_never_truncated_regardless_of_age() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        *_tool_cycle("call_1", "short result"),
        *_tool_cycle("call_2", "short result 2"),
        *_tool_cycle("call_3", "short result 3"),
    ]
    pruned = prune_tool_output_history(messages, keep_recent=1, truncate_threshold=500)
    assert pruned == messages


def test_older_tool_outputs_beyond_keep_recent_are_truncated() -> None:
    """The core requirement: with 5 large tool outputs and keep_recent=2,
    the oldest 3 get their content truncated while the most recent 2 stay
    byte-for-byte intact."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    outputs = [_long_output(f"turn{i}") for i in range(5)]
    for i, output in enumerate(outputs):
        messages += _tool_cycle(f"call_{i}", output)

    pruned = prune_tool_output_history(messages, keep_recent=2, truncate_threshold=500)

    tool_messages = [m for m in pruned if m["role"] == "tool"]
    assert len(tool_messages) == 5  # message COUNT never changes

    # Oldest 3 (turn0, turn1, turn2) truncated.
    for i in range(3):
        content = tool_messages[i]["content"]
        assert content != outputs[i]
        assert content.startswith("[Pruned to save context]")
        assert content.count(f"turn{i}-") > 0  # head survives
        assert outputs[i][:200] in content
        assert outputs[i][-200:] in content
        assert len(content) < len(outputs[i])

    # Most recent 2 (turn3, turn4) left completely untouched.
    assert tool_messages[3]["content"] == outputs[3]
    assert tool_messages[4]["content"] == outputs[4]

    # Every non-tool message (system/user/assistant tool_calls announcements)
    # passes through unchanged too.
    assert pruned[0] == messages[0]
    assert pruned[1] == messages[1]
    assistant_messages = [m for m in pruned if m["role"] == "assistant"]
    original_assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert assistant_messages == original_assistant_messages


def test_prune_tool_output_history_never_changes_message_count_or_order() -> None:
    """CRITICAL API constraint: OpenAI/Groq require every tool_calls message
    to be immediately followed by its matching tool-result message — pruning
    must never remove/add/reorder a message, only rewrite a content string."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(5):
        messages += _tool_cycle(f"call_{i}", _long_output(f"turn{i}"))

    pruned = prune_tool_output_history(messages, keep_recent=1, truncate_threshold=500)

    assert len(pruned) == len(messages)
    assert [m["role"] for m in pruned] == [m["role"] for m in messages]
    # Every tool_calls message is still immediately followed by its own
    # matching tool-result message, by id, in the same relative position.
    for i, message in enumerate(pruned):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_id = message["tool_calls"][0]["id"]
            assert pruned[i + 1]["role"] == "tool"
            assert pruned[i + 1]["tool_call_id"] == call_id


def test_no_op_when_tool_message_count_is_within_keep_recent_budget() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for i in range(2):
        messages += _tool_cycle(f"call_{i}", _long_output(f"turn{i}"))
    pruned = prune_tool_output_history(messages, keep_recent=2, truncate_threshold=500)
    assert pruned == messages


def test_non_string_tool_content_is_left_alone() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        *_tool_cycle("call_1", _long_output("old")),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": {"not": "a string"}},
        *_tool_cycle("call_3", "recent, short"),
    ]
    pruned = prune_tool_output_history(messages, keep_recent=1, truncate_threshold=500)
    malformed = [m for m in pruned if m.get("tool_call_id") == "call_2"][0]
    assert malformed["content"] == {"not": "a string"}


def test_prune_tool_output_history_does_not_mutate_the_input() -> None:
    messages = [{"role": "system", "content": "sys"}]
    for i in range(3):
        messages += _tool_cycle(f"call_{i}", _long_output(f"turn{i}"))
    before = copy.deepcopy(messages)

    result = prune_tool_output_history(messages, keep_recent=1, truncate_threshold=500)

    assert messages == before  # untouched
    assert result != messages  # pruning actually happened
