"""Deterministic Scratchpad filter — compress raw tool outputs before LLM context."""

from __future__ import annotations

DEFAULT_MAX_LENGTH = 1200
_HEAD_CHARS = 500
_TAIL_CHARS = 500


def compress_tool_output(
    output_string: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Compress oversized tool observations for the ReAct / MoA context window.

    If ``len(output_string) <= max_length``, return unchanged. Otherwise keep the
    first 500 and last 500 characters and replace the middle with a clear
    Scratchpad delimiter noting how many characters were removed.
    """
    text = "" if output_string is None else str(output_string)
    try:
        limit = max(1, int(max_length))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_LENGTH
    if len(text) <= limit:
        return text

    head_n = min(_HEAD_CHARS, limit)
    tail_n = min(_TAIL_CHARS, max(0, limit - head_n))
    # Prefer the canonical 500/500 split when the budget allows.
    if limit >= _HEAD_CHARS + _TAIL_CHARS:
        head_n = _HEAD_CHARS
        tail_n = _TAIL_CHARS

    head = text[:head_n]
    tail = text[-tail_n:] if tail_n else ""
    removed = max(0, len(text) - head_n - tail_n)
    delimiter = (
        f"\n\n... [SCRATCHPAD COMPRESSION: {removed} characters removed] ...\n\n"
    )
    return f"{head}{delimiter}{tail}"
