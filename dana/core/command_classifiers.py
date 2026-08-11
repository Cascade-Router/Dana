"""Deterministic STT command classifiers + conversation-memory helpers.

Extracted verbatim from ``dana.core_agent`` (Phase 7 of the core_agent.py
decomposition; see docs/architecture/phase7_core_agent_decomposition.md).
These are fast-path checks ``dana.core.agent_loop``'s ``conversation_worker``
runs on a Whisper transcript before handing it to the LLM (standby/sleep,
clear-context, lockdown, wall-clock-time), plus the short TTS filler spoken
the instant a tool call is about to run.
"""

from __future__ import annotations

import re

from dana.core.shared_state import (
    _tool_working_ack_sent,
    conversation_history,
    conversation_history_lock,
    set_subtitle,
)
from dana.audio.tts_manager import enqueue_speech_impl as enqueue_speech
from dana.logging import log, log_conversation, log_debug
from dana.tools import ToolCall

_STANDBY_PHRASES = frozenset(
    {
        "stand by",
        "standby",
        "go to sleep",
        "stop listening",
        "shut up",
        "bye",
        "quit",
        "exit",
        "stop",
        "goodbye",
        "good bye",
        "",
        "",
    }
)
_STANDBY_TAIL_WORDS = frozenset(
    {"bye", "quit", "exit", "stop", "goodbye", "standby"}
)

_CLEAR_CONTEXT_PHRASES = frozenset(
    {
        "clear context",
        "clear the context",
        "kill context",
        "kill your context",
        "kill the context",
        "forget that",
        "forget this",
        "forget everything",
        "start over",
        "reset memory",
        "reset context",
        "wipe context",
        "wipe memory",
        "new conversation",
        "fresh start",
        "   ",
        " ",
        " ",
    }
)


_LOCKDOWN_PHRASES = frozenset(
    {
        "lockdown",
        "lock down",
        "lock yourself",
        "secure the system",
    }
)

_TIME_PHRASES = (
    "what time is it right now",
    "what's the time right now",
    "what is the time right now",
    "what time of the day is it",
    "what time of day is it",
    "can you tell me what time of the day is it",
    "can you tell me what time it is",
    "tell me the time",
    "what's the time",
    "what is the time",
    "what time is it",
    "current time",
)
def is_standby_command(text: str) -> bool:
    """True if STT is an explicit standby / sleep system command (bypass LLM)."""
    raw = (text or "").strip()
    if not raw:
        return False
    # Collapsed ASCII form for EN phrases (handles "And bye.").
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _STANDBY_PHRASES:
        return True
    for phrase in sorted(_STANDBY_PHRASES, key=len, reverse=True):
        if " " in phrase and (
            ascii_norm == phrase or ascii_norm.endswith(" " + phrase)
        ):
            return True
    words = [w for w in re.split(r"\s+", ascii_norm) if w]
    if words and words[-1].strip(".,!?;:\"'`") in _STANDBY_TAIL_WORDS:
        return True
    # Exact phrase match after light whitespace normalize.
    fa_norm = re.sub(r"\s+", " ", raw).strip(" .,!?;:\"'`")
    return fa_norm in _STANDBY_PHRASES
def is_clear_context_command(text: str) -> bool:
    """True if STT asks to wipe the short-term conversation memory window."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _CLEAR_CONTEXT_PHRASES:
        return True
    for phrase in sorted(_CLEAR_CONTEXT_PHRASES, key=len, reverse=True):
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
        # Allow wake-prefixed forms: "Dana, clear context"
        if ascii_norm.startswith(phrase + " ") or f" {phrase}" in f" {ascii_norm}":
            # Avoid matching unrelated sentences that merely contain a substring
            # of a multi-word phrase mid-word; require phrase as contiguous tokens.
            if phrase in ascii_norm:
                return True
    fa_norm = re.sub(r"\s+", " ", raw).strip(" .,!?;:\"'`")
    return fa_norm in _CLEAR_CONTEXT_PHRASES
def flush_conversation_memory(*, reason: str = "manual") -> int:
    """Wipe the sliding short-term history (Memory window N/6). Returns prior turn count.

    Also runs the custom-tools context-wipe failsafe (delete Desktop custom_tools
    ``.py`` files, unregister, clear ``sys.modules``).
    """
    global conversation_history
    with conversation_history_lock:
        prior = [
            m
            for m in conversation_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        n = len(prior)
        conversation_history.clear()
    log("Conversation", f"Memory window flushed ({reason}); cleared {n} msgs")
    log_conversation("System", f"Context cleared ({reason}); wiped {n} msgs")
    try:
        from dana.tools.registry import wipe_custom_tools

        wiped = wipe_custom_tools(reason=f"context_wipe:{reason}")
        if wiped:
            log("Conversation", f"Custom tools wipe companion: {wiped!r}")
    except Exception as exc:  # noqa: BLE001
        log("Conversation", f"WARNING: custom tools wipe failed ({exc})")
    return n
def clear_context_spoken_reply(text: str = "") -> str:
    """Ack phrase after flushing short-term memory."""
    from dana.settings import resolve_reply_lang

    if resolve_reply_lang(text or "") == "fa":
        return " —    ‌."
    return "Okay — fresh start. Context cleared."
def is_lockdown_command(text: str) -> bool:
    """True if STT is an explicit vault lockdown / kill-switch command."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    if ascii_norm in _LOCKDOWN_PHRASES:
        return True
    for phrase in sorted(_LOCKDOWN_PHRASES, key=len, reverse=True):
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
    return False
def is_time_command(text: str) -> bool:
    """True if STT is a wall-clock question (deterministic fast-path; bypass LLM)."""
    raw = (text or "").strip()
    if not raw:
        return False
    ascii_norm = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:\"'`")
    ascii_norm = ascii_norm.replace("whats", "what's")
    for phrase in _TIME_PHRASES:
        if ascii_norm == phrase or ascii_norm.endswith(" " + phrase):
            return True
        if phrase in ascii_norm and len(ascii_norm) <= len(phrase) + 12:
            return True
    return False
def wall_clock_spoken_reply() -> str:
    """Format local wall clock for TTS (no LLM)."""
    from datetime import datetime

    current_time = datetime.now().strftime("%I:%M %p").lstrip("0")
    return f"It is {current_time}."
def speak_tool_working_ack(call: ToolCall, reply_lang: str) -> None:
    """Short TTS filler as soon as we know a tool will run (before slow LLM/search)."""
    if _tool_working_ack_sent.is_set():
        return
    _tool_working_ack_sent.set()
    tool_id = getattr(call, "tool_id", "") or ""
    if reply_lang == "fa":
        phrase = {
            "web_search": "  .",
            "describe_spatial_scene": "  .",
            "read_vault_memory": " ‌   .",
            "read_clipboard_context": " ‌  .",
            "run_terminal_command": "    .",
            "shell_execute": "  .",
            "execute_powershell": "  .",
            "write_to_file": "  .",
            "execute_command": "  .",
            "execute_python_script": "  .",
            "get_sandbox_job_status": "  .",
            "fetch_webpage": "  .",
            "file_editor": "  .",
            "python_repl": "  .",
            "flush_memory": "  ‌   ‌.",
            "publish_tool_to_general": "      ‌.",
            "open_application": "    ‌.",
            "read_local_file": "   .",
            "read_system_architecture": " .",
            "dispatch_research_swarm": "    ‌.",
            "dispatch_watchdog": "   ‌.",
            "kill_watchdog": "    ‌.",
            "save_script_to_library": "      ‌.",
        }.get(tool_id, " .")
    else:
        phrase = {
            "web_search": "Let me check.",
            "describe_spatial_scene": "Let me look.",
            "read_vault_memory": "Let me check my memory.",
            "read_clipboard_context": "Let me check the clipboard.",
            "run_terminal_command": "Let me run that in the terminal.",
            "shell_execute": "Running that in the local shell.",
            "execute_powershell": "Running that in PowerShell.",
            "write_to_file": "Okay — writing that file.",
            "execute_command": "Okay — running that command.",
            "execute_python_script": "Okay — running that Python script in the sandbox.",
            "get_sandbox_job_status": "Let me check that sandbox job.",
            "fetch_webpage": "Let me open that page.",
            "file_editor": "Working on that file.",
            "python_repl": "Running that in the Python sandbox.",
            "flush_memory": "Okay — wiping short-term memory.",
            "publish_tool_to_general": "Okay — promoting that tool to general.",
            "open_application": "Okay — opening that now.",
            "read_local_file": "Let me read that file.",
            "read_system_architecture": "Let me see.",
            "dispatch_research_swarm": "Sending that to the research swarm.",
            "dispatch_watchdog": "Okay — deploying a watchdog.",
            "kill_watchdog": "Okay — stopping that watchdog.",
            "save_script_to_library": "Okay — saving that script to the library.",
        }.get(tool_id, "Let me see.")
    log_debug("Conversation", f'Tool working ack ({tool_id}): "{phrase}"')
    set_subtitle(phrase)
    # Fire-and-forget so Piper plays while Ollama / web_search run on this thread.
    # Short filler acks are uninterruptible (avoid self-barge from speaker bleed).
    enqueue_speech(phrase, interruptible=False)
