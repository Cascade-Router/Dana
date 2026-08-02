"""ReAct / plan-execute loop for llama3.2 — LangChain ChatOllama + native tools."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from dana.schema import AgenticResult
from dana.tools.broker import IntentBroker, ToolValidationError, get_broker
from dana.tools.schema import ToolCall

REACT_MAX_ITERS = 3

# Spoken when the local Ollama HTTP endpoint is down / refuses connections.
OLLAMA_UNREACHABLE_SPEECH = (
    "I cannot reach the local Ollama service. Please start the Ollama server."
)


def is_ollama_connection_error(exc: BaseException) -> bool:
    """True for refused / unreachable Ollama (requests, httpx, or OS-level)."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(exc).__name__
    module = getattr(type(exc), "__module__", "") or ""
    if name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "NetworkError",
        "ConnectionError",
    }:
        return True
    if "httpx" in module or "httpcore" in module or "urllib3" in module:
        if "connect" in name.lower() or "timeout" in name.lower():
            return True
    text = str(exc).lower()
    needles = (
        "connection refused",
        "actively refused",
        "failed to establish",
        "name or service not known",
        "nodename nor servname",
        "cannot reach ollama",
        "all connection attempts failed",
        "10061",
        "connection reset",
    )
    if any(n in text for n in needles):
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_ollama_connection_error(cause)
    return False


# Process-wide mode: chat (default) | developer | vision | research.
donna_mode: str = "chat"
_DONNA_MODE_LOCK = threading.Lock()
_VALID_DONNA_MODES = frozenset({"chat", "developer", "vision", "research"})

DEVELOPER_MODE_ACK = "Developer mode active."
CHAT_MODE_ACK = "Chat mode active."
VISION_MODE_ACK = "Vision mode active."
RESEARCH_MODE_ACK = "Research mode active."
CHAT_MEMORY_CLEARED_ACK = "Memory cleared."

_LIGHTWEIGHT_CHAT_SYSTEM = (
    "You are Dānā, a local Windows control plane with direct access to desktop "
    "tools (Florence-2 vision, Win32 ROI, HITL ticket gate). You MUST construct "
    "an execution plan using your tools for any desktop or window requests. "
    "Do not issue generic refusals. "
    "In lightweight chat you have no tools bound — stay warm and concise for "
    "casual talk only (1–3 TTS sentences). "
    "CRITICAL: If the user asks about the desktop/window/screen, to log a ticket, "
    "interact with the system, read a file, or write code, you MUST escalate to "
    "the tool-graph route. Never invent tool results or say you cannot help with "
    "desktop control when tools exist on the ReAct path."
)

_CHAT_CAPABILITY_CARD = (
    "Capability card (informational only — tools are NOT bound in chat mode):\n"
    "- Modes: chat (current default for casual talk), developer (tools/code), "
    "vision (camera/screen), research (research swarm). "
    "Users may say 'switch to vision' / 'switch to developer' without the word 'mode'.\n"
    "- Local Ollama brain: this chat path only runs when Ollama is reachable.\n"
    "- Research store / filesystem / code edits / Ollama diagnostics escalate to "
    "the tool-graph — do not invent those capabilities here.\n"
    "- Short-term memory: prior user/assistant turns in this session are provided "
    "as separate messages below the system prompt."
)

# Isolated Memory Buffer (strictly for lightweight chat). Never shared with ReAct.
# Holds the last 5 conversational turns to prevent context window overflow.
CHAT_MEMORY_WINDOW_K = 5
chat_memory_buffer: deque[dict[str, str]] = deque(maxlen=CHAT_MEMORY_WINDOW_K)
_CHAT_MEMORY_LOCK = threading.Lock()

_MODE_SWITCH_DEVELOPER_RE = re.compile(
    r"(?:\b(?:switch\s+to|enter|enable|go\s+to|activate)\s+"
    r"(?:developer|agent)(?:\s+mode)?\b"
    r"|^\s*(?:please\s+)?(?:developer|agent)\s+mode\.?\s*$)",
    re.IGNORECASE,
)
_MODE_SWITCH_CHAT_RE = re.compile(
    r"(?:\b(?:switch\s+to|enter|enable|go\s+to|activate)\s+chat(?:\s+mode)?\b"
    r"|^\s*(?:please\s+)?chat\s+mode\.?\s*$)",
    re.IGNORECASE,
)
_MODE_SWITCH_VISION_RE = re.compile(
    r"(?:\b(?:switch\s+to|enter|enable|go\s+to|activate)\s+vision(?:\s+mode)?\b"
    r"|^\s*(?:please\s+)?vision\s+mode\.?\s*$)",
    re.IGNORECASE,
)
_MODE_SWITCH_RESEARCH_RE = re.compile(
    r"(?:\b(?:switch\s+to|enter|enable|go\s+to|activate)\s+research(?:\s+mode)?\b"
    r"|^\s*(?:please\s+)?research\s+mode\.?\s*$)",
    re.IGNORECASE,
)

# Greetings / small-talk — always stay on lightweight chat (never LangGraph).
_SMALL_TALK_RE = re.compile(
    r"(?i)^\s*(?:(?:hey|hi|hello)\s+)?(?:donna\s*[,!]?\s*)?"
    r"(?:"
    r"how\s+are\s+you(?:\s+doing)?(?:\s+today)?|"
    r"how'?s\s+it\s+going|"
    r"what'?s\s+up|"
    r"hello|hi|hey|"
    r"good\s+(?:morning|afternoon|evening)|"
    r"thanks?(?:\s+you)?|thank\s+you"
    r")"
    r"[?.!]?\s*$",
)

# Aggressive chat→tool escalation: system / filesystem / coding intents must
# never stay on the lightweight no-tools chat node.
_TOOL_GRAPH_INTENT_RE = re.compile(
    r"("
    r"\b(?:file|files|filepath|filename)\b|"
    r"\b(?:read|write|rewrite|overwrite)\b|"
    r"\b(?:code|coding|codebase)\b|"
    r"\b(?:terminal|shell|powershell|cmd)\b|"
    r"\b(?:run|running|execute|execution|exec)\b|"
    r"\b(?:system|subprocess)\b|"
    r"\b(?:script|scripts)\b|"
    r"\b(?:change|fix|update|patch|edit|modify)\b|"
    r"\b(?:tool|tools)\b|"
    r"\b(?:research\s+store|research\s+swarm)\b|"
    r"\b(?:ollama|olama|allama|olam)\b|"
    r"\b(?:local\s+server|brain\s+server)\b.*\b(?:online|offline|status|up|down)\b|"
    r"\b(?:online|offline|status)\b.*\b(?:ollama|olama|allama|local\s+server)\b|"
    r"\b(?:shell_execute|execute_powershell|fetch_webpage|file_editor|python_repl|"
    r"run_terminal_command|read_local_file)\b|"
    r"\b[\w./\\-]+\.(?:py|json|md|txt|log|csv|yml|yaml|toml|ini)\b|"
    # Desktop / actuator / log intents (mirrors agentic_planning._DESKTOP_PLAN_RE).
    r"\b(?:close|quit|kill|launch|open)\s+(?:[a-zA-Z0-9_-]+\s+)?"
    r"(?:app|application|discord|browser|window|program)\b|"
    r"\b(?:how\s+many|list|show|count)\s+(?:windows|tabs|apps)\b|"
    r"\b(?:open|show|read|retrack|view)\s+.*\b(?:logs?|runtime\.log|conversation\.log)\b|"
    # Vision / on-screen observe (must not soft-drop to filler chat).
    r"\b(?:what\s+you\s+see|see\s+on\s+(?:my\s+)?screen|on\s+my\s+screen)\b|"
    r"\b(?:describe|summarize)\b.*\b(?:screen|display|desktop|window)\b|"
    # Episodic / vault memory search intents (phrase-level; not bare ``memory``).
    r"\bepisodic\s+memory\b|"
    r"\b(?:search|query|recall|read)\s+(?:the\s+)?(?:episodic\s+|vault\s+)?memory\b|"
    r"\bread_vault_memory\b|"
    # Chroma / codebase vault cues — phrase-level (bare ``memory`` stays chat).
    r"\b(?:search\s+your\s+memory|search\s+memory|look\s+in\s+(?:the\s+)?vault|"
    r"search\s+(?:the\s+)?vault|query\s+chroma|search\s+chroma|look\s+up\s+chroma|"
    r"ingest\s+(?:this\s+)?(?:directory|folder|codebase)|"
    r"ingest\s+(?:the\s+)?(?:vault|chroma|memory|codebase)|"
    r"index\s+(?:this\s+)?(?:directory|folder|codebase)|"
    r"codebase\s+vault|ingest_local_directory|search_vault|"
    r"(?:vault|chroma)\s+(?:search|ingest|query|embeddings))\b"
    r")",
    re.IGNORECASE,
)


def requires_tool_graph(text: str) -> bool:
    """True when the utterance must use the tool-enabled MoA / ReAct graph.

    Lightweight chat remains for greetings and generic Q&A; any local execution,
    file, coding, or desktop/window/ticket intent forces the tool loop.
    """
    blob = (text or "").strip()
    t0 = time.perf_counter()
    decision = "lightweight"
    try:
        if not blob:
            return False
        # Greetings / "how are you" must never escalate to LangGraph.
        if _SMALL_TALK_RE.search(blob):
            decision = "lightweight_smalltalk"
            return False
        if bool(_TOOL_GRAPH_INTENT_RE.search(blob)):
            decision = "tool_graph"
            return True
        try:
            from dana.agentic_planning import desktop_plan_intent

            if desktop_plan_intent(blob):
                decision = "tool_graph_desktop"
                return True
        except Exception:  # noqa: BLE001
            pass
        return False
    finally:
        try:
            from dana.perf import log_perf

            log_perf(
                "intent_routing",
                (time.perf_counter() - t0) * 1000.0,
                decision=decision,
                preview=(blob or "")[:80],
            )
        except Exception:  # noqa: BLE001
            pass


_MODE_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:hey\s+)?donna\b[\s,.\-!:]*",
    re.IGNORECASE,
)
_CLEAR_CHAT_MEMORY_RE = re.compile(
    r"(?:"
    r"\b(?:clear|reset|wipe|forget)\s+(?:the\s+)?(?:chat\s+)?(?:memory|conversation|history)\b"
    r"|\b(?:reset|clear)\s+conversation\b"
    r"|\bclear\s+chat\s+memory\b"
    r")",
    re.IGNORECASE,
)


def get_donna_mode() -> str:
    """Return the active mode (``chat`` / ``developer`` / ``vision`` / ``research``)."""
    with _DONNA_MODE_LOCK:
        mode = (donna_mode or "chat").strip().lower()
    if mode == "agent":
        return "developer"
    if mode in _VALID_DONNA_MODES:
        return mode
    return "chat"


def get_voice_mode() -> str:
    """User conversational mode (Blackboard), immune to system-job escalations."""
    try:
        from dana.memory.blackboard import get_voice_session_mode

        return get_voice_session_mode()
    except Exception:  # noqa: BLE001
        return get_donna_mode()


def set_donna_mode(mode: str, *, as_voice: bool = True) -> str:
    """Set and return the normalized mode (``agent`` → ``developer``).

    When ``as_voice`` is True (mailroom / user switches), also persist
    ``voice_session_mode`` on the Blackboard. System job escalations should
    pass ``as_voice=False`` so they do not steal the live conversation mode.
    """
    global donna_mode
    raw = (mode or "").strip().lower()
    if raw == "agent":
        raw = "developer"
    if raw not in _VALID_DONNA_MODES:
        raw = "chat"
    with _DONNA_MODE_LOCK:
        donna_mode = raw
    if as_voice:
        try:
            from dana.memory.blackboard import set_voice_session_mode

            set_voice_session_mode(raw)
        except Exception:  # noqa: BLE001
            pass
    return raw


def restore_voice_mode() -> str:
    """Restore process mode from durable voice_session_mode after a system job."""
    voice = get_voice_mode()
    return set_donna_mode(voice, as_voice=False)


def parse_mode_switch(text: str) -> str | None:
    """Return a mode id when the utterance is a mode switch, else None."""
    # Prefer Cascade's deterministic keyword map (covers "stretch to vision", etc.).
    try:
        from dana.cascade_router import match_state_toggle

        hit = match_state_toggle(text)
        if hit is not None:
            return hit
    except Exception:  # noqa: BLE001
        pass
    blob = _normalize_mode_utterance(text)
    if not blob:
        return None
    if _MODE_SWITCH_DEVELOPER_RE.search(blob):
        return "developer"
    if _MODE_SWITCH_CHAT_RE.search(blob):
        return "chat"
    if _MODE_SWITCH_VISION_RE.search(blob):
        return "vision"
    if _MODE_SWITCH_RESEARCH_RE.search(blob):
        return "research"
    return None


def mode_switch_spoken_ack(mode: str) -> str:
    """Ack phrase for the mode being entered (WAV-cache friendly)."""
    normalized = (mode or "").strip().lower()
    if normalized in {"developer", "agent"}:
        return DEVELOPER_MODE_ACK
    if normalized == "vision":
        return VISION_MODE_ACK
    if normalized == "research":
        return RESEARCH_MODE_ACK
    return CHAT_MODE_ACK


def _normalize_mode_utterance(text: str) -> str:
    """Strip wake wrappers for mode / memory fast-path matching."""
    raw = (text or "").strip()
    if not raw:
        return ""
    try:
        cleaned = sanitize_voice_intent(raw) or raw
    except Exception:  # noqa: BLE001
        cleaned = raw
    blob = _MODE_WAKE_PREFIX_RE.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", blob).strip(" \t.,;:!-")


def parse_clear_chat_memory(text: str) -> bool:
    """True when the user asked to empty the lightweight chat rolling buffer."""
    blob = _normalize_mode_utterance(text)
    if not blob:
        return False
    return bool(_CLEAR_CHAT_MEMORY_RE.search(blob))


def _format_chat_memory_context() -> str:
    """Legacy flattened history block (debug / tests only). Prefer role messages."""
    with _CHAT_MEMORY_LOCK:
        turns = list(chat_memory_buffer)
    if not turns:
        return ""
    history_lines = [
        f"User: {turn.get('user', '')}\nDonna: {turn.get('dana', '')}"
        for turn in turns
        if (turn.get("user") or turn.get("donna"))
    ]
    if not history_lines:
        return ""
    return "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"


def chat_memory_as_role_messages() -> list[dict[str, str]]:
    """Role-separated rolling turns for the lightweight chat Ollama payload."""
    with _CHAT_MEMORY_LOCK:
        turns = list(chat_memory_buffer)
    out: list[dict[str, str]] = []
    for turn in turns:
        user = (turn.get("user") or "").strip()
        donna = (turn.get("donna") or "").strip()
        if user:
            out.append({"role": "user", "content": user})
        if donna:
            out.append({"role": "assistant", "content": donna})
    return out


def append_chat_memory_turn(user_text: str, assistant_text: str) -> None:
    """Append one ``{user, donna}`` turn into ``chat_memory_buffer``."""
    user = (user_text or "").strip()
    donna = (assistant_text or "").strip()
    if not user and not donna:
        return
    with _CHAT_MEMORY_LOCK:
        chat_memory_buffer.append({"user": user, "donna": donna})


def clear_chat_memory() -> bool:
    """Fast-path trigger to wipe the rolling buffer."""
    with _CHAT_MEMORY_LOCK:
        chat_memory_buffer.clear()
    return True


def chat_memory_size() -> int:
    """Number of stored conversational turns (max ``CHAT_MEMORY_WINDOW_K``)."""
    with _CHAT_MEMORY_LOCK:
        return len(chat_memory_buffer)


_DRAFT_CURSOR_TPM_RULE = (
    "TECHNICAL PRODUCT MANAGER RULE: When the user gives a high-level or casual "
    "voice command for a code change, you must act as a Technical Product Manager. "
    "Translate their vague request into a highly detailed technical prompt for the "
    "Cursor IDE. If the user does not provide file paths, use your reasoning to "
    "outline clear architectural goals, logic steps, and acceptance criteria. "
    "Pass your detailed architectural plan directly into the `context` argument of "
    "the function call. Do not ask the user for more details—expand their intent "
    "into a usable developer ticket."
)

_DRAFT_CURSOR_TERMINATION_RULE = (
    "TERMINATION RULE: Your sole job is to log the ticket using the tool. Once the "
    "`draft_cursor_prompt` tool returns a success message, you MUST immediately end "
    "your response with a simple confirmation (e.g., 'Ticket logged.'). Do NOT "
    "attempt to write code, solve the problem, or explain the architecture."
)

_INTERACTION_UX_RULE = (
    "INTERACTION_UX_RULE: You are a highly capable technical partner. "
    "After a tool successfully executes, output ONE brief, casual sentence "
    "confirming completion (e.g., 'The ticket is on the board.'). Never read raw "
    "code, JSON, or markdown out loud."
)

_R1_REASONING_RULE = (
    "R1_REASONING_RULE: You are a reasoning model. You MUST use your "
    "`<think> ... </think>` block to plan the architectural ticket and context "
    "payload. Immediately after your closing `</think>` tag, you MUST execute the "
    "`draft_cursor_prompt` tool natively. Do not output any conversational text "
    "outside the think block until the tool returns."
)

_TOOL_EXECUTION_RULE = (
    "TOOL_EXECUTION_RULE: You are an autonomous agent, not a conversational "
    "chatbot. You have access to functional tools. When instructed to log a ticket "
    "or perform an action, you MUST invoke the `draft_cursor_prompt` tool using the "
    "system's native tool-calling schema (JSON/function call). "
    "Do NOT output bash commands, CLI scripts, or markdown code blocks instructing "
    "the user how to do it. You must physically execute the function yourself. "
    "Once the tool returns a success message, output your ONE sentence summary and "
    "terminate."
)

_STRICT_TOOL_ENFORCEMENT_RULE = (
    "STRICT_TOOL_ENFORCEMENT_RULE: Your sole authorized method for logging tickets "
    "is the `draft_cursor_prompt` tool. You are strictly FORBIDDEN from generating "
    "bash commands, CLI snippets, or markdown code blocks as a substitute for tool "
    "execution. If you believe a CLI command is part of the solution, you MUST "
    "include that command INSIDE the 'context' argument of the `draft_cursor_prompt` "
    "tool, not as plain text in your response. If you do not call the tool, you have "
    "failed the task."
)

_EXPLICIT_TOOL_INVOCATION_RULE = (
    "CRITICAL: If the user explicitly asks to use a tool (e.g., draft_cursor_prompt), "
    "you MUST invoke that tool's JSON schema in your response before outputting a "
    "final natural language answer. Do not claim a task or ticket has been logged "
    "unless the tool execution trace confirms it."
)

_VOICE_SANITIZER_RULE = (
    "VOICE_SANITIZER_RULE: When processing input from voice mode, you must filter out "
    "conversational wrapper text (e.g., 'use the draft cursor prompt tool to log a "
    "ticket', 'can you please log...'). Extract only the underlying technical intent. "
    "When the user omits file paths, use the voice topic map for Target Files only: "
    "audio/audio pipeline/glitches → dana/core_agent.py; "
    "cursor handling/deepseek navigation → dana/cascade_router.py and dana/agentic.py; "
    "patch ledger/cursor prompt handling → dana_security/patch_ledger.md and "
    "dana/tools/broker.py. You must invent concrete refactoring steps from the user's "
    "specific request — never reuse generic template steps."
)

# Voice topic → target files only (LLM invents refactoring steps from the user request).
_VOICE_TOPIC_FILE_MAP: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("audio pipeline", "audio glitch", "glitches", " audio ", "tts", "piper", "whisper", "vad"),
        ("dana/core_agent.py",),
    ),
    (
        (
            "cursor handling",
            "deepseek navigation",
            "deepseek cursor",
            "cursor lag",
            "cursor navigation",
            "deepseek routing",
        ),
        ("dana/cascade_router.py", "dana/agentic.py"),
    ),
    (
        (
            "patch ledger",
            "cursor prompt handling",
            "draft cursor prompt handling",
        ),
        ("dana_security/patch_ledger.md", "dana/tools/broker.py"),
    ),
)

_VOICE_WRAPPER_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bdonna[,:]?\s*"),
    re.compile(
        r"(?i)\buse the draft[_\s-]*cursor[_\s-]*prompt tool to log "
        r"(?:a )?(?:self-improvement )?ticket(?:\s+to)?\b"
    ),
    re.compile(
        r"(?i)\blog a self-improvement ticket(?:\s+to)?\b"
    ),
    re.compile(r"(?i)\blog a ticket(?:\s+to)?\b"),
    re.compile(r"(?i)\bcan you(?:\s+please)?\b"),
    re.compile(r"(?i)\bwould you(?:\s+please)?\b"),
    re.compile(r"(?i)\bplease\b"),
    re.compile(r"(?i)\bcould you(?:\s+please)?\b"),
)
_TECH_INTENT_LABEL_RE = re.compile(
    r"(?i)(?:\*\*\s*)?Technical\s+intent:\s*(?:\*\*\s*)?"
)
_ENRICHED_CONTEXT_RE = re.compile(
    r"(?is)\*\*Technical intent:\*\*.+\*\*Target Files:\*\*"
)
_REPEAT_CLAUSE_RE = re.compile(
    r"(?i)\b(.{20,}?)(?:[.!]?\s+)\1\b"
)


def _collapse_self_duplication(text: str) -> str:
    """Collapse consecutive repeated clauses (common after double-enrich)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    prev = None
    while prev != t:
        prev = t
        t = _REPEAT_CLAUSE_RE.sub(r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" \t.,;:!-")


def sanitize_voice_intent(text: str) -> str:
    """Strip conversational voice wrappers; keep the technical intent phrase."""
    t = (text or "").strip()
    if not t:
        return ""
    t = _TECH_INTENT_LABEL_RE.sub(" ", t)
    for pat in _VOICE_WRAPPER_RES:
        t = pat.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" \t.,;:!-")
    return _collapse_self_duplication(t)


def _full_sentence_boundary(text: str) -> str:
    """Keep the first complete sentence; never mid-word hard-cap."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    # Prefer first sentence ending; otherwise keep the full phrase.
    m = re.match(r"(.+?[.!?])(?:\s|$)", t)
    if m:
        return m.group(1).strip()
    # Single line without terminator — keep whole line (no char cap).
    return t.split("\n", 1)[0].strip()


def _looks_enriched_context(context: str) -> bool:
    return bool(_ENRICHED_CONTEXT_RE.search(context or ""))


def _extract_enriched_intent(context: str) -> str:
    m = re.search(
        r"(?im)^\s*\*\*Technical intent:\*\*\s*(.+?)\s*$",
        context or "",
    )
    if not m:
        return ""
    return sanitize_voice_intent(m.group(1))


def _primary_intent_source(raw_text: str, objective: str, context: str) -> str:
    """Pick one scrubbed source — never concatenate overlapping copies."""
    raw = (raw_text or "").strip()
    obj = (objective or "").strip()
    ctx = (context or "").strip()
    if _looks_enriched_context(ctx):
        return _extract_enriched_intent(ctx) or sanitize_voice_intent(obj) or raw
    # Broker often sets context=full raw and objective=first sentence of same raw.
    if raw:
        if (not obj or obj.lower() in raw.lower()) and (
            not ctx or ctx == raw or ctx.lower() in raw.lower()
        ):
            return raw
        if ctx and ctx != raw and obj and obj.lower() not in ctx.lower():
            return raw
        return raw
    if obj and ctx and (obj.lower() in ctx.lower() or ctx.lower() in obj.lower()):
        return ctx if len(ctx) >= len(obj) else obj
    return " ".join(x for x in (obj, ctx) if x)


def _dedupe_enriched_context(context: str) -> str:
    """Keep a single Technical intent / Target Files block (preserve LLM steps if any)."""
    ctx = (context or "").strip()
    if not ctx:
        return ""
    intent = _extract_enriched_intent(ctx)
    targets_m = re.search(
        r"(?im)^\s*\*\*Target Files:\*\*\s*(.+?)\s*$",
        ctx,
    )
    targets = targets_m.group(1).strip() if targets_m else ""
    # Preserve any LLM-authored body after Target Files (steps/notes), without templates.
    rest_lines: list[str] = []
    past_targets = False
    for line in ctx.splitlines():
        if re.match(r"(?im)^\s*\*\*Target Files:\*\*", line):
            past_targets = True
            continue
        if not past_targets:
            continue
        if re.match(r"(?im)^\s*\*\*Technical intent:\*\*", line):
            continue
        rest_lines.append(line)
    rest = "\n".join(rest_lines).strip()
    if not intent and not targets:
        parts = _TECH_INTENT_LABEL_RE.split(ctx)
        return _collapse_self_duplication(" ".join(p for p in parts if p.strip()))
    lines = [
        f"**Technical intent:** {intent or 'Self-improvement code change'}",
        f"**Target Files:** {targets or 'dana/agentic.py'}",
    ]
    if rest:
        lines.extend(["", rest])
    return "\n".join(lines)


def map_voice_topics(text: str) -> list[str]:
    """Return target file paths from the voice topic dictionary (files only)."""
    blob = f" {(text or '').lower()} "
    targets: list[str] = []
    seen_t: set[str] = set()
    for keywords, files in _VOICE_TOPIC_FILE_MAP:
        if not any(k in blob for k in keywords):
            continue
        for f in files:
            if f not in seen_t:
                seen_t.add(f)
                targets.append(f)
    return targets


def enrich_draft_cursor_args(
    *,
    raw_text: str = "",
    objective: str = "",
    context: str = "",
) -> dict[str, str]:
    """Sanitize voice wrappers and auto-map Target Files only.

    Does **not** inject hardcoded refactoring steps — those must come from the
    user's request / LLM. Idempotent on already-enriched ticket bodies.
    """
    # Already enriched by broker foresight / a prior loop — do not append again.
    if _looks_enriched_context(context):
        cleaned_ctx = _dedupe_enriched_context(context)
        intent = _extract_enriched_intent(cleaned_ctx) or sanitize_voice_intent(
            objective
        )
        obj = sanitize_voice_intent(objective) or intent or "Self-improvement code change"
        # Full sentence only — never hard-cap mid-word.
        obj = _full_sentence_boundary(obj)
        targets_m = re.search(
            r"(?im)^\s*\*\*Target Files:\*\*\s*(.+?)\s*$",
            cleaned_ctx,
        )
        targets = targets_m.group(1).strip() if targets_m else "dana/agentic.py"
        return {
            "objective": obj,
            "context": cleaned_ctx,
            "target_files": targets,
        }

    source = _primary_intent_source(raw_text, objective, context)
    intent = sanitize_voice_intent(source) or "Self-improvement code change"
    obj = sanitize_voice_intent(objective) or intent
    obj = _full_sentence_boundary(obj)

    targets = map_voice_topics(source or intent)
    if not targets:
        targets = ["dana/agentic.py"]

    # Prefer LLM/user-authored context body when it already carries detail.
    cleaned_ctx = (context or "").strip()
    if cleaned_ctx and cleaned_ctx != (raw_text or "").strip():
        # Strip wrappers but keep structured detail the model already wrote.
        body = cleaned_ctx
        if _looks_enriched_context(body):
            body = _dedupe_enriched_context(body)
        else:
            # If body is just a second copy of the voice phrase, drop it.
            scrubbed = sanitize_voice_intent(body)
            if scrubbed and scrubbed.lower() in intent.lower():
                body = ""
            elif scrubbed:
                body = scrubbed
    else:
        body = ""

    ctx_lines = [
        f"**Technical intent:** {intent}",
        f"**Target Files:** {', '.join(targets)}",
    ]
    if body and not _looks_enriched_context(body):
        # Dynamic notes/steps from the caller — never template stubs.
        if "refactoring steps" in body.lower() or body.startswith("**"):
            ctx_lines.extend(["", body])
        else:
            ctx_lines.extend(["", f"**Request detail:** {body}"])
    elif body and _looks_enriched_context(body):
        # Merge: keep deduped body (may include LLM steps) over bare intent+targets.
        return {
            "objective": obj,
            "context": _dedupe_enriched_context(body),
            "target_files": ", ".join(targets),
        }

    return {
        "objective": obj,
        "context": "\n".join(ctx_lines),
        "target_files": ", ".join(targets),
    }


# Friendly TTS phrases for astream on_tool_start (never speak raw tool ids).
_TOOL_TTS_FRIENDLY: dict[str, str] = {
    "draft_cursor_prompt": "Drafting architectural ticket...",
    "web_search": "Searching the web...",
    "dispatch_research_swarm": "Sending that to the research swarm...",
    "dispatch_watchdog": "Setting up a watchdog...",
    "kill_watchdog": "Stopping that watchdog...",
    "architect_new_tool": "Building a new tool...",
    "read_local_file": "Reading that file...",
    "read_vault_memory": "Checking memory...",
    "write_vault_memory": "Saving that to memory...",
    "run_terminal_command": "Running that in the terminal...",
    "open_application": "Opening that now...",
    "capture_and_analyze_screen": "Looking at your screen...",
    "evaluate_slide_and_type": "Reviewing the slide...",
    "dispatch_titan_repair": "Drafting repair patches...",
    "delegate_to_cursor": "Preparing a Cursor handoff...",
    "list_todo_basket": "Checking the todo list...",
    "flush_memory": "Clearing short-term memory...",
    "read_system_architecture": "Checking the architecture...",
    "describe_spatial_scene": "Looking around...",
    "execute_os_keystrokes": "Typing that now...",
}

_STREAM_TTS_NOISE_RE = re.compile(
    r"</?tool\b[^>]*>|"
    r"<\|eot_id\|>|"
    r"<\|[^|>]+?\|>|"
    r"</?\|?tool_calls?\|?>",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class R1ThinkExtract:
    """Separated DeepSeek-R1 reasoning vs clean downstream payload."""

    think_text: str
    clean_text: str


def extract_r1_think_blocks(text: str) -> R1ThinkExtract:
    """Split R1 ``<think>...</think>`` reasoning from the clean response.

    Handles:
    - multiple closed think blocks
    - unclosed ``<think>`` (captures through end-of-string — early halt)
    - optional attributes on the opening tag (``<think ...>``)
    - empty extracts (no tags / empty bodies): ``clean_text`` is the full
      usable string so Stage-3 Pydantic guards still receive a payload
    """
    if not text:
        return R1ThinkExtract("", "")

    raw = str(text)
    lower = raw.lower()
    thinks: list[str] = []
    clean_parts: list[str] = []
    pos = 0
    n = len(raw)

    while pos < n:
        idx = lower.find("<think", pos)
        if idx < 0:
            clean_parts.append(raw[pos:])
            break
        # Only treat real tags: <think> or <think ...>
        after = lower[idx + 6 : idx + 7] if idx + 6 < n else ""
        if after not in {">", " ", "\t", "\n", "\r"}:
            clean_parts.append(raw[pos : idx + 6])
            pos = idx + 6
            continue
        clean_parts.append(raw[pos:idx])
        gt = raw.find(">", idx)
        if gt < 0:
            # Malformed open tag — remainder is reasoning.
            thinks.append(raw[idx:].strip())
            break
        body_start = gt + 1
        close_idx = lower.find("</think>", body_start)
        if close_idx < 0:
            # Unclosed think — capture through EOF (generation halt).
            thinks.append(raw[body_start:].strip())
            break
        thinks.append(raw[body_start:close_idx].strip())
        pos = close_idx + len("</think>")

    think_text = "\n\n".join(t for t in thinks if t)
    clean = "".join(clean_parts)
    clean = re.sub(r"</think\s*>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    # Empty extract fallback: if no usable clean payload, forward full raw
    # so downstream Pydantic guards can reject/retry.
    if not clean and raw.strip():
        clean = raw.strip()
    return R1ThinkExtract(think_text=think_text, clean_text=clean)


def strip_r1_think_blocks(text: str) -> str:
    """Remove DeepSeek-R1 think blocks (closed or unclosed); return clean text only."""
    if not text:
        return ""
    return extract_r1_think_blocks(text).clean_text


class ThinkBlockTtsFilter:
    """Stateful filter: strip R1 think blocks across streamed TTS chunks."""

    def __init__(self) -> None:
        self._in_think = False
        self._hold = ""

    def reset(self) -> None:
        self._in_think = False
        self._hold = ""

    @staticmethod
    def _partial_tag_suffix(buf: str, tag: str) -> str:
        """If ``buf`` ends with a proper prefix of ``tag``, return that suffix."""
        bl = buf.lower()
        tl = tag.lower()
        max_n = min(len(buf), len(tag) - 1)
        for n in range(max_n, 0, -1):
            if tl.startswith(bl[-n:]):
                return buf[-n:]
        return ""

    def feed(self, text: str) -> str:
        """Return only speakable text (outside think blocks) from ``text``."""
        if not text:
            return ""
        data = f"{self._hold}{text}"
        self._hold = ""
        out: list[str] = []
        i = 0
        lower = data.lower()
        open_l = _THINK_OPEN.lower()
        close_l = _THINK_CLOSE.lower()
        while i < len(data):
            if self._in_think:
                idx = lower.find(close_l, i)
                if idx < 0:
                    self._hold = self._partial_tag_suffix(data[i:], _THINK_CLOSE)
                    break
                i = idx + len(_THINK_CLOSE)
                self._in_think = False
                continue
            idx = lower.find(open_l, i)
            if idx < 0:
                partial = self._partial_tag_suffix(data[i:], _THINK_OPEN)
                if partial:
                    out.append(data[i : len(data) - len(partial)])
                    self._hold = data[len(data) - len(partial) :]
                else:
                    out.append(data[i:])
                break
            out.append(data[i:idx])
            i = idx + len(_THINK_OPEN)
            self._in_think = True
        return "".join(out)


# LangGraph checkpointer — consistent thread_id keeps cross-turn graph memory.
_REACT_CHECKPOINTER: Any | None = None
_REACT_THREAD_ID = "donna-react-session"


def _react_checkpointer() -> Any:
    global _REACT_CHECKPOINTER
    if _REACT_CHECKPOINTER is None:
        from langchain_core.load.load import Reviver
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        import langgraph.checkpoint.serde.jsonplus as _jp_serde

        # Explicit allowlist suppresses LangChainPendingDeprecationWarning on load().
        _jp_serde.LC_REVIVER = Reviver(allowed_objects="all")
        _REACT_CHECKPOINTER = MemorySaver(serde=JsonPlusSerializer())
    return _REACT_CHECKPOINTER


def _friendly_tool_tts(tool_name: str) -> str:
    """Map a tool id to a short spoken status line for voice UX."""
    key = (tool_name or "").strip()
    return _TOOL_TTS_FRIENDLY.get(key, "Working on that...")


def _stream_chunk_for_tts(chunk: Any) -> str:
    """Extract speakable text from an on_chat_model_stream chunk."""
    piece = ""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        piece = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        piece = "".join(parts)
    else:
        piece = str(content or "") if content is not None else ""
    piece = _STREAM_TTS_NOISE_RE.sub("", piece)
    piece = strip_r1_think_blocks(piece)
    piece = strip_protocol_speech_anchors(piece)
    if (
        looks_like_raw_json_speech(piece)
        or looks_like_raw_tool_observation(piece)
        or looks_like_ticket_receipt_speech(piece)
    ):
        return ""
    return piece


class StreamSentenceTtsBuffer:
    """Accumulate streaming tokens; emit only complete sentences to TTS."""

    _END_RE = re.compile(r"([.!?])(?:\s+|$)|(\n+)")

    def __init__(self) -> None:
        self._buf = ""
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._buf = ""

    def feed(self, piece: str) -> list[str]:
        """Append ``piece``; return any newly completed sentences."""
        text = piece or ""
        if not text:
            return []
        out: list[str] = []
        with self._lock:
            self._buf += text
            while True:
                m = self._END_RE.search(self._buf)
                if not m:
                    break
                end = m.end()
                sentence = self._buf[:end].strip()
                self._buf = self._buf[end:]
                if sentence:
                    out.append(sentence)
            # Soft cap: force-flush if buffer grows without punctuation.
            if len(self._buf) >= 180 and self._buf[-1:].isspace():
                chunk = self._buf.strip()
                self._buf = ""
                if chunk:
                    out.append(chunk)
        return out

    def flush(self) -> str:
        """Return and clear any incomplete remainder."""
        with self._lock:
            rem = self._buf.strip()
            self._buf = ""
            return rem


_stream_sentence_tts = StreamSentenceTtsBuffer()
_STREAM_TTS_AGENT = "broker"
_STREAM_TTS_AGENT_LOCK = threading.Lock()


def set_stream_tts_agent(agent_id: str | None) -> str:
    """Stage 8.8 — bind streaming / callback TTS to an active agent persona."""
    global _STREAM_TTS_AGENT
    try:
        from dana.audio.multi_voice_tts import (
            normalize_agent_id,
            set_active_tts_agent,
        )

        aid = normalize_agent_id(agent_id) or "broker"
        set_active_tts_agent(aid)
    except Exception:  # noqa: BLE001
        aid = (agent_id or "broker").strip().lower() or "broker"
    with _STREAM_TTS_AGENT_LOCK:
        _STREAM_TTS_AGENT = aid
    return aid


def get_stream_tts_agent() -> str:
    with _STREAM_TTS_AGENT_LOCK:
        return _STREAM_TTS_AGENT


def agent_id_from_label(label: str | None) -> str:
    """Map bureaucratic labels (MoA_Reasoner, Vision_Agent, …) → voice agent_id."""
    raw = (label or "").strip().lower()
    if not raw:
        return "broker"
    if "moa" in raw or "deepseek" in raw or "reasoner" in raw:
        return "moa"
    if "jason" in raw or "cto" in raw:
        return "jason"
    if "vision" in raw or "yolo" in raw or "florence" in raw:
        return "vision"
    if "typist" in raw:
        return "typist"
    if "mail" in raw:
        return "broker"
    return "broker"


def reset_stream_sentence_tts() -> None:
    _stream_sentence_tts.reset()


def flush_stream_sentence_tts() -> bool:
    """Speak any leftover stream buffer (end of turn / shutdown). Returns True if spoken."""
    rem = _stream_sentence_tts.flush()
    if rem:
        _enqueue_tts_nonblocking(rem)
        return True
    return False


def end_stream_sentence_tts() -> bool:
    """Terminate streaming TTS: flush remainder and clear the sentence buffer.

    Call when ReAct finishes an iteration without a tool call so producers do not
    wait on an open stream buffer / speech_idle latch.
    """
    spoken = flush_stream_sentence_tts()
    reset_stream_sentence_tts()
    return spoken


def feed_stream_tts(piece: str, *, agent_id: str | None = None) -> int:
    """Buffer streaming tokens; enqueue only complete sentences. Returns count.

    Sentence-level chunking lets Piper/pyttsx3 start before the LangGraph node
    finishes emitting the full token stream (Stage 8.8 latency fix).
    """
    sentences = _stream_sentence_tts.feed(piece or "")
    aid = agent_id or get_stream_tts_agent()
    for sentence in sentences:
        _enqueue_tts_nonblocking(sentence, agent_id=aid)
    return len(sentences)


def _enqueue_tts_nonblocking(
    phrase: str,
    *,
    agent_id: str | None = None,
) -> None:
    """Dispatch TTS without blocking the async graph / event stream."""
    text = (phrase or "").strip()
    if not text:
        return
    aid = agent_id or get_stream_tts_agent()

    def _run() -> None:
        try:
            from dana.core_agent import enqueue_speech

            enqueue_speech(text, agent_id=aid)
        except Exception:  # noqa: BLE001
            pass

    try:
        threading.Thread(target=_run, daemon=True, name="donna-react-tts").start()
    except Exception:  # noqa: BLE001
        pass


def _first_sentence(text: str, *, limit: int = 120) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""
    m = re.search(r"([^.!?]+[.!?])", raw)
    snippet = (m.group(1) if m else raw).strip()
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet


_FINAL_RE = re.compile(
    r"^\s*(?:FINAL|Final|final)\s*[:：]\s*(.*)\s*$",
    re.DOTALL,
)
_TOOL_LINE_RE = re.compile(
    r"^\s*(?:TOOL|Tool|tool|Action|ACTION)\s*[:：]\s*(.+?)\s*$",
    re.DOTALL,
)
# TTS hygiene: models occasionally leak protocol markers into spoken text.
_TTS_TOOL_LEAK_RE = re.compile(
    r"(?:^|\s)(?:TOOL|Action|FINAL)\s*[:：]\s*.*$",
    re.IGNORECASE | re.DOTALL,
)
_RAW_TOOL_OBS_RE = re.compile(
    r"^\s*(?:OK|ERROR|LOCKED|SpatialIR)\s*[:＝=]",
    re.IGNORECASE,
)
# Ledger / draft_cursor receipts must never be spoken (file + console only).
_TICKET_RECEIPT_SPEECH_RE = re.compile(
    r"(?i)("
    r"ticket\s+receipt|"
    r"ticket\s+name\s*:|"
    r"technical\s+intent\s*:|"
    r"target\s+files?\s*:|"
    r"refactoring\s+steps?\s*:|"
    r"acceptance\s+criteria\s*:|"
    r"cursor\s+receipt|"
    r"verification\s+note|"
    r"ticket\s+added\s+to\s+patch[_ ]?ledger|"
    r"patch_ledger\.md|"
    r"status\s+is\s+\[?pending\]?|"
    r"awaiting\s+compilation|"
    r"date\s+drafted\s*:|"
    r"security\s+&\s+guardrails"
    r")"
)
_DRAFT_CURSOR_OBS_OK_RE = re.compile(
    r"(?i)ticket\s+added|patch_ledger\.md|status\s+is\s+pending"
)
_SYSTEM_META_HALLUCINATION_RE = re.compile(
    r"(?:mistake|error|problem|issue|bug)\s+in\s+(?:your|the|my)\s+"
    r"system\s+(?:prompt|instructions|message)|"
    r"system\s+prompt\s+(?:is|seems|appears|looks)\s+"
    r"(?:wrong|broken|invalid|confused|incorrect)|"
    r"(?:mistake|error|problem|issue|bug)\s+with\s+(?:your|the|my)\s+"
    r"(?:tools?\.json|config(?:uration)?|system\s+(?:prompt|instructions|message))",
    re.IGNORECASE,
)
# Fixture / sandbox doc dumps must never be spoken unless the user explicitly asked.
_CONFIDENTIAL_FIXTURE_LEAK_RE = re.compile(
    r"CONFIDENTIAL\s+STATUS\s+REPORT\s*[-–—]?\s*PROJECT\s+OMEGA|"
    r"Lead\s+Engineer:\s*Narges|"
    r"secure\s+vault\s+architecture\s+is\s+fully\s+operational|"
    r"multi-agent\s+swarm\s+deployment\s+has\s+encountered\s+a\s+latency",
    re.IGNORECASE,
)
# Few-shot / roleplay leakage from llama3.2 (must never reach PiperTTS).
_DIALOG_ROLE_LINE_RE = re.compile(
    r"^\s*(?:User|Me|Human|Question|Example|System|Prompt)\s*[:：]\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)
_ARROW_SPEAK_LINE_RE = re.compile(
    r"^\s*(?:→|->)\s*(?:then\s+)?speak\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_ARROW_ROUTE_LINE_RE = re.compile(
    r"^\s*(?:→|->)\s*(?:call|do\s+not|never|use|prefer|then\s+call)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_LABEL_LINE_RE = re.compile(
    r"^\s*(?:Answer|Donna|Assistant|Response|Final)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_USER_ECHO_RE = re.compile(
    r"\bUser\s*[:：]\s*(['\"].*?['\"]|\S+)",
    re.IGNORECASE,
)
_GENERIC_GREETING_RE = re.compile(
    r"^\s*(?:hi(?:\s+there)?|hello(?:\s+there)?|hey(?:\s+there)?|howdy)"
    r"(?:\s*[!.]*)?\s*$",
    re.IGNORECASE,
)
_MD_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
# Aggressive catch for leaked tool-call / schema JSON that must never hit Piper TTS.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_TOOLISH_JSON_KEYS = frozenset(
    {
        "name",
        "tool",
        "tool_id",
        "parameters",
        "arguments",
        "args",
        "tool_calls",
        "function",
    }
)

# Content-JSON recovery allowlist — cuts false-positive tool dispatches.
_CONTENT_TOOL_ALLOWLIST = frozenset(
    {
        "draft_cursor_prompt",
        "web_search",
        "read_local_file",
        "run_terminal_command",
        "architect_new_tool",
        "dispatch_research_swarm",
        "dispatch_watchdog",
        "kill_watchdog",
        "read_vault_memory",
        "write_vault_memory",
        "flush_memory",
        "open_application",
        "list_todo_basket",
        "delegate_to_cursor",
        "dispatch_titan_repair",
        "capture_and_analyze_screen",
        "execute_os_keystrokes",
        "evaluate_slide_and_type",
        "read_system_architecture",
        "publish_tool_to_general",
        "save_script_to_library",
    }
)


def looks_like_ticket_receipt_speech(text: str) -> bool:
    """True when text is a patch-ledger / draft_cursor receipt (never for TTS)."""
    return bool(_TICKET_RECEIPT_SPEECH_RE.search(text or ""))


# Exact TTS string — must match ``dana/assets/audio_cache/the_ticket_is_on_the_board.wav``.
DRAFT_CURSOR_UX_ACK = "The ticket is on the board."


def draft_cursor_tool_succeeded(
    *,
    last_obs: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
) -> bool:
    """True when ``draft_cursor_prompt`` wrote a ticket successfully this turn."""
    for row in tool_trace or []:
        if str(row.get("tool") or "") != "draft_cursor_prompt":
            continue
        obs = str(row.get("observation") or "")
        if obs.upper().startswith("ERROR"):
            continue
        if _DRAFT_CURSOR_OBS_OK_RE.search(obs) or looks_like_ticket_receipt_speech(obs):
            return True
    obs = (last_obs or "").strip()
    if obs and not obs.upper().startswith("ERROR") and _DRAFT_CURSOR_OBS_OK_RE.search(obs):
        return True
    return False


def draft_cursor_spoken_ack(reply_lang: str = "en") -> str:
    """INTERACTION_UX short confirmation after a successful ledger write.

    Always English so Piper hits the static WAV cache (never live-synth / LLM prose).
    ``reply_lang`` is accepted for call-site compatibility only.
    """
    _ = reply_lang
    return DRAFT_CURSOR_UX_ACK


def log_tool_receipt_console(observation: str, *, tool_id: str = "") -> None:
    """Print tool receipts to the console logger (never enqueue for TTS)."""
    obs = (observation or "").strip()
    if not obs:
        return
    label = (tool_id or "tool").strip() or "tool"
    try:
        from dana.logging import log

        log("Ledger" if "draft_cursor" in label or "patch_ledger" in obs.lower() else "Tool", obs)
    except Exception:  # noqa: BLE001
        print(f"[{label}] {obs}", flush=True)


def looks_like_raw_tool_observation(text: str) -> bool:
    """True when spoken text is clearly a tool IR dump, not a user-facing reply."""
    t = (text or "").strip()
    if not t:
        return False
    if looks_like_ticket_receipt_speech(t):
        return True
    if _RAW_TOOL_OBS_RE.match(t):
        return True
    if re.search(
        r"\b(?:naming_fix|replacements=\d+|changed=(?:true|false))\b",
        t,
        re.I,
    ):
        return True
    if t.startswith("{") and (
        '"tool"' in t or '"tool_id"' in t or '"name"' in t or '"parameters"' in t
    ):
        return True
    return False


def looks_like_raw_json_speech(text: str) -> bool:
    """True when the reply is (or is dominated by) a JSON object / tool-call blob."""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("```"):
        inner = _MD_FENCE_RE.search(t)
        if inner:
            t = inner.group(1).strip()
    if not (t.startswith("{") or t.startswith("[")):
        # Also catch leading prose + JSON tool call dumps.
        m = _JSON_OBJECT_RE.search(t)
        if not m:
            return False
        # If most of the string is the JSON blob, treat as leak.
        blob = m.group(0)
        if len(blob) < 8 or len(blob) < 0.6 * len(t):
            return False
        t = blob
    try:
        data = json.loads(t)
    except Exception:
        # Unparseable brace soup still must not be spoken.
        return t[:1] in "{[" and ('"' in t or "'" in t)
    if isinstance(data, dict):
        keys = {str(k).lower() for k in data.keys()}
        if keys & _TOOLISH_JSON_KEYS:
            return True
        # Pure JSON object with no conversational keys → never speak.
        if not any(k in keys for k in ("answer", "final", "message", "text", "reply")):
            return True
    if isinstance(data, list) and data:
        return True
    return False


def strip_raw_json_from_speech(text: str) -> str:
    """Remove fenced / inline JSON tool-call blobs; return leftover prose (may be empty)."""
    raw = (text or "").strip()
    if not raw:
        return ""
    # Drop markdown fences first.
    raw = _MD_FENCE_RE.sub("", raw).strip()
    if looks_like_raw_json_speech(raw):
        return ""
    # Strip residual inline JSON objects that look like tool calls.
    def _sub(match: re.Match[str]) -> str:
        blob = match.group(0)
        try:
            data = json.loads(blob)
        except Exception:
            return blob
        if isinstance(data, dict):
            keys = {str(k).lower() for k in data.keys()}
            if keys & _TOOLISH_JSON_KEYS:
                return ""
        return blob

    cleaned = _JSON_OBJECT_RE.sub(_sub, raw)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \t-–—|,")
    return cleaned


def _parse_content_tool_call(text: str) -> dict[str, Any] | None:
    """If the model dumped a tool call into ``content``, recover name+args.

    Local models often emit raw JSON (``name``/``parameters`` or ``args``) instead
    of LangChain structured ``tool_calls``. Also accepts markdown-fenced JSON and
    bare ``{objective, context}`` payloads for ``draft_cursor_prompt``.
    """
    for data in _iter_json_dicts(text):
        parsed = _normalize_tool_payload(data)
        if parsed is not None:
            return parsed
    return None


def _iter_json_dicts(text: str) -> list[dict[str, Any]]:
    """Collect candidate JSON objects from prose / fences / brace slices."""
    raw = (text or "").strip()
    if not raw:
        return []
    candidates: list[str] = []
    for fence in _MD_FENCE_RE.finditer(raw):
        body = (fence.group(1) or "").strip()
        if body:
            candidates.append(body)
    candidates.append(raw)
    # Greedy outer object (handles nested parameters/context).
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    # Additional non-greedy scans for multiple sibling objects.
    depth = 0
    begin = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and begin >= 0:
                candidates.append(raw[begin : i + 1])
                begin = -1

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        key = cand.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            data = json.loads(key)
        except Exception:
            # Tolerate trailing commas / single quotes lightly via repair pass.
            try:
                repaired = re.sub(r",\s*}", "}", key)
                repaired = re.sub(r",\s*]", "]", repaired)
                data = json.loads(repaired)
            except Exception:
                continue
        if isinstance(data, dict):
            out.append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    out.append(item)
    return out


def _normalize_tool_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Map heterogeneous tool JSON shapes → LangChain-style tool_call dict."""
    if not isinstance(data, dict):
        return None

    # OpenAI-style: {"tool_calls":[{"function":{"name":...,"arguments":...}}]}
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            nested = _normalize_tool_payload(item)
            if nested is not None:
                return nested
            fn = item.get("function")
            if isinstance(fn, dict):
                nested = _normalize_tool_payload(
                    {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or fn.get("parameters"),
                    }
                )
                if nested is not None:
                    return nested

    # {"function": {"name": "...", "arguments": {...}}}
    fn = data.get("function")
    if isinstance(fn, dict) and (fn.get("name") or fn.get("arguments") or fn.get("parameters")):
        return _normalize_tool_payload(
            {
                "name": fn.get("name"),
                "arguments": fn.get("arguments") or fn.get("parameters") or {},
            }
        )

    name = str(
        data.get("name")
        or data.get("tool")
        or data.get("tool_id")
        or data.get("toolName")
        or ""
    ).strip()
    args = (
        data.get("parameters")
        or data.get("arguments")
        or data.get("args")
        or data.get("input")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"text": args}
    if not isinstance(args, dict):
        args = {}

    # Bare draft_cursor_prompt payload: {"objective": "...", "context": "..."}
    if not name and (
        "objective" in data or "context" in data
    ) and not any(k in data for k in ("query", "command", "path", "goal")):
        obj = str(data.get("objective") or "").strip()
        ctx = str(data.get("context") or "").strip()
        if obj or ctx:
            name = "draft_cursor_prompt"
            args = {
                "objective": obj or ctx[:200],
                "context": ctx or obj,
            }

    if not name:
        return None

    # Normalize common aliases / display names → registry id.
    norm = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "draftcursorprompt": "draft_cursor_prompt",
        "draft_cursor": "draft_cursor_prompt",
        "draft_cursor_prompt_tool": "draft_cursor_prompt",
        "log_ticket": "draft_cursor_prompt",
        "patch_ledger": "draft_cursor_prompt",
    }
    name = aliases.get(norm, name if name == "draft_cursor_prompt" else (norm or name))
    if norm == "draft_cursor_prompt":
        name = "draft_cursor_prompt"

    # Ensure draft_cursor args use objective/context keys when nested oddly.
    if name == "draft_cursor_prompt":
        if "objective" not in args and "context" not in args:
            # Sometimes models nest under "properties" or put text under "input".
            if isinstance(data.get("context"), str) or isinstance(data.get("objective"), str):
                args = {
                    "objective": str(data.get("objective") or args.get("text") or ""),
                    "context": str(data.get("context") or args.get("text") or ""),
                }
        if not str(args.get("objective") or "").strip() and not str(
            args.get("context") or ""
        ).strip():
            return None

    # Reject unknown tool ids from prose JSON (reduces false-positive dispatches).
    if name not in _CONTENT_TOOL_ALLOWLIST:
        return None

    return {
        "name": name,
        "args": {str(k): v for k, v in dict(args).items()},
        "id": f"content-json-{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }


def extract_final(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Prefer an explicit FINAL segment even when the model also emitted a tool JSON.
    m_final = re.search(
        r"(?:FINAL| )\s*[:：]\s*(.+)$",
        raw,
        re.DOTALL | re.I,
    )
    if m_final:
        return m_final.group(1).strip()
    m = _FINAL_RE.match(raw)
    if m:
        return m.group(1).strip()
    if _extract_json_object(raw) is not None:
        return None
    if _TOOL_LINE_RE.match(raw) or re.match(r"^\s*(?:TOOL|Action)\s*[:：]", raw, re.I):
        return None
    return raw


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Detect a tool-shaped JSON blob so it is never spoken as a FINAL answer."""
    raw = (text or "").strip()
    if not raw:
        return None
    candidates: list[str] = []
    fence = _MD_FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        keys = {str(k).lower() for k in data.keys()}
        if keys & _TOOLISH_JSON_KEYS:
            return data
        if "tool" in data or "tool_id" in data:
            return data
    return None


def _obs_fallback(observation: str, reply_lang: str) -> str:
    """User-facing fallback when ReAct hits the iter cap — never dump raw tool IR."""
    obs = (observation or "").strip()
    # Memory miss is recoverable — speak a natural line instead of the crash fallback.
    if _is_memory_key_miss(obs):
        return (
            "‌      ‌ ."
            if reply_lang == "fa"
            else "I don't know what you're referring to — that isn't in my memory."
        )
    # draft_cursor_prompt / ledger receipts → short UX ack only (never read ticket body).
    if _DRAFT_CURSOR_OBS_OK_RE.search(obs) or looks_like_ticket_receipt_speech(obs):
        return DRAFT_CURSOR_UX_ACK
    # Prefer a short natural line; hide SpatialIR / OK: tool dumps from TTS.
    looks_raw = (
        obs.upper().startswith("OK:")
        or obs.upper().startswith("ERROR:")
        or obs.startswith("SpatialIR=")
        or "TOOL" in obs.upper()[:20]
        or looks_like_raw_json_speech(obs)
        or looks_like_ticket_receipt_speech(obs)
    )
    # Successful Tool Forge — speak a clear confirmation instead of the crash fallback.
    if obs.upper().startswith("OK:") and (
        "tool forge" in obs.lower() or "hot-loaded" in obs.lower() or "forged" in obs.lower()
    ):
        loaded = re.findall(r"`([^`]+)`", obs)
        if not loaded:
            loaded = re.findall(r"loaded=\[([^\]]+)\]", obs)
            if loaded:
                loaded = [
                    x.strip().strip("'\"")
                    for x in loaded[0].split(",")
                    if x.strip()
                ]
        if reply_lang == "fa":
            if len(loaded) > 1:
                return f"{len(loaded)}   : {', '.join(loaded[:5])}."
            if loaded:
                return f" `{loaded[0]}`    ."
            return "     ."
        if len(loaded) > 1:
            return f"Done — I forged and hot-loaded {len(loaded)} tools: {', '.join(loaded[:5])}."
        if loaded:
            return f"Done — I forged and hot-loaded `{loaded[0]}`."
        return "Done — Tool Forge forged and hot-loaded the new tool."
    if obs.upper().startswith("OK:") and "evaluate_slide_and_type" in obs.lower():
        cm = re.search(r"COMMENT=([^\n]+)", obs)
        vd = re.search(r"verdict=(\w+)", obs, re.I)
        comment = (cm.group(1).strip() if cm else "").strip()
        verdict = (vd.group(1).upper() if vd else "DONE")
        if reply_lang == "fa":
            return f"   ({verdict}).   ."
        if comment:
            return f"Slide review {verdict}. I typed: {comment[:160]}"
        return f"Slide review complete ({verdict}); comment typed into the active window."
    if "evaluate_slide_and_type" in obs.lower() and obs.upper().startswith("ERROR:"):
        if reply_lang == "fa":
            return "       ."
        snip = obs.split("COMMENT_READY=", 1)[-1].split("\n", 1)[0].strip() if "COMMENT_READY=" in obs else ""
        if snip:
            return f"Slide review finished but typing failed. Comment was: {snip[:160]}"
        return "Slide review finished but typing into the active window failed."
    # Pull paraphrasable payload out of common OK: tool dumps (e.g. naming fix).
    if looks_raw:
        m = re.search(r"\btext=('([^']*)'|\"([^\"]*)\")", obs)
        if m:
            extracted = (m.group(2) or m.group(3) or "").strip()
            if extracted:
                return extracted
    if reply_lang == "fa":
        if looks_raw or not obs:
            return "    ‌  ."
        return obs
    if looks_raw or not obs:
        return "I couldn't finish that cleanly — please ask me again."
    return obs


_MEMORY_MISS_RE = re.compile(
    r"(?i)(?:memory key not found|key ['\"][^'\"]+['\"] not found|vault key not found)",
)


def _is_memory_key_miss(observation: str) -> bool:
    return bool(_MEMORY_MISS_RE.search(observation or ""))


def _is_draft_ticket_validation_bounce(observation: str) -> bool:
    """True for draft_cursor_prompt Pydantic / intent-echo rejects (not crashes)."""
    text = (observation or "").strip().lower()
    if not text:
        return False
    markers = (
        "validation error",
        "intent-echo",
        "ticket validation failed",
        "max retries reached: ticket validation",
        "context is an intent-echo",
        "require root cause, step-by-step",
        "draft rejected. context must include",
    )
    return any(m in text for m in markers)


_FALLBACK_EN = "I couldn't finish that cleanly — please ask me again."
_FALLBACK_FA = "    ‌  ."


def _maybe_record_bug_tracker(
    *,
    user_text: str,
    spoken: str,
    last_obs: str,
    tool_trace: list[dict[str, Any]],
    had_errors: bool,
) -> None:
    """Append to docs/bug_tracker.json when the loop collapses to the fallback phrase."""
    spoken_l = (spoken or "").strip()
    is_fallback = spoken_l in {_FALLBACK_EN, _FALLBACK_FA} or (
        "couldn't finish that cleanly" in spoken_l.lower()
        or "couldn't complete that" in spoken_l.lower()
    )
    if not is_fallback and not had_errors:
        return
    # Prefer ERROR observations from the trace as the exception payload.
    error = ""
    tb = ""
    for row in reversed(tool_trace or []):
        if row.get("error"):
            error = str(row.get("error"))
            break
        obs = str(row.get("observation") or "")
        if obs.upper().startswith("ERROR"):
            error = obs
            break
    if not error:
        error = (last_obs or spoken_l or "agentic_fallback").strip()[:2000]
    # Soft memory misses are not terminal failures — don't pollute the bug tracker.
    if _is_memory_key_miss(error) or _is_memory_key_miss(last_obs):
        return
    # Ticket validation / intent-echo bounces belong in the MoA rewrite loop,
    # not tracker/bug_tracker.json (that misroutes "PENDING" away from patch_ledger).
    if _is_draft_ticket_validation_bounce(error) or _is_draft_ticket_validation_bounce(
        last_obs
    ):
        return
    try:
        from dana.bug_tracker import log_bug_to_tracker

        log_bug_to_tracker(
            error,
            context=(
                f"user_query={(user_text or '').strip()[:1500]}\n"
                f"spoken={spoken_l[:400]}\n"
                f"tool_trace={tool_trace!r}"[:2500]
            ),
            status="PENDING",
            source="agentic_fallback",
            user_query=(user_text or "").strip()[:2000],
        )
    except Exception:  # noqa: BLE001
        try:
            from dana.logging import log_exception

            log_exception("BugTracker", "failed to append bug_tracker.json")
        except Exception:
            pass


def strip_simulated_dialog_leaks(text: str) -> str:
    """Remove few-shot roleplay / template echoes before TTS or conversation logs.

    llama3.2 often mirrors prompt examples like ``User: "What time is it?"`` or
    ``→ speak: ...``. Keep only Donna's direct narrative payload.
    """
    raw = (text or "").strip()
    if not raw:
        return raw

    kept: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if _DIALOG_ROLE_LINE_RE.match(s):
            continue
        if _ARROW_ROUTE_LINE_RE.match(s):
            continue
        speak_m = _ARROW_SPEAK_LINE_RE.match(s)
        if speak_m:
            payload = (speak_m.group(1) or "").strip()
            if payload:
                kept.append(payload)
            continue
        label_m = _ANSWER_LABEL_LINE_RE.match(s)
        if label_m:
            payload = (label_m.group(1) or "").strip()
            if payload:
                kept.append(payload)
            continue
        kept.append(s)

    out = " ".join(kept).strip()
    out = _INLINE_USER_ECHO_RE.sub("", out)
    # Residual arrow speak fragments mid-string.
    out = re.sub(
        r"(?:→|->)\s*(?:then\s+)?speak\s*[:：]\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s{2,}", " ", out).strip(" \t-–—|")
    return out


def looks_like_confidential_fixture_leak(text: str) -> bool:
    """True when speech looks like the Project Omega sandbox fixture dump."""
    return bool(_CONFIDENTIAL_FIXTURE_LEAK_RE.search(text or ""))


def sanitize_spoken_reply(
    text: str,
    *,
    reply_lang: str,
    last_obs: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
    draft_cursor_ok: bool | None = None,
) -> str:
    """Final TTS gate: never speak raw tool IR, JSON tool-calls, or few-shot dialog.

    After a successful ``draft_cursor_prompt``, any LLM final string is discarded
    and replaced with ``DRAFT_CURSOR_UX_ACK`` (WAV-cache hit).
    """
    # Tool-specific override — intercept LLM chatter before any other sanitization.
    ok = (
        bool(draft_cursor_ok)
        if draft_cursor_ok is not None
        else draft_cursor_tool_succeeded(last_obs=last_obs, tool_trace=tool_trace)
    )
    if ok:
        return DRAFT_CURSOR_UX_ACK

    spoken = strip_simulated_dialog_leaks(text or "")
    # Strip markdown fences early so Piper never reads generated Python aloud.
    try:
        from dana.core_agent import sanitize_text_for_tts

        spoken = sanitize_text_for_tts(spoken)
    except Exception:  # noqa: BLE001
        spoken = re.sub(
            r"```[\w+-]*\n?[\s\S]*?```",
            "[Code block generated]",
            spoken or "",
        )
    spoken = strip_raw_json_from_speech(spoken)
    # Successful ledger write (obs-only path): always speak the short INTERACTION_UX ack.
    if _DRAFT_CURSOR_OBS_OK_RE.search(last_obs or "") or (
        looks_like_ticket_receipt_speech(spoken)
        and _DRAFT_CURSOR_OBS_OK_RE.search(last_obs or spoken or "")
    ):
        return DRAFT_CURSOR_UX_ACK
    if looks_like_ticket_receipt_speech(spoken):
        spoken = ""
    if looks_like_raw_json_speech(spoken) or looks_like_raw_tool_observation(spoken):
        spoken = ""
    if looks_like_confidential_fixture_leak(spoken) or looks_like_confidential_fixture_leak(
        last_obs
    ):
        # Never TTS sandbox fixtures / vault-adjacent test docs as "answers".
        return (
            "  ‌."
            if reply_lang == "fa"
            else "Sorry — that looked like an internal document dump. Please ask again."
        )
    if not spoken:
        safe_obs = last_obs
        if looks_like_confidential_fixture_leak(safe_obs or ""):
            safe_obs = ""
        return _obs_fallback(safe_obs, reply_lang) if safe_obs else (
            "  ‌."
            if reply_lang == "fa"
            else "Sorry — please ask me again."
        )
    if looks_like_raw_tool_observation(spoken):
        if looks_like_confidential_fixture_leak(last_obs or spoken):
            return (
                "Sorry — that looked like an internal document dump. Please ask again."
                if reply_lang != "fa"
                else "  ‌."
            )
        return _obs_fallback(last_obs or spoken, reply_lang)
    if _SYSTEM_META_HALLUCINATION_RE.search(spoken):
        return (
            "  ‌."
            if reply_lang == "fa"
            else "Sorry — I couldn't complete that request cleanly. Please try again."
        )
    # If sanitization left only a greeting shell around leaked dialog, reject.
    if _GENERIC_GREETING_RE.match(spoken) and last_obs:
        if looks_like_confidential_fixture_leak(last_obs):
            return (
                "Sorry — that looked like an internal document dump. Please ask again."
                if reply_lang != "fa"
                else "  ‌."
            )
        return _obs_fallback(last_obs, reply_lang)
    return spoken


def _spoken_fact_from_search_obs(observation: str, user_text: str = "") -> str | None:
    """Best-effort spoken line from web_search text when the LLM never FINALs.

    Prefers upcoming (on/after today) date+clock pairs so we don't speak a past
    fixture scraped from a random snippet.
    """
    from datetime import date as _date

    obs = observation or ""
    if not obs or obs.upper().startswith("ERROR"):
        return None

    today = _date.today()
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    def _parse_date(near: str) -> _date | None:
        m = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b",
            near,
            re.I,
        )
        if m:
            month = month_map[m.group(1).lower()]
            day = int(m.group(2))
            year = int(m.group(3) or today.year)
            try:
                return _date(year, month, day)
            except ValueError:
                return None
        m = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|"
            r"July|August|September|October|November|December)\b",
            near,
            re.I,
        )
        if m:
            day = int(m.group(1))
            month = month_map[m.group(2).lower()]
            try:
                return _date(today.year, month, day)
            except ValueError:
                return None
        return None

    candidates: list[tuple[int, str]] = []
    for clock_m in re.finditer(
        r"\b(?:\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm)|"
        r"(?:[01]?\d|2[0-3]):[0-5]\d)\b",
        obs,
    ):
        clock = clock_m.group(0).strip()
        window = obs[max(0, clock_m.start() - 80) : clock_m.end() + 40]
        d = _parse_date(window)
        if d is None and re.search(r"\btoday\b", window, re.I):
            d = today
        if d is None and re.search(r"\btomorrow\b", window, re.I):
            d = today.fromordinal(today.toordinal() + 1)
        after = obs[clock_m.end() : clock_m.end() + 12].split()
        tz = ""
        if after:
            tok = re.sub(r"[^A-Za-z]", "", after[0])
            if _TZ_TOKEN_RE.match(tok):
                tz = f" {tok.upper()}"
        spoken_clock = clock
        # Convert labeled ET/PT/… kickoffs into the user's local timezone.
        if tz:
            try:
                from dana.settings import format_kickoff_in_local_tz

                src_map = {
                    "ET": "America/New_York",
                    "EST": "America/New_York",
                    "EDT": "America/New_York",
                    "PT": "America/Los_Angeles",
                    "PST": "America/Los_Angeles",
                    "PDT": "America/Los_Angeles",
                    "CT": "America/Chicago",
                    "CST": "America/Chicago",
                    "CDT": "America/Chicago",
                    "MT": "America/Denver",
                    "MST": "America/Denver",
                    "MDT": "America/Denver",
                    "UTC": "UTC",
                    "GMT": "UTC",
                }
                src_tz = src_map.get(tz.strip().upper())
                cm = re.match(
                    r"^\s*(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\s*$",
                    clock,
                )
                if src_tz and cm:
                    hour = int(cm.group(1))
                    minute = int(cm.group(2) or 0)
                    ampm = (cm.group(3) or "").upper()
                    if ampm == "PM" and hour < 12:
                        hour += 12
                    elif ampm == "AM" and hour == 12:
                        hour = 0
                    local_s = format_kickoff_in_local_tz(
                        hour, minute, source_tz=src_tz
                    )
                    if local_s:
                        spoken_clock = local_s
                        tz = ""
            except Exception:
                pass
        if d is not None:
            label = d.strftime("%B %d")
            spoken = f"{label} at {spoken_clock}{tz}."
            # Prefer soonest upcoming (on/after today); demote past fixtures.
            if d >= today:
                days_out = (d - today).days
                score = 200 - days_out  # sooner is better
            else:
                score = -50
            if re.search(r"\b(next|upcoming|today|tonight|tomorrow)\b", window, re.I):
                score += 15
            if re.search(r"\b(opening|final|championship)\b", window, re.I) and d < today:
                score -= 30
            # Prefer ET/PT labeled kickoffs over bare 24h clocks.
            if tz or re.search(r"\b(ET|PT|pm|am)\b", clock, re.I) or spoken_clock != clock:
                score += 5
        else:
            spoken = f"Kickoff is {spoken_clock}{tz}."
            score = 10
        candidates.append((score, spoken))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]
    if best_score < 0 and _wants_event_clock(user_text):
        # All dated hits are in the past — don't invent; let LLM/retry handle.
        return None
    return best


_FACT_ASK_RE = re.compile(
    r"\b(when|what\s+hour|what\s+time|hour|time|kickoff|o'?clock|next|match|"
    r"date|where|who|how\s+many|schedule)\b",
    re.IGNORECASE,
)
_CLIPBOARD_ASK_RE = re.compile(
    r"("
    r"\bclipboard\b|\bread\s+my\s+(?:clipboard|screen)\b|"
    r"\bcheck\s+what\s+i\s+copied\b|\bwhat\s+(?:did\s+i|i\s+just)\s+copied\b|"
    r"\bsummarize\s+what\s+i\s+(?:just\s+)?copied\b|\bwhat'?s\s+on\s+my\s+clipboard\b|"
    r"\s*|[\s\-‌]*"
    r")",
    re.IGNORECASE,
)
_HOUR_ASK_RE = re.compile(
    r"\b(what\s+hour|what\s+time|hour|o'?clock|kickoff\s+time)\b",
    re.IGNORECASE,
)
_EVENT_WHEN_RE = re.compile(
    r"\b(when|next|hour|time|kickoff|o'?clock|schedule)\b",
    re.IGNORECASE,
)
_EVENT_NOUN_RE = re.compile(
    r"\b(match|game|fifa|world\s*cup|fixture|tournament)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b|\b\d{1,2}\s*(?:AM|PM|am|pm)\b",
    re.IGNORECASE,
)
_TIME_FACT_RE = re.compile(
    r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b|\b\d{1,2}\s*(?:AM|PM|am|pm)\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?\b",
    re.IGNORECASE,
)
_PURE_SITE_TOUR_RE = re.compile(
    r"^(?:you can find|you can also|check (?:the )?(?:schedule|it) on|"
    r"various websites)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|org|net|io)\b",
    re.IGNORECASE,
)
_TZ_TOKEN_RE = re.compile(r"^(?:ET|PT|CT|MT|UTC|GMT|EST|PST|CST|MST)$", re.IGNORECASE)


def _wants_event_clock(user_text: str) -> bool:
    text = user_text or ""
    if _HOUR_ASK_RE.search(text):
        return True
    # "Next FIFA match" / "when is the match" → want time of day, not date alone.
    if _EVENT_WHEN_RE.search(text) and _EVENT_NOUN_RE.search(text):
        return True
    # Vague STT like "FIFA match." still implies schedule/kickoff lookup.
    if re.search(r"\b(fifa|world\s*cup)\b", text, re.I) and re.search(
        r"\b(match|game|fixture|kickoff)\b", text, re.I
    ):
        return True
    return False


def wrap_user_query_for_react(
    user_text: str,
    reply_lang: str,
    *,
    context_suffix: str = "",
) -> str:
    """Anchor 8B attention on the user question; English lock when reply_lang=en.

    ``context_suffix`` (Visual Context / short-term memory tags) is appended
    *after* the utterance so recency wins over the long system prompt.
    """
    text = (user_text or "").strip()
    anchored = (
        f"USER'S ACTUAL QUESTION: {text}\n"
        "PRIORITY: Answer this question directly using your available tools or memory."
    )
    if re.search(r"\b(research|latest\s+updates?|updates?\s+on)\b", text, re.I):
        anchored += (
            "\nREQUIRED THIS TURN: Call the web_search tool on step 1 "
            "(native tool call). Do NOT call read_vault_memory or write_vault_memory "
            "for live research."
        )
    if reply_lang == "en":
        base = f"[User Query - Respond in English]:\n{anchored}"
    else:
        base = anchored
    suffix = (context_suffix or "").strip()
    if suffix:
        return f"{base}\n\n{suffix}"
    return base


def strip_protocol_speech_anchors(text: str) -> str:
    """Remove structural FINAL markers the model may echo into spoken text."""
    out = (text or "").strip()
    out = re.sub(r"^\s*\[STRICTLY ENGLISH TEXT\]\s*", "", out, flags=re.I)
    out = re.sub(r"\s*\[STRICTLY ENGLISH TEXT\]\s*", " ", out, flags=re.I)
    # Strip leaked XML context wrappers (system-prompt bleed into TTS).
    out = re.sub(
        r"</?(?:visual_context|memory)\b[^>]*>",
        " ",
        out,
        flags=re.I,
    )
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def clip_spoken_answer(user_text: str, answer: str, *, max_words: int = 22) -> str:
    """Keep FINAL short enough for TTS — especially factual / time answers."""
    text = strip_protocol_speech_anchors(answer or "")
    if not text:
        return text
    # Strip leaked ReAct / tool scaffolding before TTS (leading or trailing).
    text = re.sub(
        r"^(?:Observation|TOOL|Action|FINAL|OK|ERROR|LOCKED)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    # If the whole reply is still raw tool IR, leave a marker for sanitize_spoken_reply.
    if looks_like_raw_tool_observation(answer or ""):
        return (answer or "").strip()
    text = _TTS_TOOL_LEAK_RE.sub("", text).strip()
    text = strip_protocol_speech_anchors(text)
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text).strip(" ,;")

    if not _FACT_ASK_RE.search(user_text or ""):
        words = text.split()
        if len(words) <= 40:
            return text
        first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        return first or text

    want_hour = _wants_event_clock(user_text)
    primary = _CLOCK_RE if want_hour else _TIME_FACT_RE
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chosen = ""
    for pattern in ((primary, _TIME_FACT_RE) if want_hour else (_TIME_FACT_RE,)):
        for s in sentences:
            if not pattern.search(s):
                continue
            if _PURE_SITE_TOUR_RE.match(s):
                continue
            chosen = re.sub(
                r"^(?:For example,?|Also,|Additionally,)\s*",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip()
            break
        if chosen:
            break
    if not chosen:
        for s in sentences:
            if not _PURE_SITE_TOUR_RE.match(s):
                chosen = s
                break
    if not chosen:
        chosen = sentences[0] if sentences else text

    if want_hour:
        m = _CLOCK_RE.search(chosen)
        if m:
            after = chosen[m.end() :].split()
            date_m = re.search(
                r"\b(?:January|February|March|April|May|June|July|August|September|"
                r"October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?\b",
                chosen,
                re.I,
            )
            clock = m.group(0).strip()
            tz = ""
            if after:
                tok = re.sub(r"[^A-Za-z]", "", after[0])
                if _TZ_TOKEN_RE.match(tok):
                    tz = f" {tok.upper()}"
            if date_m and not _HOUR_ASK_RE.search(user_text or ""):
                return f"{date_m.group(0)} at {clock}{tz}.".replace("..", ".")
            return f"Kickoff is {clock}{tz}.".replace("..", ".")

    words = chosen.split()
    if len(words) > max_words:
        m = primary.search(chosen) or _TIME_FACT_RE.search(chosen)
        if m:
            before = chosen[: m.start()].split()
            after = chosen[m.end() :].split()
            nugget = " ".join(before[-5:] + [m.group(0)] + after[:3]).strip(" ,;")
            chosen = nugget.rstrip(",;:") + ("." if not nugget.endswith(".") else "")
            words = chosen.split()
        if len(words) > max_words:
            chosen = " ".join(words[:max_words]).rstrip(",;:") + "."
    # Final TTS safety net: never speak raw TOOL:/FINAL: IR.
    chosen = _TTS_TOOL_LEAK_RE.sub("", chosen).strip()
    return chosen.strip()


def _maybe_reflect(
    *,
    user_text: str,
    tool_trace: list[dict[str, Any]],
    reflect_fn: Callable[[list[dict[str, str]]], str] | None,
    vault_client: Any | None,
    enable_reflection: bool,
) -> tuple[dict[str, Any] | None, float, bool]:
    if not enable_reflection:
        return None, 0.0, False
    from dana.reflector import persist_lesson, run_reflection, trace_has_failure

    if not trace_has_failure(tool_trace):
        return None, 0.0, False
    result = run_reflection(
        user_text=user_text,
        tool_trace=tool_trace,
        reflect_fn=reflect_fn,
    )
    persisted = False
    if result.lesson is not None and vault_client is not None:
        try:
            persisted = bool(persist_lesson(vault_client, result.lesson))
        except Exception:  # noqa: BLE001
            persisted = False
    payload = {
        "triggered": result.triggered,
        "rule": result.lesson.rule if result.lesson else None,
        "tool_id": result.lesson.tool_id if result.lesson else None,
        "persisted": persisted,
        "error": result.error or None,
    }
    return payload, result.latency_ms, True


def _chat_situational_context() -> str:
    """Current time + user name + hardware load for lightweight chat."""
    from datetime import datetime

    import psutil

    now = datetime.now().astimezone()
    time_line = now.strftime("%A, %B %d, %Y %I:%M %p %Z").strip()
    user_name = ""
    try:
        from dana.core_agent import VAULT_HOT_CACHE

        user_name = str((VAULT_HOT_CACHE or {}).get("user_name") or "").strip()
    except Exception:  # noqa: BLE001
        user_name = ""
    try:
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().percent)
        load_line = f"System load: CPU {cpu:.0f}%, RAM {ram:.0f}%."
    except Exception:  # noqa: BLE001
        load_line = ""
    bits = [f"Current local time: {time_line}."]
    if user_name:
        bits.append(f"The user's name is {user_name}.")
    if load_line:
        bits.append(load_line)
    return " ".join(bits)


def build_lightweight_chat_system_prompt(
    *,
    reply_lang: str = "en",
    visual_context: str | None = None,
) -> str:
    """Persona + capability card for chat mode (no tool / TPM rules)."""
    parts = [_LIGHTWEIGHT_CHAT_SYSTEM, _CHAT_CAPABILITY_CARD]
    parts.append(_chat_situational_context())
    try:
        parts.append(f"Active Donna mode: {get_donna_mode()}.")
    except Exception:  # noqa: BLE001
        pass
    if reply_lang == "fa":
        parts.append("Reply in language (language) unless the user writes in English.")
    else:
        parts.append("Reply in English unless the user clearly writes in another language.")
    vision = (visual_context or "").strip()
    if vision:
        parts.append(f"Optional scene context (do not invent tools): {vision}")
    prompt = "\n".join(parts)
    try:
        from dana.memory.blackboard import append_persona_mixer_override

        return append_persona_mixer_override(prompt)
    except Exception:  # noqa: BLE001
        return prompt


def run_lightweight_chat(
    *,
    user_text: str,
    system_prompt: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    model: str = "llama3.2",
    ask_fn: Callable[..., str] | None = None,
    visual_context: str | None = None,
    use_chat_memory: bool = True,
    session_id: str = "",
) -> AgenticResult:
    """Bypasses ReAct/MoA and injects rolling memory for natural conversation.

    History is passed as role-separated ``user`` / ``assistant`` messages.
    ReAct ``prior_messages`` are never merged into chat (isolated durable
    session state is ``chat_memory_buffer`` only).
    """
    _ = prior_messages  # ReAct history must not leak into lightweight chat
    from dana.settings import resolve_reply_lang

    reply_lang = resolve_reply_lang(user_text or "")
    user_clean = (user_text or "").strip()
    base = (system_prompt or "").strip() or build_lightweight_chat_system_prompt(
        reply_lang=reply_lang,
        visual_context=visual_context,
    )
    # Stage 4.3 — conversational piggyback of unread actuator completions.
    try:
        from dana.memory.blackboard import (
            format_background_system_alert,
            get_and_clear_unread_notifications,
        )

        unread = get_and_clear_unread_notifications(session_id or None)
        # Stage 6.5 — Chat-side Andon safety net for failed operators.
        try:
            from dana.management.jason_supervisor import (
                OPERATOR_ANDON_TOOLS,
                trigger_andon_cord,
            )

            for row in unread:
                if str(row.get("status") or "").lower() != "failed":
                    continue
                tool = str(row.get("tool_name") or "").strip()
                if tool not in OPERATOR_ANDON_TOOLS:
                    continue
                err = str(row.get("error_context") or row.get("result") or "").strip()
                trigger_andon_cord(
                    task_name=tool,
                    error_context=err or "operator failed",
                    failed_action_id=int(row.get("action_id") or 0),
                    failed_arguments=dict(row.get("arguments") or {}),
                    session_id=session_id or "",
                    run_recovery=True,
                )
        except Exception:  # noqa: BLE001
            pass
        alert = format_background_system_alert(unread)
        if alert:
            base = f"{base}\n\n{alert}"
            try:
                from dana.telemetry import log_notification_piggyback

                log_notification_piggyback(
                    alert,
                    session_id=session_id or "",
                    count=len(unread),
                    payload={
                        "action_ids": [int(r.get("action_id") or 0) for r in unread]
                    },
                )
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # Stage 6.4 — live Persona Mixer splice (always refresh from Blackboard).
    try:
        from dana.memory.blackboard import append_persona_mixer_override

        base = append_persona_mixer_override(base)
    except Exception:  # noqa: BLE001
        pass
    messages: list[dict[str, str]] = [{"role": "system", "content": base}]
    if use_chat_memory:
        messages.extend(chat_memory_as_role_messages())
    messages.append({"role": "user", "content": user_clean})

    if ask_fn is None:
        raise RuntimeError("run_lightweight_chat requires ask_fn (Ollama chat callable)")

    try:
        try:
            raw = ask_fn(messages, model=model)
        except TypeError:
            raw = ask_fn(messages)
    except Exception as exc:  # noqa: BLE001
        if is_ollama_connection_error(exc):
            spoken = OLLAMA_UNREACHABLE_SPEECH
            return AgenticResult(
                final_text=spoken,
                iterations=0,
                tool_trace=[{"error": f"ollama_unreachable:{exc}"}],
                reply_lang=reply_lang,
                reflection=None,
                reflection_ms=0.0,
                had_errors=True,
                tts_streamed=False,
            )
        raise

    spoken = strip_r1_think_blocks(str(raw or ""))
    spoken = clip_spoken_answer(user_text, spoken, max_words=40)
    spoken = sanitize_spoken_reply(
        spoken,
        reply_lang=reply_lang,
        last_obs="",
        tool_trace=None,
    )
    if not (spoken or "").strip():
        spoken = (
            "  ‌."
            if reply_lang == "fa"
            else "Sorry — please ask me again."
        )
    # Filler / verbal escalate must not pollute chat memory (caller escalates).
    reject_final = False
    try:
        from dana.graph.completion_gate import should_reject_chat_final

        reject_final = should_reject_chat_final(spoken, user_clean)
    except Exception:  # noqa: BLE001
        reject_final = False
    if use_chat_memory and not reject_final:
        append_chat_memory_turn(user_clean, spoken)
    return AgenticResult(
        final_text=spoken,
        iterations=1,
        tool_trace=[],
        reply_lang=reply_lang,
        reflection=None,
        reflection_ms=0.0,
        had_errors=False,
        tts_streamed=False,
    )


def run_react_loop(
    *,
    user_text: str,
    system_prompt: str,
    execute_fn: Callable[[ToolCall], str],
    max_iters: int = REACT_MAX_ITERS,
    broker: IntentBroker | None = None,
    reflect_fn: Callable[[list[dict[str, str]]], str] | None = None,
    vault_client: Any | None = None,
    enable_reflection: bool = True,
    prior_messages: list[dict[str, str]] | None = None,
    on_tool_start: Callable[[ToolCall, str], None] | None = None,
    visual_context: str | None = None,
    model: str = "llama3.2",
    forced_tool: ToolCall | None = None,
    tts_callback: Callable[[str], None] | None = None,
) -> AgenticResult:
    """Native tool loop: ChatOllama.bind_tools → ToolMessage → spoken answer.

    Runs the async LangGraph ReAct path (MemorySaver + astream_events TTS).
    """
    try:
        return asyncio.run(
            _run_react_loop_langchain(
                user_text=user_text,
                system_prompt=system_prompt,
                execute_fn=execute_fn,
                max_iters=max_iters,
                broker=broker,
                reflect_fn=reflect_fn,
                vault_client=vault_client,
                enable_reflection=enable_reflection,
                prior_messages=prior_messages,
                on_tool_start=on_tool_start,
                visual_context=visual_context,
                model=model,
                forced_tool=forced_tool,
                tts_callback=tts_callback,
            )
        )
    except RuntimeError:
        # Already inside an event loop (e.g. nested async host).
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _run_react_loop_langchain(
                    user_text=user_text,
                    system_prompt=system_prompt,
                    execute_fn=execute_fn,
                    max_iters=max_iters,
                    broker=broker,
                    reflect_fn=reflect_fn,
                    vault_client=vault_client,
                    enable_reflection=enable_reflection,
                    prior_messages=prior_messages,
                    on_tool_start=on_tool_start,
                    visual_context=visual_context,
                    model=model,
                    forced_tool=forced_tool,
                    tts_callback=tts_callback,
                )
            )
        finally:
            loop.close()


def _build_seed_messages(
    *,
    user_text: str,
    system_prompt: str,
    prior_messages: list[dict[str, str]] | None,
    visual_context: str | None,
    reply_lang: str,
) -> list[dict[str, str]]:
    from dana.prompts.spatial_synthesis import format_recency_context_block

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    prior_count = 0
    if prior_messages:
        for m in prior_messages:
            role = (m or {}).get("role")
            content = str((m or {}).get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
                prior_count += 1
    context_suffix = format_recency_context_block(
        vision_line=visual_context or "",
        prior_turn_count=prior_count,
    )
    messages.append(
        {
            "role": "user",
            "content": wrap_user_query_for_react(
                user_text,
                reply_lang,
                context_suffix=context_suffix,
            ),
        }
    )
    return messages


def _dicts_to_lc_messages(messages: list[dict[str, str]]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out: list[Any] = []
    for m in messages:
        role = (m or {}).get("role")
        content = sanitize_react_observation(str((m or {}).get("content") or ""))
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


_OBS_MAX_CHARS = 2000
_DATA_URI_RE = re.compile(
    r"data:image/[a-zA-Z0-9+.-]+;base64,[A-Za-z0-9+/=\s]{80,}",
    re.IGNORECASE,
)
_LONG_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{400,}={0,2}\b")
_IMAGES_JSON_RE = re.compile(
    r'"images"\s*:\s*\[[^\]]{80,}\]',
    re.IGNORECASE | re.DOTALL,
)


def sanitize_react_observation(
    text: str,
    *,
    max_chars: int = _OBS_MAX_CHARS,
) -> str:
    """Strip raw image/base64 payloads and truncate tool observations for ReAct.

    Downstream reasoners must never see base64 image bytes — only extracted text.
    """
    s = text or ""
    if not s:
        return ""
    s = _DATA_URI_RE.sub("[IMAGE_STRIPPED]", s)
    s = _IMAGES_JSON_RE.sub('"images":["[IMAGE_STRIPPED]"]', s)
    s = _LONG_B64_RE.sub("[BASE64_STRIPPED]", s)
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "...[TRUNCATED for context window]"
    return s


def sanitize_react_message_history(messages: list[Any]) -> list[Any]:
    """In-place reducer: drop multimodal image blocks + sanitize string content."""
    for msg in messages or []:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            try:
                msg.content = sanitize_react_observation(content)
            except Exception:
                pass
            continue
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type") or "").lower()
                if btype in ("text", "input_text"):
                    texts.append(str(block.get("text") or block.get("content") or ""))
                elif "image" in btype or block.get("image_url") or block.get("images"):
                    texts.append("[IMAGE_STRIPPED]")
                else:
                    # Unknown structured block — stringify without raw blobs.
                    texts.append(sanitize_react_observation(str(block), max_chars=400))
            try:
                msg.content = sanitize_react_observation("\n".join(texts))
            except Exception:
                pass
    return messages


def _tool_call_from_lc(tc: Any, *, raw_text: str = "") -> ToolCall:
    """Normalize a LangChain tool_call dict / object into Donna ToolCall IR."""
    if isinstance(tc, dict):
        name = str(tc.get("name") or tc.get("tool") or tc.get("tool_id") or "")
        args = (
            tc.get("args")
            or tc.get("arguments")
            or tc.get("parameters")
            or tc.get("input")
            or {}
        )
        call_id = str(tc.get("id") or "")
    else:
        name = str(getattr(tc, "name", "") or "")
        args = (
            getattr(tc, "args", None)
            or getattr(tc, "arguments", None)
            or getattr(tc, "parameters", None)
            or {}
        )
        call_id = str(getattr(tc, "id", "") or "")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"text": args}
    if not isinstance(args, dict):
        try:
            args = dict(args)
        except Exception:
            args = {}
    return ToolCall(
        tool_id=name,
        arguments={str(k): v for k, v in args.items()},
        source_lang="en",
        raw_text=raw_text or "",
        confidence=1.0,
    )


async def _run_react_loop_langchain(
    *,
    user_text: str,
    system_prompt: str,
    execute_fn: Callable[[ToolCall], str],
    max_iters: int,
    broker: IntentBroker | None,
    reflect_fn: Callable[[list[dict[str, str]]], str] | None,
    vault_client: Any | None,
    enable_reflection: bool,
    prior_messages: list[dict[str, str]] | None,
    on_tool_start: Callable[[ToolCall, str], None] | None,
    visual_context: str | None,
    model: str,
    forced_tool: ToolCall | None = None,
    tts_callback: Callable[[str], None] | None = None,
) -> AgenticResult:
    """Async LangGraph ReAct loop (MemorySaver + astream_events TTS)."""
    from dana.agentic_react_graph import run_react_langgraph
    from dana.settings import resolve_reply_lang

    # Pre-flight: abort before the graph if Ollama is down (clear TTS, no silent fail).
    if not ollama_service_reachable():
        return AgenticResult(
            final_text=OLLAMA_UNREACHABLE_SPEECH,
            iterations=0,
            tool_trace=[{"error": "ollama_unreachable:preflight"}],
            reply_lang=resolve_reply_lang(user_text or ""),
            reflection=None,
            reflection_ms=0.0,
            had_errors=True,
            tts_streamed=False,
        )

    try:
        return await run_react_langgraph(
            user_text=user_text,
            system_prompt=system_prompt,
            execute_fn=execute_fn,
            max_iters=max_iters,
            broker=broker,
            reflect_fn=reflect_fn,
            vault_client=vault_client,
            enable_reflection=enable_reflection,
            prior_messages=prior_messages,
            on_tool_start=on_tool_start,
            visual_context=visual_context,
            model=model,
            forced_tool=forced_tool,
            tts_callback=tts_callback,
        )
    except Exception as exc:  # noqa: BLE001
        if is_ollama_connection_error(exc):
            return AgenticResult(
                final_text=OLLAMA_UNREACHABLE_SPEECH,
                iterations=0,
                tool_trace=[{"error": f"ollama_unreachable:{exc}"}],
                reply_lang=resolve_reply_lang(user_text or ""),
                reflection=None,
                reflection_ms=0.0,
                had_errors=True,
                tts_streamed=False,
            )
        raise


def ollama_service_reachable(
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 2.0,
) -> bool:
    """Cheap health check against the local Ollama ``/api/tags`` endpoint."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200) or 200) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
        return False
    except Exception:  # noqa: BLE001
        return False
