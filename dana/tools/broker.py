"""Intent broker: bilingual utterance -> validated ToolCall IR -> execution hooks."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from dana.tools.normalize import detect_lang, normalize_text, normalize_tool_arguments
from dana.tools.schema import ToolCall, ToolSpec, load_tool_registry, tool_schema_public

# LLM-style tool call patterns (EN / FA / mixed).
_TOOL_CALL_RE = re.compile(
    r"(?:tool|call)\s*[:=]\s*([a-zA-Z_][\w]*)\s*(?:\((.*)\)|\{(.*)\})?",
    re.IGNORECASE | re.DOTALL,
)
_KV_RE = re.compile(
    r"([a-zA-Z_][\w]*)\s*[:=]\s*[\"']?([^,\"'\}\)]+)[\"']?",
)
# Event / schedule cues → prefer web_search over vision.
# Wall-clock "what time is it" is NOT a schedule cue (hot-cache System Clock).
_WALL_CLOCK_RE = re.compile(
    r"\b("
    r"what\s+time\s+is\s+it|what'?s?\s+the\s+time|"
    r"what\s+time\s+of\s+(?:the\s+)?day|time\s+of\s+(?:the\s+)?day|"
    r"current\s+time|tell\s+me\s+the\s+time"
    r")\b",
    re.IGNORECASE,
)
_SCHEDULE_HINT_RE = re.compile(
    r"\b("
    r"when|what\s+hour|next|match(?:es)?|kickoff|"
    r"schedule|fifa|world\s*cup|insotter|fiefall|tournament|fixture"
    r")\b",
    re.IGNORECASE,
)
# Live research / "latest updates" → force web_search (never vault memory).
_RESEARCH_HINT_RE = re.compile(
    r"\b("
    r"research(?:\s+the\s+latest)?|latest\s+updates?|updates?\s+on|"
    r"look\s+up|search\s+(?:the\s+)?(?:web|online)|google|find\s+online"
    r")\b",
    re.IGNORECASE,
)
# Deep / multi-agent research → dispatch_research_swarm (not a single web_search).
_DEEP_RESEARCH_RE = re.compile(
    r"\b("
    r"deep\s+research|research\s+this\s+thoroughly|"
    r"comprehensive\s+report|deep\s+dive|"
    r"write\s+a\s+(?:comprehensive\s+)?(?:research\s+)?(?:brief|report)|"
    r"write\s+(?:me\s+)?a\s+report|"
    r"investigate\s+deeply|"
    r"send\s+to\s+the\s+swarm|dispatch\s+research|"
    r"background\s+research"
    r")\b",
    re.IGNORECASE,
)
# Explicit remember/save cues must still win over research force-route.
_MEM_WRITE_HINT_RE = re.compile(
    r"\b("
    r"remember\s+(this|that|my)|save\s+(this|that)|store\s+(this|that)|"
    r"write\s+vault|my\s+name\s+is|call\s+me|i\s+live\s+in|"
    r"my\s+timezone\s+is|my\s+wife\s+is|my\s+partner\s+is|my\s+kids\s+are"
    r")\b",
    re.IGNORECASE,
)
# Tool Forge / architect — NEVER vault memory or generic chat.
# Matches singular ("build a tool") and batch ("build three tools", "tools back-to-back").
_TOOL_FORGE_HINT_RE = re.compile(
    r"\b("
    # Allow adjectives between determiner and "tool" (e.g. "build a custom tool").
    r"build\s+(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"create\s+(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"code\s+a\s+(?:script|tool)|"
    r"make\s+(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"forge\s+(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"write\s+(?:me\s+)?(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"generate\s+(?:(?:a|an|\d+)\s+)?(?:(?:custom|new|python|simple|ephemeral)\s+)*tools?|"
    r"architect\s+(?:a\s+)?(?:custom\s+)?tool|synthesize\s+(?:a\s+)?(?:custom\s+)?tool|"
    r"write\s+a\s+new\s+tool|new\s+tool\s+that|"
    r"compile\s+(?:the\s+)?logic\s+into\s+(?:the\s+)?custom\s+tools?|"
    r"tool\s+that\s+(?:can|will|does)|"
    r"tools?\s+back[- ]?to[- ]?back|"
    r"back[- ]?to[- ]?back\s+(?:tools?|forge)|"
    r"architect_new_tool"
    r")\b",
    re.IGNORECASE,
)
# Titan self-repair swarm — read bug_tracker + draft pending patches.
_TITAN_REPAIR_HINT_RE = re.compile(
    r"\b("
    r"(?:run|start|launch|trigger)\s+(?:the\s+)?(?:titan\s+)?(?:self[- ]?)?repair|"
    r"titan\s+repair|self[- ]?heal(?:ing)?|fix\s+(?:your\s+)?bugs|"
    r"repair\s+(?:the\s+)?bug\s+tracker|autonomous\s+bug\s+tracker"
    r")\b",
    re.IGNORECASE,
)
# Todo basket / pending bugs listing.
_TODO_BASKET_HINT_RE = re.compile(
    r"\b("
    r"list\s+(?:the\s+)?todo\s+basket|show\s+(?:the\s+)?todo\s+basket|"
    r"(?:list|show)\s+pending\s+bugs|what(?:'s|\s+is)\s+in\s+the\s+bug\s+tracker|"
    r"open\s+bugs|pending\s+bugs"
    r")\b",
    re.IGNORECASE,
)
# Cursor IDE handoff — complex self-healing delegation.
_CURSOR_HANDOFF_HINT_RE = re.compile(
    r"\b("
    r"delegate\s+to\s+cursor|hand(?:\s+it)?\s*[- ]?off\s+to\s+cursor|cursor\s+handoff|"
    r"write\s+(?:an?\s+)?implementation\s+plan|"
    r"fix\s+my\s+bug(?:\s+and\s+hand(?:\s+it)?\s*[- ]?off(?:\s+to\s+cursor)?)?|"
    r"open\s+cursor\s+and\s+(?:fix|patch|implement)"
    r")\b",
    re.IGNORECASE,
)
# Async self-improvement handoff via draft_cursor_prompt (not live hot-patch).
# Prefer ticket / ledger language over bare "watchdog" alias hits.
_DRAFT_CURSOR_PROMPT_HINT_RE = re.compile(
    r"\b("
    r"draft[_ ]cursor[_ ]prompt|"
    r"write\s+(?:an?\s+)?upgrade\s+ticket|"
    r"write\s+(?:a\s+)?ticket|"
    r"log\s+(?:a\s+)?(?:self[- ]?improvement\s+)?ticket|"
    r"create\s+(?:a\s+|an\s+)?(?:self[- ]?improvement|architectural)\s+ticket|"
    r"self[- ]?improvement\s+ticket|"
    r"self[- ]?improvement\s+handoff|"
    r"architectural\s+ticket|"
    r"patch\s+ledger|"
    r"upgrade\s+(?:your\s+|my\s+)?(?:own\s+)?repository|"
    r"modify\s+your\s+own\s+core\s+files|"
    r"architectural\s+change"
    r")\b",
    re.IGNORECASE,
)


def parse_draft_cursor_prompt_args(raw: str) -> dict[str, str]:
    """Extract objective / context from a structured upgrade ticket request.

    Applies voice sanitizer + topic→file auto-map so payloads include concrete
    Target Files; refactoring steps come from the LLM/user request only.
    """
    text = (raw or "").strip()
    args: dict[str, str] = {}
    if not text:
        return args

    # Section headers are line-anchored; body may span lines until the next header.
    obj_m = re.search(
        r"(?im)^\s*[-*]?\s*Objective\s*:\s*(.+?)(?=\n\s*[-*]?\s*Target\s+Files\s*:|\n\s*[-*]?\s*Context\s*:|\Z)",
        text,
        flags=re.DOTALL,
    )
    ctx_m = re.search(
        r"(?im)^\s*[-*]?\s*Context\s*:\s*(.+)\Z",
        text,
        flags=re.DOTALL,
    )

    if obj_m:
        args["objective"] = obj_m.group(1).strip()
    if ctx_m:
        args["context"] = ctx_m.group(1).strip()

    if not args.get("objective"):
        first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        if not first:
            first = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip()
        # Full first sentence — never mid-word character caps.
        args["objective"] = first or "Self-improvement ticket"
    # Do not copy the full raw string into context when sections were absent —
    # enrich_draft_cursor_args scrubs raw_text once; duplicating it here caused
    # stacked "Technical intent:" / "Donna," echoes in patch_ledger.md.
    if not args.get("context"):
        args["context"] = ""

    try:
        from dana.agentic import enrich_draft_cursor_args

        enriched = enrich_draft_cursor_args(
            raw_text=text,
            objective=str(args.get("objective") or ""),
            context=str(args.get("context") or ""),
        )
        return {
            "objective": enriched["objective"],
            "context": enriched["context"],
        }
    except Exception:  # noqa: BLE001
        if not args.get("context"):
            args["context"] = text
        return args
# OS computer-use — screen capture / physical typing.
_SCREEN_CAPTURE_HINT_RE = re.compile(
    r"\b("
    r"capture\s+(?:and\s+analyze\s+)?(?:my\s+)?screen|"
    r"analyze\s+(?:my\s+)?screen|screenshot\s+(?:and\s+)?(?:describe|analyze)|"
    r"what(?:'s|\s+is)\s+on\s+(?:my\s+)?screen\s+right\s+now"
    r")\b",
    re.IGNORECASE,
)
_OS_TYPE_HINT_RE = re.compile(
    r"\b("
    r"type\s+.+\s+into\s+the\s+(?:focused\s+)?window|"
    r"type\s+[\"'].+[\"']|"
    r"execute\s+os\s+keystrokes|physical(?:ly)?\s+type|"
    r"press\s+ctrl\s*\+\s*[a-z]"
    r")\b",
    re.IGNORECASE,
)
# Slide review composite — capture → Cascade judge → type into active window.
_SLIDE_REVIEW_HINT_RE = re.compile(
    r"\b("
    r"evaluate\s+(?:the\s+)?slide|"
    r"evaluate_slide_and_type|"
    r"slide\s+on\s+(?:my\s+)?screen|"
    r"slide\s+review|"
    r"check\s+if\s+(?:the\s+)?slide\s+follows|"
    r"type\s+(?:your\s+)?evaluation\s+summary|"
    r"stealth\s+keystrokes|"
    r"clear\s+title\s+and\s+less\s+than\s+\d+\s+words"
    r")\b",
    re.IGNORECASE,
)
# Promote custom forge tool → general library.
_PUBLISH_TOOL_HINT_RE = re.compile(
    r"\b("
    r"publish\s+(?:tool\s+)?(?:to\s+)?general|"
    r"promote\s+(?:(?:this|the|a)\s+)?(?:custom\s+)?tool(?:\s+to\s+general)?|"
    r"publish_tool_to_general|"
    r"move\s+(?:(?:this|the)\s+)?tool\s+to\s+general"
    r")\b",
    re.IGNORECASE,
)
# Explicit visual / spatial triggers required for describe_spatial_scene.
# NOTE: never include empty ``\s*`` alternatives — they match every string.
_VISUAL_HINT_RE = re.compile(
    r"("
    r"\bwhat am i looking at\b|\bwhat do you see\b|\bon my screen\b|"
    r"\bcheck what(?:'s|\s+is)\s+on\s+my\s+screen\b|"
    r"\bdescribe the (?:screen|room|scene)\b|\blooking at\b"
    r")",
    re.IGNORECASE,
)
_WEBCAM_HINT_RE = re.compile(
    r"\b(?:webcam|camera|look at me|on camera)\b",
    re.IGNORECASE,
)
# Florence-2 OCR / UI grounding (read text, find buttons, terminal traceback).
# Sidekick: OCR modality must win over YOLO for rules/paragraph/page reads.
_OCR_GROUND_HINT_RE = re.compile(
    r"\b("
    r"ocr|"
    r"read\s+the\s+text|"
    r"read\s+(?:my\s+)?(?:the\s+)?(?:screen|page|rules|paragraph|instructions)|"
    r"read\s+(?:what(?:'s|\s+is)\s+)?on\s+(?:my\s+)?(?:the\s+)?screen|"
    r"ingest\s+(?:the\s+)?(?:rules|ocr|text)|"
    r"capture\s+(?:the\s+)?(?:rules|text)|"
    r"what\s+(?:does|do)\s+(?:the|this)\s+(?:page|rules?|paragraph)\s+say|"
    r"read\s+(?:my\s+)?(?:the\s+)?terminal|"
    r"text\s+in\s+(?:my\s+)?(?:the\s+)?terminal|"
    r"traceback|"
    r"where\s+is\s+the\s+\w[\w\s-]{0,40}button|"
    r"find\s+the\s+\w[\w\s-]{0,40}button|"
    r"submit\s+button|"
    r"ground(?:ing)?\s+(?:the\s+)?(?:text|ui|button|label)|"
    r"what\s+does\s+(?:the|this)\s+(?:button|label|dialog)\s+say"
    r")\b",
    re.IGNORECASE,
)
# Multi-step local execution chains (write → run → read). Must bind the full
# REPL suite so foresight cannot starve python_repl for a lone file_editor hit.
_REPL_SUITE_HINT_RE = re.compile(
    r"("
    r"\b(?:python_repl|shell_execute|execute_powershell|file_editor)\b|"
    r"\b(?:write|create|generate)\s+(?:a\s+|some\s+)?(?:python\s+|py\s+)?"
    r"(?:script|code|program|snippet)\b|"
    r"\b(?:run|execute)\s+(?:the\s+|this\s+|it\b|them\b|(?:python\s+)?"
    r"(?:script|code|program))\b|"
    r"\b(?:then|and)\s+(?:read|open|check|inspect)\b|"
    r"\bread\s+(?:the\s+)?(?:result|output|logfile)\b|"
    r"\bwrite\b.+\b(?:run|execute)\b|\b(?:run|execute)\b.+\bread\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_REPL_SUITE_TOOL_IDS: tuple[str, ...] = (
    "python_repl",
    "shell_execute",
    "execute_powershell",
    "file_editor",
    "run_terminal_command",
)
# Windows PowerShell / OS shell actuators — bind even when aliases miss
# ("Use PowerShell to…") and the multi-step REPL suite regex does not fire.
# Bare "shell" nouns are avoided; require an execution cue (use/run/…) or
# cmd/terminal/command-prompt phrasing so chat about "shells" does not over-bind.
_POWERSHELL_HINT_RE = re.compile(
    r"("
    r"\b(?:powershell|pwsh|execute_powershell)\b|"
    r"\b(?:get-netadapter|get-process|get-service|get-childitem)\b|"
    r"\bnetwork\s+adapters?\b|"
    r"\b(?:use|run|execute|open)\s+(?:(?:a|the|my)\s+)?"
    r"(?:powershell|pwsh|cmd|command\s*prompt|terminal|shell)\b|"
    r"\b(?:in|via|with|through)\s+(?:(?:a|the|my)\s+)?"
    r"(?:powershell|pwsh|cmd|command\s*prompt)\b|"
    r"\b(?:shell|terminal|cmd)\s+(?:command|script|to)\b|"
    r"\bcommand\s*prompt\b"
    r")",
    re.IGNORECASE,
)
# Jason/Titan conversational supervisor → Donna coder handoff (not Watchdog).
_JASON_SUPERVISOR_HINT_RE = re.compile(
    r"("
    r"(?:hey\s+|hi\s+|ok\s+)?(?:jason|titan)\b[\s\S]{0,500}?"
    r"(?:have\s+her|ask\s+donna|donna|delegate)\b[\s\S]{0,300}?"
    r"(?:write|create|code|generate)\b|"
    r"(?:hey\s+|hi\s+)?(?:jason|titan)\b[\s\S]{0,300}?"
    r"\bread\b[\s\S]{0,160}?\bnotes\.txt\b"
    r")",
    re.IGNORECASE,
)
# Dual-intent: conversational Q + create/write a notes/summary file in one turn.
_FILE_WRITE_HINT_RE = re.compile(
    r"("
    r"\b(?:create|write|save)\s+(?:(?:a|the|my)\s+)?(?:file\s+)?"
    r"[\w./\\-]+\.(?:txt|md|json|csv|log)\b|"
    r"\bnotes\.txt\b|"
    r"\bwrite\s+down\s+a\s+\d+[-\s]?point\b|"
    r"\bsave\s+(?:(?:a|the)\s+)?(?:summary|notes)\b|"
    r"\bwrite\s+(?:a\s+)?(?:summary|notes)\s+(?:to|into)\b"
    r")",
    re.IGNORECASE,
)


def _bindable_registry(registry: dict[str, ToolSpec] | None = None) -> dict[str, ToolSpec]:
    """Drop phantom ``dynamic: true`` tools that have no executable source."""
    from dana.tools.registry import _dynamic_source_available

    raw = registry if registry is not None else load_tool_registry()
    out: dict[str, ToolSpec] = {}
    for tid, spec in raw.items():
        if getattr(spec, "dynamic", False) and not _dynamic_source_available(tid):
            continue
        out[tid] = spec
    return out


# Explicit project-directory listing requests must not drift into read_local_file
# or describe_spatial_scene.
_PROJECT_LIST_RE = re.compile(
    r"\b("
    r"what(?:'s|\s+is)\s+in\s+(?:your|the|this|my)\s+project\s+(?:list|folder|directory)|"
    r"(?:show|list)\s+(?:me\s+)?(?:your|the|this|my)\s+project\s+(?:files|folder|directory|list)|"
    r"what\s+files\s+are\s+in\s+(?:your|the|this|my)\s+project|"
    r"project\s+(?:list|files|folder|directory)"
    r")\b",
    re.IGNORECASE,
)
# Named-file reads — force read_local_file, never vision.
# Bare "json"/"titan" codenames are NOT file reads (see _TITAN_PROTOCOL_RE).
_FILE_READ_RE = re.compile(
    r"\b("
    r"read\s+(?:the\s+)?(?:file|document|code)|"
    r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:file|document)|"
    r"open\s+(?:this\s+)?(?:file|document)|"
    r"show\s+(?:me\s+)?(?:the\s+)?contents?\s+of|"
    r"read\s+(?:(?:the|a|my|this|our)\s+)?[\w./\\-]+\.(?:py|txt|md|json|log|csv)"
    r")\b",
    re.IGNORECASE,
)
_GENERIC_CHAT_WORD_RE = re.compile(
    r"\b(relationship|relationships|reflection|reflections|things?)\b",
    re.IGNORECASE,
)
_EXPLICIT_LOCAL_FILE_REQUEST_RE = re.compile(
    r"\b("
    r"read|open|show|display|load|summari[sz]e|edit|modify|"
    r"look\s+up|find|inspect|check|what(?:'s|\s+is)\s+in"
    r")\b.*\b("
    r"file|files|document|documents|doc|docs|code|script|path|folder|directory|"
    r"local|project|vault"
    r")\b|"
    r"\b[\w./\\-]+\.(?:py|txt|md|json|log|csv)\b",
    re.IGNORECASE,
)
# Titan Protocol / Vanguard — spoken codenames; must not route to read_local_file.
_TITAN_PROTOCOL_RE = re.compile(
    r"\b("
    r"(?:activate|start|run|deploy|engage|launch)\s+(?:the\s+)?(?:titan|vanguard)"
    r"(?:\s+(?:initiative|protocol|supervisor))?|"
    r"titan\s+(?:initiative|protocol|supervisor)|"
    r"vanguard\s+protocol|"
    r"self[- ]?improvement\s+protocol"
    r")\b",
    re.IGNORECASE,
)
# Whisper often emits "JSON" when the user said "Jason"/"Titan".
_JSON_CODENAME_RE = re.compile(
    r"\b(?:activate|start|run|deploy|engage|launch)\s+(?:the\s+)?json"
    r"(?:\s+(?:initiative|protocol|supervisor))?\b",
    re.IGNORECASE,
)
# Memory recall — force vault, never vision.
_MEM_READ_RE = re.compile(
    r"\b("
    r"what(?:'s|\s+is)\s+my\s+(?:name|wife|partner|timezone|saved)|"
    r"who(?:'s|\s+is)\s+my\s+(?:wife|partner)|"
    r"recall\s+my|from\s+memory|read\s+vault|what\s+did\s+i\s+save|"
    r"what(?:'s|\s+is)\s+in\s+(?:your|the)\s+memory"
    r")\b",
    re.IGNORECASE,
)


class ToolValidationError(ValueError):
    pass


# Whisper STT often regurgitates its initial_prompt bias as a "user" utterance.
_WHISPER_BIAS_ECHO_RE = re.compile(
    r"\b("
    r"project_omega_status(?:\.txt)?|"
    r"file_jail_enforcer|"
    r"confidential\s+status\s+report|"
    r"project\s+omega"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_whisper_bias_echo(raw: str) -> bool:
    """True for STT echoes of Whisper bias / confidential fixture names."""
    text = (raw or "").strip()
    if not text:
        return False
    hits = _WHISPER_BIAS_ECHO_RE.findall(text)
    if len(hits) >= 2:
        return True
    # Lone Omega "read the file …" with no path / extra intent → bias echo.
    # Explicit paths (e.g. docs/project_omega_status.txt) are real file reads.
    low = text.lower()
    if "project_omega" in low and re.search(r"\bread\s+(?:the\s+)?file\b", low):
        if re.search(r"[\w./\\-]+\.(?:py|txt|md|json|log|csv)\b", low):
            return False
        if not re.search(
            r"\b(summarize|explain|what\s+does|tell\s+me\s+about)\b", low
        ):
            return True
    return False


def _should_force_file_read(raw: str, *, titan_hit: bool = False) -> bool:
    """Return true only for explicit local file/document/code requests."""
    if titan_hit:
        return False
    if _looks_like_whisper_bias_echo(raw):
        return False
    if not _FILE_READ_RE.search(raw):
        return False
    if _GENERIC_CHAT_WORD_RE.search(raw) and not _EXPLICIT_LOCAL_FILE_REQUEST_RE.search(raw):
        return False
    return bool(_EXPLICIT_LOCAL_FILE_REQUEST_RE.search(raw))


def _foresight_cascade(raw: str, tool_id: str | None) -> None:
    """Broker-side cognitive routing foresight (local llama vs Cascade)."""
    try:
        from dana.cascade_router import decide_route
        from dana.logging import log

        decision = decide_route(raw, forced_tool=tool_id)
        log(
            "Cascade",
            f"broker foresight tool={tool_id or '-'} → "
            f"{decision.backend}/{decision.complexity} model={decision.model}",
        )
    except Exception:
        pass


def explicit_tool_ids_in_text(
    raw: str,
    known_ids: Iterable[str] | None = None,
) -> list[str]:
    """Pre-flight keyword sweep: tool ids mentioned verbatim in the user prompt."""
    blob = (raw or "").lower()
    if not blob.strip():
        return []
    if known_ids is None:
        try:
            known_ids = list(load_tool_registry().keys())
        except Exception:  # noqa: BLE001
            known_ids = []
    found: list[str] = []
    # Longer ids first so nested/partial names do not shadow full tool ids.
    for tid in sorted((str(x) for x in known_ids), key=len, reverse=True):
        name = tid.strip().lower()
        if len(name) < 3:
            continue
        if name in blob:
            found.append(tid)
    return list(dict.fromkeys(found))


def repl_suite_tool_ids(user_text: str) -> list[str]:
    """Return REPL/exec/file tools when the utterance is a multi-step local chain."""
    if not _REPL_SUITE_HINT_RE.search(user_text or ""):
        return []
    try:
        known = set(load_tool_registry().keys())
    except Exception:  # noqa: BLE001
        known = set(_REPL_SUITE_TOOL_IDS)
    return [tid for tid in _REPL_SUITE_TOOL_IDS if tid in known]


def merge_bound_tool_ids(
    *,
    user_text: str,
    forced_tool_id: str | None = None,
    mode: str | None = None,
    known_ids: Iterable[str] | None = None,
) -> list[str]:
    """Merge mode/forced foresight with explicitly named tools (deduped).

    Vision mode keeps ``analyze_visual_context`` bound (JIT vision path) while
    any tool id spelled in the raw prompt is appended so mode overrides cannot
    starve an explicit request (e.g. ``draft_cursor_prompt``).
    OCR/grounding prompts also keep ``ocr_with_region`` (Florence-2) bound.
    Multi-step write/run/read prompts bind the full local REPL suite so
    ``file_editor`` foresight cannot starve ``python_repl`` / ``shell_execute``.
    PowerShell / network-adapter OS prompts always bind ``execute_powershell``
    so semantic top-K cannot starve the shell actuator with vision/memory tools.
    """
    if known_ids is None:
        try:
            known_ids = list(load_tool_registry().keys())
        except Exception:  # noqa: BLE001
            known_ids = []
    known = {str(x) for x in known_ids}
    merged: list[str] = []

    def _add(tid: str | None) -> None:
        name = (tid or "").strip()
        if not name or name not in known:
            return
        if name not in merged:
            merged.append(name)

    _add(forced_tool_id)
    mode_norm = (mode or "").strip().lower()
    jason_sup = (forced_tool_id or "").strip() == "dispatch_jason_supervisor" or bool(
        _JASON_SUPERVISOR_HINT_RE.search(user_text or "")
    )
    if mode_norm == "vision":
        _add("analyze_visual_context")
        if _OCR_GROUND_HINT_RE.search(user_text or ""):
            _add("ocr_with_region")
    # Jason→Donna supervisor owns file reads/writes inside its graph — do not
    # also bind dual-intent file_editor on the outer ReAct loop (avoids loops).
    if _FILE_WRITE_HINT_RE.search(user_text or "") and not jason_sup:
        _add("file_editor")
    if not jason_sup:
        for tid in repl_suite_tool_ids(user_text or ""):
            _add(tid)
        if _POWERSHELL_HINT_RE.search(user_text or ""):
            _add("execute_powershell")
    for tid in explicit_tool_ids_in_text(user_text, known):
        if jason_sup and tid != "dispatch_jason_supervisor":
            continue
        _add(tid)
    # Stage 4.1 — lightweight Chat never binds live vision; it SELECTs the BB topic.
    if mode_norm == "chat":
        return [tid for tid in merged if tid != "analyze_visual_context"]
    return merged


def initialize_tool_registry() -> list:
    """Eagerly load dynamic tool objects outside the broker dispatch hot path.

    Returns a list of initialized tool objects as
    ``(tool_id, kind, payload)`` tuples where ``kind`` is ``\"plugin\"`` or
    ``\"registry\"``.
    """
    from dana.tools.plugins import list_plugin_ids, resolve_plugin_handler
    from dana.tools.registry import get_tool_registry
    from dana.tools.schema import ToolCall

    initialized: list[Any] = []
    for tool_id in list_plugin_ids():
        plugin = resolve_plugin_handler(tool_id)
        if plugin is not None:
            initialized.append((str(tool_id), "plugin", plugin))
    try:
        reg = get_tool_registry()
        for name, entry in list(reg.tools.items()):
            if (
                entry is not None
                and entry.callable is not None
                and not entry.is_ephemeral
            ):
                initialized.append((str(name), "registry", entry))
    except Exception:  # noqa: BLE001
        pass

    # Wired / Bluetooth headphone endpoint switcher (Windows Core Audio).
    def _handle_toggle_audio(call: ToolCall) -> str:
        from dana.tools.audio_switcher import toggle_audio_endpoint

        args = dict(call.arguments or {})
        target = str(args.get("target_type") or args.get("target") or "").strip()
        return toggle_audio_endpoint(target)

    initialized.append(("toggle_audio_endpoint", "plugin", _handle_toggle_audio))

    # Sweep RESOLVED/FAILED tickets into patch_ledger_archive.md.
    def _handle_archive_ledger(call: ToolCall) -> str:
        from dana.tools.archive_ledger import archive_completed_tickets

        return archive_completed_tickets()

    initialized.append(("archive_ledger", "plugin", _handle_archive_ledger))

    # Always wire draft_cursor_prompt (do not wait on general disk hot-load).
    def _handle_draft_cursor_prompt(call: ToolCall) -> str:
        from dana.tools.general.draft_cursor_prompt import draft_cursor_prompt

        args = dict(call.arguments or {})
        return draft_cursor_prompt(
            objective=str(args.get("objective") or ""),
            context=str(args.get("context") or ""),
        )

    initialized.append(
        ("draft_cursor_prompt", "plugin", _handle_draft_cursor_prompt)
    )
    return initialized


_FILE_MODIFICATION_TOOL_IDS = frozenset({"edit_file", "write_file", "delete_file"})


class IntentBroker:
    """Language-agnostic tool router with structural validation + self-correction."""

    def __init__(self, registry: dict[str, ToolSpec] | None = None) -> None:
        self.registry = _bindable_registry(registry)
        self._lessons_provider: Callable[[], list[Any]] | None = None
        self._initialized_tools: list[Any] = initialize_tool_registry()

    def set_lessons_provider(self, provider: Callable[[], list[Any]] | None) -> None:
        """Optional callback returning Lesson-like objects from the vault."""
        self._lessons_provider = provider

    def reload_registry(self, path: str | None = None) -> dict[str, ToolSpec]:
        """Hot-reload tools.json (including dynamically registered tools)."""
        self.registry = _bindable_registry(load_tool_registry(path))
        self._initialized_tools = initialize_tool_registry()
        return self.registry

    def public_schema(self) -> list[dict[str, Any]]:
        return tool_schema_public(self.registry)

    def lessons_for_utterance(
        self,
        text: str,
        *,
        tool_hint: str | None = None,
    ) -> list[Any]:
        """Retrieve lessons_learned relevant to this intent (for prompt injection)."""
        if self._lessons_provider is None:
            return []
        try:
            from dana.reflector import match_lessons_for_intent

            lessons = list(self._lessons_provider() or [])
            hint = tool_hint
            if hint is None:
                parsed = self.parse_utterance(text)
                if parsed is not None:
                    hint = parsed.tool_id
            return match_lessons_for_intent(text, lessons, tool_hint=hint)
        except Exception:  # noqa: BLE001
            return []

    def augment_system_prompt(self, system_prompt: str, user_text: str) -> str:
        """Prepend matched lessons_learned into the system prompt.

        Also injects TOOL_EXECUTION, STRICT_TOOL_ENFORCEMENT, explicit-tool CRITICAL,
        R1_REASONING, VOICE_SANITIZER, INTERACTION_UX, TECHNICAL PRODUCT MANAGER,
        and TERMINATION.
        """
        tool_exec = (
            "TOOL_EXECUTION_RULE: You are an autonomous agent, not a conversational "
            "chatbot. You have access to functional tools. When instructed to log a "
            "ticket or perform an action, you MUST invoke the `draft_cursor_prompt` "
            "tool using the system's native tool-calling schema (JSON/function call). "
            "Do NOT output bash commands, CLI scripts, or markdown code blocks "
            "instructing the user how to do it. You must physically execute the "
            "function yourself. "
            "Once the tool returns a success message, output your ONE sentence "
            "summary and terminate."
        )
        strict_tool = (
            "STRICT_TOOL_ENFORCEMENT_RULE: Your sole authorized method for logging "
            "tickets is the `draft_cursor_prompt` tool. You are strictly FORBIDDEN "
            "from generating bash commands, CLI snippets, or markdown code blocks as "
            "a substitute for tool execution. If you believe a CLI command is part of "
            "the solution, you MUST include that command INSIDE the 'context' "
            "argument of the `draft_cursor_prompt` tool, not as plain text in your "
            "response. If you do not call the tool, you have failed the task."
        )
        explicit_tool = (
            "CRITICAL: If the user explicitly asks to use a tool (e.g., draft_cursor_prompt), "
            "you MUST invoke that tool's JSON schema in your response before outputting a "
            "final natural language answer. Do not claim a task or ticket has been logged "
            "unless the tool execution trace confirms it."
        )
        r1 = (
            "R1_REASONING_RULE: You are a reasoning model. You MUST use your "
            "`<think> ... </think>` block to plan the architectural ticket and "
            "context payload. Immediately after your closing `</think>` tag, you "
            "MUST execute the `draft_cursor_prompt` tool natively. Do not output any "
            "conversational text outside the think block until the tool returns."
        )
        voice = (
            "VOICE_SANITIZER_RULE: When processing input from voice mode, you must "
            "filter out conversational wrapper text (e.g., 'use the draft cursor "
            "prompt tool to log a ticket', 'can you please log...'). Extract only the "
            "underlying technical intent. When the user omits file paths, use the "
            "voice topic map for Target Files only: audio/audio pipeline/glitches → "
            "dana/core_agent.py; cursor handling/deepseek navigation → "
            "dana/cascade_router.py and dana/agentic.py; patch ledger/cursor prompt "
            "handling → dana_security/patch_ledger.md and dana/tools/broker.py. "
            "Invent concrete refactoring steps from the user's specific request — "
            "never reuse generic template steps. Put paths + your dynamic steps into "
            "the `context` argument of `draft_cursor_prompt`."
        )
        ux = (
            "INTERACTION_UX_RULE: You are a highly capable technical partner. "
            "After a tool successfully executes, output ONE brief, casual sentence "
            "confirming completion (e.g., 'The ticket is on the board.'). Never read "
            "raw code, JSON, or markdown out loud."
        )
        tpm = (
            "TECHNICAL PRODUCT MANAGER RULE: When the user gives a high-level or "
            "casual voice command for a code change, you must act as a Technical "
            "Product Manager. Translate their vague request into a highly detailed "
            "technical prompt for the Cursor IDE. If the user does not provide file "
            "paths, use your reasoning to outline clear architectural goals, logic "
            "steps, and acceptance criteria. Pass your detailed architectural plan "
            "directly into the `context` argument of the function call. Do not ask "
            "the user for more details—expand their intent into a usable developer "
            "ticket."
        )
        termination = (
            "TERMINATION RULE: Your sole job is to log the ticket using the tool. "
            "Once the `draft_cursor_prompt` tool returns a success message, you MUST "
            "immediately end your response with a simple confirmation (e.g., "
            "'Ticket logged.'). Do NOT attempt to write code, solve the problem, or "
            "explain the architecture."
        )
        prompt = system_prompt or ""
        if tool_exec not in prompt:
            prompt = f"{prompt}\n\n{tool_exec}"
        if strict_tool not in prompt:
            prompt = f"{prompt}\n\n{strict_tool}"
        if explicit_tool not in prompt:
            prompt = f"{prompt}\n\n{explicit_tool}"
        if r1 not in prompt:
            prompt = f"{prompt}\n\n{r1}"
        if voice not in prompt:
            prompt = f"{prompt}\n\n{voice}"
        if ux not in prompt:
            prompt = f"{prompt}\n\n{ux}"
        if tpm not in prompt:
            prompt = f"{prompt}\n\n{tpm}"
        if termination not in prompt:
            prompt = f"{prompt}\n\n{termination}"
        lessons = self.lessons_for_utterance(user_text)
        if not lessons:
            return prompt
        try:
            from dana.reflector import inject_lessons_into_prompt

            return inject_lessons_into_prompt(prompt, lessons)
        except Exception:  # noqa: BLE001
            return prompt

    def parse_utterance(self, text: str) -> ToolCall | None:
        """Map EN/FA speech or pseudo tool-call syntax into a ToolCall IR."""
        raw = (text or "").lstrip("\ufeff").strip()
        if not raw:
            return None
        lang = detect_lang(raw)
        norm = normalize_text(raw).lower() if lang != "en" else raw.lower()

        structured = self._parse_structured(raw, lang)
        if structured is not None:
            return self.validate_and_correct(structured)

        best: tuple[int, ToolCall] | None = None
        for spec in self.registry.values():
            alias_maps = []
            if lang in ("en", "mixed"):
                alias_maps.append(spec.aliases_en)
            if lang in ("fa", "mixed"):
                alias_maps.append(spec.aliases_fa)
            if not alias_maps:
                alias_maps = [spec.aliases_en, spec.aliases_fa]

            haystack = norm if lang in ("fa", "mixed") else raw.lower()
            for amap in alias_maps:
                for enum_val, phrases in amap.items():
                    for phrase in phrases:
                        needle = (
                            normalize_text(phrase)
                            if lang in ("fa", "mixed")
                            else phrase.lower()
                        )
                        if not self._phrase_hit(haystack, needle):
                            continue
                        args: dict[str, Any] = {}
                        if not str(enum_val).startswith("_") and spec.parameters:
                            first = spec.parameters[0]
                            if first.enum and enum_val in first.enum:
                                args[first.name] = enum_val
                            elif not first.enum and not str(enum_val).startswith("_"):
                                args[first.name] = enum_val
                        call = ToolCall(
                            tool_id=spec.id,
                            arguments=args,
                            source_lang=lang,
                            raw_text=raw,
                            confidence=0.85,
                        )
                        score = len(needle)
                        if best is None or score > best[0]:
                            if str(enum_val).startswith("_"):
                                best = (score, call)
                            else:
                                try:
                                    best = (score, self.validate_and_correct(call))
                                except ToolValidationError:
                                    best = (score, call)

        # Routing guardrail: schedule/event queries must not fall through
        # to describe_spatial_scene (common when STT is garbled).
        # Wall-clock time queries stay tool-free (System Clock hot-cache).
        wall_clock_hit = bool(_WALL_CLOCK_RE.search(raw))
        schedule_hit = bool(_SCHEDULE_HINT_RE.search(raw)) and not wall_clock_hit
        visual_hit = bool(_VISUAL_HINT_RE.search(raw))
        research_hit = bool(_RESEARCH_HINT_RE.search(raw))
        deep_research_hit = bool(_DEEP_RESEARCH_RE.search(raw))
        mem_write_hit = bool(_MEM_WRITE_HINT_RE.search(raw))
        project_hit = bool(_PROJECT_LIST_RE.search(raw))
        titan_hit = bool(_TITAN_PROTOCOL_RE.search(raw)) or bool(
            _JSON_CODENAME_RE.search(raw)
        )
        file_hit = _should_force_file_read(raw, titan_hit=titan_hit)
        mem_read_hit = bool(_MEM_READ_RE.search(raw))
        forge_hit = bool(_TOOL_FORGE_HINT_RE.search(raw))
        titan_repair_hit = bool(_TITAN_REPAIR_HINT_RE.search(raw))
        todo_basket_hit = bool(_TODO_BASKET_HINT_RE.search(raw))
        cursor_handoff_hit = bool(_CURSOR_HANDOFF_HINT_RE.search(raw))
        draft_cursor_hit = bool(_DRAFT_CURSOR_PROMPT_HINT_RE.search(raw))
        screen_capture_hit = bool(_SCREEN_CAPTURE_HINT_RE.search(raw))
        os_type_hit = bool(_OS_TYPE_HINT_RE.search(raw))
        slide_review_hit = bool(_SLIDE_REVIEW_HINT_RE.search(raw))
        publish_hit = bool(_PUBLISH_TOOL_HINT_RE.search(raw))
        jason_supervisor_hit = bool(_JASON_SUPERVISOR_HINT_RE.search(raw))
        if wall_clock_hit and not visual_hit:
            _foresight_cascade(raw, None)
            return None
        # STT regurgitated Whisper bias / Omega fixture names → no tool, no vault.
        if _looks_like_whisper_bias_echo(raw) and not forge_hit and not mem_write_hit:
            _foresight_cascade(raw, None)
            return None
        # Jason/Titan supervisor → Donna coder (before notes.txt dual-intent /
        # Watchdog titan protocol so "Hey Jason, read notes… have her write" works).
        if jason_supervisor_hit and not mem_write_hit and not forge_hit:
            _foresight_cascade(raw, "dispatch_jason_supervisor")
            return ToolCall(
                tool_id="dispatch_jason_supervisor",
                arguments={"query": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.98,
            )
        # Slide review composite BEFORE bare screen-capture / type shortcuts.
        if slide_review_hit and not mem_write_hit:
            rule = raw
            m = re.search(
                r"(?:rule\s+of|follows?\s+the\s+rule\s+of|check\s+if\s+it\s+follows)\s+(.+?)(?:\.|,|\s+then\b|$)",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m:
                rule = m.group(1).strip(" .")
            _foresight_cascade(raw, "evaluate_slide_and_type")
            return ToolCall(
                tool_id="evaluate_slide_and_type",
                arguments={"rule": rule or raw, "focus_delay_sec": 1.5},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        # Self-improvement / upgrade-ticket → draft_cursor_prompt (force high MoA).
        # Prefer this over generic delegate_to_cursor when the async draft tool is named.
        if draft_cursor_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "draft_cursor_prompt")
            return ToolCall(
                tool_id="draft_cursor_prompt",
                arguments=parse_draft_cursor_prompt_args(raw),
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        if cursor_handoff_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "delegate_to_cursor")
            return ToolCall(
                tool_id="delegate_to_cursor",
                arguments={"query": raw, "goal": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        if screen_capture_hit and not mem_write_hit:
            _foresight_cascade(raw, "capture_and_analyze_screen")
            return ToolCall(
                tool_id="capture_and_analyze_screen",
                arguments={"prompt": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        # Dual-intent: conversational answer + create/write notes/summary file.
        # Prefer file_editor BEFORE forge/phantom "build_tool_*" alias traps.
        write_hit = bool(_FILE_WRITE_HINT_RE.search(raw))
        if write_hit and not mem_write_hit and not forge_hit:
            path = "notes.txt"
            m = re.search(
                r"([\w./\\-]+\.(?:txt|md|json|csv|log))",
                raw,
                flags=re.I,
            )
            if m:
                path = m.group(1).replace("\\", "/")
            _foresight_cascade(raw, "file_editor")
            return ToolCall(
                tool_id="file_editor",
                arguments={"action": "write", "filepath": path},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        # Multi-step write→run→read: force python_repl FIRST so foresight cannot
        # jump straight to file_editor on a not-yet-written path.
        repl_chain = bool(_REPL_SUITE_HINT_RE.search(raw))
        if repl_chain and not mem_write_hit and not forge_hit:
            _foresight_cascade(raw, "python_repl")
            return ToolCall(
                tool_id="python_repl",
                arguments={},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        # Bare PowerShell / network-adapter OS asks (aliases miss "Use PowerShell…").
        ps_hit = bool(_POWERSHELL_HINT_RE.search(raw))
        if ps_hit and not mem_write_hit and not forge_hit and not write_hit:
            _foresight_cascade(raw, "execute_powershell")
            return ToolCall(
                tool_id="execute_powershell",
                arguments={},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        # Local project file reads beat OCR/vision — "read foo.py" is not on-screen OCR.
        # Skip when this is a REPL chain (handled above).
        if file_hit and not mem_write_hit and not repl_chain:
            path = ""
            m = re.search(
                r"([\w./\\-]+\.(?:py|txt|md|json|log|csv|yml|yaml|toml|ini))",
                raw,
                flags=re.I,
            )
            if m:
                path = m.group(1).replace("\\", "/")
                # Bare module filenames often live under dana/tools/.
                if "/" not in path and path.lower().endswith(".py"):
                    candidate = f"dana/tools/{path}"
                    try:
                        from dana.paths import PROJECT_ROOT

                        if (PROJECT_ROOT / candidate).is_file():
                            path = candidate
                        elif (PROJECT_ROOT / "dana" / path).is_file():
                            path = f"dana/{path}"
                    except Exception:  # noqa: BLE001
                        pass
            _foresight_cascade(raw, "file_editor")
            args: dict[str, str] = {"action": "read"}
            if path:
                args["filepath"] = path
            return ToolCall(
                tool_id="file_editor",
                arguments=args,
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        # Florence-2 OCR / UI grounding before generic YOLO visual look/see.
        ocr_hit = bool(_OCR_GROUND_HINT_RE.search(raw))
        if ocr_hit and not mem_write_hit:
            q = ""
            m = re.search(
                r"(?:where\s+is|find|locate|read)\s+(?:the\s+)?(.+?)(?:\s+on\s+(?:my\s+)?screen)?\s*$",
                raw,
                flags=re.I,
            )
            if m:
                q = m.group(1).strip(" ?.!,")
            _foresight_cascade(raw, "ocr_with_region")
            return ToolCall(
                tool_id="ocr_with_region",
                arguments={"query": q} if q else {},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        # Visual look/see → JIT YOLO analyze_visual_context (bound in LangGraph).
        if visual_hit and not mem_write_hit:
            source = "webcam" if _WEBCAM_HINT_RE.search(raw) else "screen"
            _foresight_cascade(raw, "analyze_visual_context")
            return ToolCall(
                tool_id="analyze_visual_context",
                arguments={"source": source},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        if os_type_hit and not visual_hit and not mem_write_hit:
            # Prefer quoted text, then "type X into the window", else strip verb.
            m = re.search(r"[\"']([^\"']+)[\"']", raw)
            if not m:
                m = re.search(
                    r"\btype\s+(.+?)\s+into\s+the\b", raw, flags=re.IGNORECASE
                )
            typed = m.group(1).strip() if m else re.sub(
                r"^\s*(?:please\s+)?type\s+", "", raw, flags=re.I
            ).strip()
            _foresight_cascade(raw, "execute_os_keystrokes")
            return ToolCall(
                tool_id="execute_os_keystrokes",
                arguments={"text": typed or raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.95,
            )
        if todo_basket_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "list_todo_basket")
            return ToolCall(
                tool_id="list_todo_basket",
                arguments={},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        if titan_repair_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "dispatch_titan_repair")
            return ToolCall(
                tool_id="dispatch_titan_repair",
                arguments={"query": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        # Build/create/code a tool → Tool Forge payload = exact user request.
        # MUST NEVER fall through to read_vault_memory or generic chat.
        if forge_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "architect_new_tool")
            return ToolCall(
                tool_id="architect_new_tool",
                arguments={"goal": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        # Promote custom → general (admin) — before generic chat / vault.
        if publish_hit and not visual_hit and not mem_write_hit and not forge_hit:
            tool_name_arg = ""
            m = re.search(
                r"(?:promote|publish)\s+(?:tool\s+)?[`'\"]?([A-Za-z_][\w]*)[`'\"]?",
                raw,
                flags=re.I,
            )
            if m and m.group(1).lower() not in {
                "tool",
                "to",
                "the",
                "this",
                "a",
                "custom",
                "general",
            }:
                tool_name_arg = m.group(1)
            _foresight_cascade(raw, "publish_tool_to_general")
            return ToolCall(
                tool_id="publish_tool_to_general",
                arguments={"tool_name": tool_name_arg} if tool_name_arg else {},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        if titan_hit and not visual_hit:
            return ToolCall(
                tool_id="dispatch_watchdog",
                arguments={"task": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )
        if project_hit and not visual_hit:
            return ToolCall(
                tool_id="run_terminal_command",
                arguments={"command": "dir"},
                source_lang=lang,
                raw_text=raw,
                confidence=0.98,
            )
        if mem_read_hit and not visual_hit and not mem_write_hit and not forge_hit:
            return ToolCall(
                tool_id="read_vault_memory",
                arguments={},
                source_lang=lang,
                raw_text=raw,
                confidence=0.95,
            )
        # Hard ban: never let describe_spatial_scene win without an explicit visual cue.
        if (
            best is not None
            and best[1].tool_id == "describe_spatial_scene"
            and not visual_hit
        ):
            best = None
        if schedule_hit and not visual_hit:
            if best is None or best[1].tool_id == "describe_spatial_scene":
                return ToolCall(
                    tool_id="web_search",
                    arguments={"query": raw},
                    source_lang=lang,
                    raw_text=raw,
                    confidence=0.9,
                )
        if (
            best is not None
            and best[1].tool_id == "describe_spatial_scene"
            and not visual_hit
            and schedule_hit
        ):
            return ToolCall(
                tool_id="web_search",
                arguments={"query": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.9,
            )
        # Force deep-research swarm (Planner→Search→Writer) with a populated query.
        # Never pre-route with empty args — the LLM must not invent a blank topic.
        if deep_research_hit and not visual_hit and not mem_write_hit:
            _foresight_cascade(raw, "dispatch_research_swarm")
            return ToolCall(
                tool_id="dispatch_research_swarm",
                arguments={"query": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.96,
            )
        # Force live search for research / latest-updates asks (never vault).
        # Always supply query=raw — alias _intent hits leave args empty.
        if research_hit and not visual_hit and not mem_write_hit:
            return ToolCall(
                tool_id="web_search",
                arguments={"query": raw},
                source_lang=lang,
                raw_text=raw,
                confidence=0.92,
            )

        if (
            best is not None
            and best[1].tool_id == "read_local_file"
            and not _should_force_file_read(raw, titan_hit=titan_hit)
        ):
            return None

        # Alias _intent hits leave args empty — always fill goal from the utterance.
        if best is not None and best[1].tool_id == "architect_new_tool":
            args = dict(best[1].arguments or {})
            if not str(args.get("goal") or args.get("tool_description") or "").strip():
                args["goal"] = raw
            return replace(best[1], arguments=args, raw_text=raw or best[1].raw_text)

        # "watchdog layer" / similar must NOT steal ticket / self-improvement intents.
        if (
            best is not None
            and best[1].tool_id == "dispatch_watchdog"
            and draft_cursor_hit
            and not visual_hit
            and not mem_write_hit
        ):
            _foresight_cascade(raw, "draft_cursor_prompt")
            return ToolCall(
                tool_id="draft_cursor_prompt",
                arguments=parse_draft_cursor_prompt_args(raw),
                source_lang=lang,
                raw_text=raw,
                confidence=0.97,
            )

        # Alias _intent hits leave args empty — fill required task from utterance.
        if best is not None and best[1].tool_id == "dispatch_watchdog":
            args = dict(best[1].arguments or {})
            if not str(args.get("task") or "").strip():
                args["task"] = raw
            return replace(best[1], arguments=args, raw_text=raw or best[1].raw_text)

        return best[1] if best else None

    def _phrase_hit(self, haystack: str, needle: str) -> bool:
        if not needle:
            return False
        if " " in needle or any("\u0600" <= ch <= "\u06ff" for ch in needle):
            return needle in haystack
        return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))

    def parse_structured(self, text: str, lang: str | None = None) -> ToolCall | None:
        """Parse TOOL:/tool: syntax or JSON-ish tool payloads into ToolCall IR."""
        return self._parse_structured(text, lang or detect_lang(text or ""))

    def _parse_structured(self, text: str, lang: str) -> ToolCall | None:
        bare = re.match(
            r"^\s*([a-zA-Z_][\w]*)\s*\((.*)\)\s*$",
            (text or "").strip(),
            re.DOTALL,
        )
        match = _TOOL_CALL_RE.search(text)
        if not match and bare:
            tool_id = bare.group(1)
            body = bare.group(2) or ""
        elif not match:
            jsonish = re.search(
                r"\{\s*[\"']?(?:tool|id|tool_id)[\"']?\s*:\s*[\"']([\w]+)[\"'](.*)\}",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if not jsonish:
                return None
            tool_id = jsonish.group(1)
            body = jsonish.group(2) or ""
        else:
            tool_id = match.group(1)
            body = match.group(2) or match.group(3) or ""

        args: dict[str, Any] = {}
        for km in _KV_RE.finditer(body):
            args[km.group(1)] = km.group(2).strip().strip("'\"")
        return ToolCall(
            tool_id=tool_id,
            arguments=args,
            source_lang=lang,
            raw_text=text,
            confidence=0.7,
        )

    def validate_and_correct(self, call: ToolCall) -> ToolCall:
        from dana.tools.registry import _dynamic_source_available

        spec = self.registry.get(call.tool_id)
        if spec is None:
            corrected_id = self._fuzzy_tool_id(call.tool_id)
            if corrected_id is None:
                raise ToolValidationError(f"Unknown tool id: {call.tool_id}")
            call = replace(call, tool_id=corrected_id)
            spec = self.registry[corrected_id]
        if getattr(spec, "dynamic", False) and not _dynamic_source_available(spec.id):
            raise ToolValidationError(
                f"Phantom dynamic tool {spec.id}: source not found"
            )

        args = normalize_tool_arguments(dict(call.arguments), call.source_lang)
        fixed: dict[str, Any] = {}
        for param in spec.parameters:
            if param.name not in args:
                if param.required:
                    if param.enum:
                        fixed[param.name] = param.enum[0]
                        continue
                    raise ToolValidationError(
                        f"Missing required argument '{param.name}' for {spec.id}"
                    )
                continue
            value = args[param.name]
            if param.enum:
                value = self._coerce_enum(str(value), param.enum, spec, param.name)
            fixed[param.name] = value

        return replace(call, arguments=fixed)

    def _fuzzy_tool_id(self, tool_id: str) -> str | None:
        tid = re.sub(r"[^a-z0-9_]", "", (tool_id or "").lower())
        if tid in self.registry:
            return tid
        for key in self.registry:
            if tid in key or key in tid:
                return key
        return None

    def _coerce_enum(
        self,
        value: str,
        enum: tuple[str, ...],
        spec: ToolSpec,
        param_name: str,
    ) -> str:
        v = value.strip().lower()
        if v in enum:
            return v
        for amap in (spec.aliases_en, spec.aliases_fa):
            for enum_val, phrases in amap.items():
                if param_name and enum_val not in enum:
                    continue
                for phrase in phrases:
                    p = normalize_text(phrase).lower()
                    if v == p or v == enum_val or phrase.lower() == v:
                        return enum_val
        for e in enum:
            if e in v or v in e:
                return e
        if len(v) >= 3:
            candidates = [e for e in enum if e.startswith(v[:3]) or v.startswith(e[:3])]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return min(candidates, key=lambda e: abs(len(e) - len(v)))
        raise ToolValidationError(
            f"Invalid value '{value}' for {spec.id}.{param_name}; expected one of {enum}"
        )

    def synthesis_guard_observation(self, call: ToolCall) -> str | None:
        """Return a LOCKED observation if architect_new_tool is disabled; else None."""
        from dana.settings import is_dynamic_tool_synthesis_enabled, synthesis_locked_message

        if call.tool_id != "architect_new_tool":
            return None
        if is_dynamic_tool_synthesis_enabled():
            return None
        return (
            "LOCKED: dynamic_tool_synthesis_disabled | "
            + synthesis_locked_message(call.source_lang or "en")
        )

    def dispatch(
        self,
        call: ToolCall,
        handlers: dict[str, Callable[[ToolCall], Any]],
    ) -> Any:
        # Production safety: never route architect_new_tool into the sandbox when locked.
        locked = self.synthesis_guard_observation(call)
        if locked is not None:
            return locked

        # Pre-execution sandbox: file-mod tools must pass watchdog dry-run first.
        if call.tool_id in _FILE_MODIFICATION_TOOL_IDS:
            from dana.swarm.watchdog_graph import verify_payload

            if not verify_payload(call.tool_id, dict(call.arguments or {})):
                return (
                    f"ERROR: {call.tool_id} blocked: dry-run verification failed"
                )

        handler = handlers.get(call.tool_id)
        if handler is None:
            # Dynamic tools preloaded via initialize_tool_registry() (not per-call import).
            resolved = self._lookup_initialized_tool(call.tool_id)
            if resolved is not None:
                kind, payload = resolved
                if kind == "plugin":
                    return payload(call)
                if kind == "registry":
                    entry = payload
                    try:
                        import inspect

                        kwargs = dict(call.arguments or {})
                        try:
                            return entry.callable(**kwargs)
                        except TypeError:
                            sig = inspect.signature(entry.callable)
                            if any(
                                p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in sig.parameters.values()
                            ):
                                return entry.callable(**kwargs)
                            filtered = {
                                k: v for k, v in kwargs.items() if k in sig.parameters
                            }
                            return entry.callable(**filtered)
                    except Exception as exc:  # noqa: BLE001
                        return f"ERROR: {call.tool_id} failed: {exc}"
            handler = handlers.get("__dynamic__")
            if handler is None:
                raise ToolValidationError(f"No handler registered for {call.tool_id}")
        return handler(call)

    def _lookup_initialized_tool(self, tool_id: str) -> tuple[str, Any] | None:
        """Resolve a pre-initialized dynamic tool; refresh once on miss."""
        tid = str(tool_id or "").strip()
        for name, kind, payload in self._initialized_tools:
            if name == tid:
                return (kind, payload)
        # Late-loaded general/plugin tools — rebuild once, then retry.
        self._initialized_tools = initialize_tool_registry()
        for name, kind, payload in self._initialized_tools:
            if name == tid:
                return (kind, payload)
        return None


    def dispatch_pending_tasks(
        self,
        handler: Callable[[str], Any],
        *,
        path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Drain ``execution_jail/task_queue.json`` pending tasks one-by-one.

        For every object with ``status == "pending"``:
          1. Extract ``command``
          2. Invoke ``handler(command)`` as an isolated agentic turn
          3. Persist ``completed`` or ``failed`` before the next task

        One failed task never aborts the loop — exceptions are caught, the
        task is marked ``failed``, and the next pending task still runs.
        """
        from dana.tools.task_queue import (
            claim_pending_tasks,
            update_task_status,
        )

        results: list[dict[str, Any]] = []
        # Atomic pop: mark running on disk BEFORE any handler runs.
        snapshot = list(claim_pending_tasks(path))
        for task in snapshot:
            tid = str(task.get("id") or "").strip() or "unknown"
            command = str(task.get("command") or "").strip()
            if not command:
                try:
                    update_task_status(
                        tid, "failed", path=path, error="empty command"
                    )
                except Exception:  # noqa: BLE001
                    pass
                results.append(
                    {"id": tid, "status": "failed", "error": "empty command"}
                )
                continue

            try:
                handler(command)
            except Exception as exc:  # noqa: BLE001 — isolate failures
                err = f"{type(exc).__name__}: {exc}"
                try:
                    update_task_status(tid, "failed", path=path, error=err)
                except Exception:  # noqa: BLE001
                    pass
                results.append({"id": tid, "status": "failed", "error": err})
                continue

            try:
                update_task_status(tid, "completed", path=path)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "id": tid,
                        "status": "failed",
                        "error": f"persist completed failed: {exc}",
                    }
                )
                continue
            results.append({"id": tid, "status": "completed"})
        return results


_BROKER: IntentBroker | None = None


def get_broker() -> IntentBroker:
    global _BROKER
    if _BROKER is None:
        _BROKER = IntentBroker()
    return _BROKER


def reload_broker_registry(path: str | None = None) -> IntentBroker:
    """Reload tools.json into the process-wide broker (no daemon restart)."""
    broker = get_broker()
    broker.reload_registry(path)
    return broker


def dispatch_pending_tasks(
    handler: Callable[[str], Any],
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Module-level wrapper → ``get_broker().dispatch_pending_tasks``."""
    return get_broker().dispatch_pending_tasks(handler, path=path)
