"""Cascade Router — local Mixture of Agents (MoA), not GPT-4o.

Low-complexity turns stay on a fast local chat model (``qwen2.5-coder:7b``).
High-complexity / visual turns escalate to a **local MoA**:

  1. Vision agent  — Qwen-VL / Llama 3.2 Vision / LLaVA (Ollama) extracts image context
  2. Reasoner agent — DeepSeek (or local fallback) evaluates rules / returns final text

Env overrides:
  DONNA_LOCAL_MODEL       — fast chat model (default qwen2.5-coder:7b)
  DONNA_VISION_MODEL      — preferred vision model (default auto-detect)
  DONNA_REASONER_MODEL    — preferred reasoner (default auto-detect DeepSeek)
  DONNA_CASCADE_EXTERNAL  — set 1 to optionally allow ChatOpenAI (off by default)
  DONNA_CASCADE_MODEL     — external model id if EXTERNAL=1 (legacy)
  DONNA_FORCE_LOCAL       — set 1 to hard-bypass all cloud / Gemini fallbacks
  OLLAMA_URL              — Ollama base URL (default http://127.0.0.1:11434)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

Complexity = Literal["low", "high"]
Backend = Literal["local", "moa", "cascade"]
ComputeMode = Literal["lightweight", "react", "deep_research"]


def force_local_mode() -> bool:
    """True when ``DONNA_FORCE_LOCAL=1`` — skip all cloud / Gemini routing."""
    return (os.environ.get("DONNA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ollama_keep_alive_value() -> int | str:
    try:
        from dana.middleware.idle_monitor import ollama_keep_alive

        return ollama_keep_alive()
    except Exception:  # noqa: BLE001
        return 0

# Ultra-fast System-1 chat (TTFT); graceful fallback when not pulled locally.
_LIGHTWEIGHT_MODEL_CANDIDATES = (
    "llama3.2:1b",
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:1b-instruct",
)

_HIGH_COMPLEXITY_RE = re.compile(
    r"\b("
    r"build\s+a\s+tool|create\s+a\s+tool|code\s+a\s+(?:script|tool)|"
    r"architect|tool\s+forge|forge\s+a\s+tool|"
    r"debug|stack\s*trace|traceback|patch\s+(?:the\s+)?(?:code|source|bug)|"
    r"fix\s+(?:my\s+)?(?:bug|code|crash)|self[- ]?heal|titan\s+repair|"
    r"delegate\s+to\s+cursor|hand\s*off\s+to\s+cursor|implementation\s+plan|"
    r"comprehensive\s+report|deep\s+research|deep\s+dive|"
    r"refactor|rewrite\s+(?:the\s+)?(?:module|file|function)|"
    r"analyze\s+(?:the\s+)?(?:source|codebase|architecture)|"
    r"evaluate\s+(?:the\s+)?slide|slide\s+review|type\s+(?:your\s+)?evaluation|"
    r"self[- ]?improvement|reasoning\s+model|deepseek"
    r")\b",
    re.IGNORECASE,
)

# Hard override — never allow llama3.2 low path for these intents
# (see classify_complexity substring check; regex removed).

_LOW_COMPLEXITY_RE = re.compile(
    r"\b("
    r"what\s+time|what'?s?\s+the\s+time|hello|hi\b|thanks|thank\s+you|"
    r"what(?:'s|\s+is)\s+my\s+name|list\s+(?:the\s+)?todo|"
    r"show\s+pending\s+bugs|standing\s+by"
    r")\b",
    re.IGNORECASE,
)

_VISUAL_TASK_RE = re.compile(
    r"\b("
    r"slide|screen|screenshot|vision|image|photo|picture|"
    r"what(?:'s|\s+is)\s+on\s+(?:my\s+)?screen|capture\s+(?:and\s+analyze\s+)?(?:my\s+)?screen|"
    r"evaluate\s+(?:the\s+)?slide|look\s+at\s+(?:the\s+)?(?:slide|screen)"
    r")\b",
    re.IGNORECASE,
)

# Preferred Ollama tags (first installed match wins).
# Prefer classic LLaVA / Qwen-VL first: some Ollama builds fail to load
# llama3.2-vision (mllama) with "unknown model architecture".
_VISION_CANDIDATES = (
    "llava:7b",
    "llava:latest",
    "llava",
    "qwen2.5vl:latest",
    "qwen2.5-vl:latest",
    "qwen2-vl:latest",
    "bakllava:latest",
    "moondream:latest",
    "llama3.2-vision:latest",
    "llama3.2-vision",
)
_REASONER_CANDIDATES = (
    "deepseek-r1:8b",
    "deepseek-r1:7b",
    "deepseek-r1:latest",
    "deepseek-r1",
    "deepseek-coder-v2:latest",
    "deepseek-coder:latest",
    "deepseek-v2:latest",
    "deepseek-llm:latest",
)

_VISUAL_TOOLS = frozenset(
    {
        "evaluate_slide_and_type",
        "capture_and_analyze_screen",
        "analyze_visual_context",
        "describe_spatial_scene",
    }
)

_ollama_tags_cache: list[str] | None = None
# Models that failed to load on this host (e.g. mllama unsupported by runner).
_vision_blacklist: set[str] = set()

# Deterministic pre-Cascade command dictionary (phrase → target node / mode).
# Longer phrases are preferred on exact hits; fuzzy mailroom covers ASR garble.
STATE_TOGGLE_TRIGGERS: dict[str, str] = {
    "stretch to vision mode": "vision",
    "stretch to vision": "vision",
    "switch to vision mode": "vision",
    "switch to vision": "vision",
    "enter vision mode": "vision",
    "enter vision": "vision",
    "enable vision mode": "vision",
    "enable vision": "vision",
    "go to vision mode": "vision",
    "go to vision": "vision",
    "activate vision mode": "vision",
    "activate vision": "vision",
    "enable camera": "vision",
    "vision mode": "vision",
    # Known Whisper garble aliases (also caught by fuzzy ≥80%).
    "vision mounts": "vision",
    "vision model": "vision",
    "vision modes": "vision",
    "switch to developer mode": "developer",
    "switch to developer": "developer",
    "enter developer mode": "developer",
    "enter developer": "developer",
    "enable developer mode": "developer",
    "go to developer mode": "developer",
    "go to developer": "developer",
    "activate developer mode": "developer",
    "developer mode": "developer",
    "agent mode": "developer",
    "switch to research mode": "research",
    "switch to research": "research",
    "enter research mode": "research",
    "enter research": "research",
    "enable research mode": "research",
    "go to research mode": "research",
    "go to research": "research",
    "activate research mode": "research",
    "research mode": "research",
    "switch to chat mode": "chat",
    "switch to chat": "chat",
    "enter chat mode": "chat",
    "enter chat": "chat",
    "enable chat mode": "chat",
    "go to chat mode": "chat",
    "go to chat": "chat",
    "activate chat mode": "chat",
    "chat mode": "chat",
    # Non-mode system commands (mailroom short-circuits; not Mode Manager modes).
    "status check": "status_check",
    "system status": "status_check",
    "check status": "status_check",
    "donna status": "status_check",
    "mute": "mute",
    "mute audio": "mute",
    "be quiet": "mute",
    "silence": "mute",
}

# Alias kept for Module 2 docs / tests — same mapping as STATE_TOGGLE_TRIGGERS.
COMMAND_DICTIONARY: dict[str, str] = STATE_TOGGLE_TRIGGERS

# Mode Manager targets (set_donna_mode). Other COMMAND_DICTIONARY values are actions.
_MAILROOM_MODE_TARGETS = frozenset({"chat", "developer", "vision", "research"})

# Levenshtein / RapidFuzz ratio threshold (0–100). ≥80% short-circuits the LLM.
FUZZY_MATCH_THRESHOLD = 80.0
# Stage 3.1: long free-form utterances skip fuzzy (exact substring still allowed).
MAILROOM_MAX_WORDS = 8

_STATE_TOGGLE_WAKE_RE = re.compile(
    r"^(?:hey\s+)?donna\b[\s,.\-!:]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MailroomHit:
    """Deterministic mailroom match (exact or fuzzy ≥ threshold)."""

    command: str
    target: str
    score: float
    raw_asr: str
    method: str  # "exact" | "fuzzy"
    residual: str = ""  # text remaining after stripping the matched command


def _normalize_mailroom_asr(query: str) -> str:
    """Strip wake wrappers / punctuation for deterministic matching."""
    blob = re.sub(r"\s+", " ", (query or "").strip().lower())
    blob = _STATE_TOGGLE_WAKE_RE.sub("", blob).strip(" \t.,;:!-")
    blob = re.sub(r"[\"'`]+", "", blob)
    return blob.strip(" \t.,;:!-")


def _mailroom_word_count(query: str) -> int:
    """Word count on the raw utterance (whitespace split)."""
    return len((query or "").split())


def strip_mailroom_residual(raw: str, matched_command: str) -> str:
    """Remove the matched command phrase from ``raw``; return leftover intent text.

    Prefers text *after* the matched phrase (compound mode + fact utterances).
    """
    text = (raw or "").strip()
    phrase = (matched_command or "").strip()
    if not text or not phrase:
        return text
    match = re.search(re.escape(phrase), text, re.IGNORECASE)
    if not match:
        return text
    after = text[match.end() :].strip(" \t.,;:!-")
    if after:
        return re.sub(r"\s+", " ", after).strip(" \t.,;:!-")
    before = text[: match.start()].strip(" \t.,;:!-")
    return re.sub(r"\s+", " ", before).strip(" \t.,;:!-")


def _fuzzy_score(a: str, b: str) -> float:
    """Best RapidFuzz similarity in [0, 100] (ratio / partial / WRatio).

    Short ASR utterances (typical voice commands) may use ``partial_ratio`` so
    garble like ``vision mounts`` still hits ``vision mode``. Longer free-form
    prompts only use full-string scorers so partial hits cannot hijack MoA text.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover
        return 100.0 if a == b else 0.0
    left = (a or "").strip().lower()
    right = (b or "").strip().lower()
    if not left or not right:
        return 0.0
    # ``a`` is ASR blob, ``b`` is dictionary phrase.
    # Prefer ratio / token_set — WRatio alone false-positives free-form prompts
    # that merely share a token with a command (e.g. "Research the latest…"
    # vs "switch to research mode"). partial_ratio is only safe when the ASR
    # blob is about the same length as the dictionary phrase (Whisper garble);
    # on longer free-form text it matches lead tokens like "research …".
    long_utterance = len(left) > max(48, len(right) + 24)
    if long_utterance:
        return float(
            max(fuzz.ratio(left, right), fuzz.token_set_ratio(left, right))
        )
    scores = [
        fuzz.ratio(left, right),
        fuzz.token_set_ratio(left, right),
    ]
    if len(left) <= len(right) + 12:
        scores.append(fuzz.partial_ratio(left, right))
    return float(max(scores))


def fuzzy_match_command(
    query: str,
    *,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> MailroomHit | None:
    """Match raw ASR against ``COMMAND_DICTIONARY`` (exact substring, then fuzzy ≥ threshold)."""
    raw = (query or "").strip()
    blob = _normalize_mailroom_asr(raw)
    if not blob:
        return None

    # 1) Exact substring — prefer longest phrase (legacy behaviour, score=100).
    for phrase, target in sorted(
        COMMAND_DICTIONARY.items(), key=lambda item: -len(item[0])
    ):
        if phrase in blob:
            return MailroomHit(
                command=phrase,
                target=target,
                score=100.0,
                raw_asr=raw,
                method="exact",
                residual=strip_mailroom_residual(raw, phrase),
            )

    # Stage 3.1 length guard: long free-form prompts bypass fuzzy (and mailroom).
    if _mailroom_word_count(raw) > MAILROOM_MAX_WORDS:
        return None

    # 2) Fuzzy Levenshtein / WRatio — immunize against Whisper garble.
    best: MailroomHit | None = None
    for phrase, target in COMMAND_DICTIONARY.items():
        # Reject tiny ASR fragments that only partial-match a longer command
        # (e.g. bare "vision" must not fire "vision mode").
        if len(blob) < int(len(phrase) * 0.75):
            continue
        score = _fuzzy_score(blob, phrase)
        if score < float(threshold):
            continue
        if best is None or score > best.score or (
            score == best.score and len(phrase) > len(best.command)
        ):
            best = MailroomHit(
                command=phrase,
                target=target,
                score=score,
                raw_asr=raw,
                method="fuzzy",
                residual=strip_mailroom_residual(raw, phrase),
            )
    return best


def match_state_toggle(query: str) -> str | None:
    """Return a Mode Manager id when mailroom hits a mode target (exact or fuzzy ≥80%)."""
    hit = fuzzy_match_command(query)
    if hit is None:
        return None
    if hit.target in _MAILROOM_MODE_TARGETS:
        return hit.target
    return None


@dataclass(frozen=True)
class CascadeDecision:
    complexity: Complexity
    backend: Backend
    model: str
    reason: str
    vision_model: str = ""
    reasoner_model: str = ""


def is_cascade_enabled() -> bool:
    try:
        from dana.settings import load_donna_settings

        cfg = load_donna_settings()
        if "enable_cascade_router" in cfg:
            return bool(cfg.get("enable_cascade_router"))
    except Exception:
        pass
    env = os.environ.get("DONNA_CASCADE_ROUTER", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return False


def allow_external_cascade() -> bool:
    """Legacy GPT/OpenAI path — off unless explicitly opted in."""
    if force_local_mode():
        return False
    return os.environ.get("DONNA_CASCADE_EXTERNAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def local_model_name() -> str:
    return (
        os.environ.get("DONNA_LOCAL_MODEL", "").strip()
        or os.environ.get("OLLAMA_MODEL", "").strip()
        or "qwen2.5-coder:7b"
    )


def _ollama_tag_available(candidate: str, tags: list[str]) -> bool:
    """True when ``candidate`` matches an installed Ollama tag (prefix-tolerant)."""
    want = (candidate or "").strip().lower()
    if not want:
        return False
    want_bare = want.split(":")[0]
    for tag in tags:
        t = (tag or "").strip().lower()
        if not t:
            continue
        if t == want or t.startswith(want + "-") or t.startswith(want + ":"):
            return True
        # Allow installed ``llama3.2:1b-instruct-q4_K_M`` for candidate ``llama3.2:1b``.
        if want.count(":") == 1:
            name, ver = want.split(":", 1)
            if t.startswith(f"{name}:{ver}"):
                return True
        if ":" not in want and t.split(":")[0] == want_bare:
            return True
    return False


def lightweight_model_name() -> str:
    """Prefer ``llama3.2:1b`` for System-1 TTFT; fall back to ``qwen2.5-coder:7b``.

    Override with ``DONNA_LIGHTWEIGHT_MODEL``. When the 1b tag is not installed,
    returns the standard local chat model so inference never hard-fails.
    """
    override = (os.environ.get("DONNA_LIGHTWEIGHT_MODEL") or "").strip()
    if override:
        return override
    tags = _list_ollama_tags()
    if tags:
        for cand in _LIGHTWEIGHT_MODEL_CANDIDATES:
            if _ollama_tag_available(cand, tags):
                # Prefer the exact installed tag when a longer variant matched.
                want = cand.lower()
                for tag in tags:
                    t = (tag or "").strip()
                    if t.lower() == want or t.lower().startswith(want):
                        return t
                return cand
        _log_cascade(
            "lightweight model llama3.2:1b not in Ollama library — "
            f"falling back to {local_model_name()}"
        )
    else:
        # Tags unreachable: do not risk a missing 1b hard-fail — use standard local.
        _log_cascade(
            "Ollama tags unavailable — lightweight falling back to "
            f"{local_model_name()}"
        )
    return local_model_name()


def deep_research_model_name() -> str:
    """Heaviest local reasoner for swarm / deep-research MoA stage-1."""
    return reasoner_model_name()


def resolve_compute_mode(
    query: str = "",
    *,
    forced_tool: str | None = None,
    use_lightweight: bool = False,
) -> ComputeMode:
    """Map turn intent to proportional compute tier."""
    if use_lightweight:
        return "lightweight"
    tool = (forced_tool or "").strip()
    text = (query or "").strip()
    if tool == "dispatch_research_swarm" or text.lstrip().startswith(
        "[BACKGROUND TASK]"
    ):
        return "deep_research"
    if tool:
        return "react"
    if classify_complexity(text, forced_tool=tool or None) == "high":
        return "deep_research"
    return "react"


def cascade_model_name() -> str:
    """Legacy external model id (only used when DONNA_CASCADE_EXTERNAL=1)."""
    return os.environ.get("DONNA_CASCADE_MODEL", "").strip() or "gpt-4o-mini"


def _list_ollama_tags(*, force: bool = False) -> list[str]:
    global _ollama_tags_cache
    if _ollama_tags_cache is not None and not force:
        return _ollama_tags_cache
    try:
        req = urllib.request.Request(f"{ollama_base_url()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [str(m.get("name") or "") for m in data.get("models") or [] if m.get("name")]
        _ollama_tags_cache = names
        return names
    except Exception:
        return list(_ollama_tags_cache or [])


def _blacklist_vision(model: str, *, reason: str = "") -> None:
    name = (model or "").strip()
    if not name:
        return
    _vision_blacklist.add(name.lower())
    bare = name.lower().split(":")[0]
    _vision_blacklist.add(bare)
    if reason:
        _log_cascade(f"MoA vision blacklist {name}: {reason}", level="warning")


def _is_vision_blacklisted(model: str) -> bool:
    n = (model or "").strip().lower()
    if not n:
        return False
    return n in _vision_blacklist or n.split(":")[0] in _vision_blacklist


def _pick_installed(preferred: str, candidates: tuple[str, ...], *, fallback: str) -> str:
    tags = _list_ollama_tags()
    tags_l = {t.lower(): t for t in tags}

    def _resolve(name: str) -> str | None:
        n = (name or "").strip()
        if not n:
            return None
        if _is_vision_blacklisted(n):
            return None
        if n.lower() in tags_l:
            return tags_l[n.lower()]
        # Accept bare name matching prefix (llama3.2-vision → llama3.2-vision:11b).
        bare = n.lower().split(":")[0]
        for full, orig in tags_l.items():
            if _is_vision_blacklisted(full):
                continue
            if full == bare or full.startswith(bare + ":"):
                return orig
        return None

    if preferred:
        hit = _resolve(preferred)
        if hit:
            return hit
        # User forced a tag that isn't installed yet — still return it so Ollama
        # can pull / surface a clear error; callers may fall back.
        if not _is_vision_blacklisted(preferred):
            return preferred

    for cand in candidates:
        hit = _resolve(cand)
        if hit:
            return hit
    return fallback


def vision_model_name() -> str:
    preferred = os.environ.get("DONNA_VISION_MODEL", "").strip()
    # Empty preferred → walk candidates (llava first). Env can still force
    # llama3.2-vision when the local Ollama runner supports mllama.
    return _pick_installed(preferred, _VISION_CANDIDATES, fallback=preferred or "llava")


def reasoner_model_name() -> str:
    preferred = (
        os.environ.get("DONNA_REASONER_MODEL", "").strip() or "deepseek-r1:8b"
    )
    return _pick_installed(
        preferred,
        _REASONER_CANDIDATES,
        fallback=preferred,
    )


def is_visual_task(query: str = "", *, forced_tool: str | None = None) -> bool:
    if forced_tool and forced_tool in _VISUAL_TOOLS:
        return True
    return bool(_VISUAL_TASK_RE.search(query or ""))


def classify_complexity(query: str, *, forced_tool: str | None = None) -> Complexity:
    """Heuristic cognitive classifier for MoA vs local routing."""
    user_input = query
    print(f"\n[DEBUG ROUTER] Raw text received for classification: '{user_input}'")
    text_lower = (user_input or "").lower()
    force_high_keywords = [
        "self-improvement",
        "deepseek",
        "draft cursor prompt",
        "draft_cursor_prompt",
        "cursor handling",
        "complex query",
        "complex query patterns",
    ]
    # Absolute first: substring force-high → DeepSeek MoA (bypasses all heuristics).
    if any(keyword in text_lower for keyword in force_high_keywords):
        return "high"
    text = (user_input or "").strip()
    high_tools = {
        "architect_new_tool",
        "publish_tool_to_general",
        "dispatch_titan_repair",
        "delegate_to_cursor",
        "draft_cursor_prompt",
        "dispatch_research_swarm",
        "capture_and_analyze_screen",
        "read_system_architecture",
        "evaluate_slide_and_type",
    }
    if forced_tool and forced_tool in high_tools:
        return "high"
    if _HIGH_COMPLEXITY_RE.search(text):
        return "high"
    if _LOW_COMPLEXITY_RE.search(text):
        return "low"
    return "low"


def _donna_mode_is_chat() -> bool:
    """True when Mode Manager is in chat (bypass MoA / high-complexity escalate)."""
    try:
        from dana.agentic import get_donna_mode

        return get_donna_mode() == "chat"
    except Exception:  # noqa: BLE001
        return False


def allows_react_task_jail() -> bool:
    """False in chat mode — task_queue / ReAct jail must not accept tool prompts.

    ``vision`` / ``research`` modes remain scaffolded and still allow the jail
    until their dedicated graphs are wired.
    """
    return not _donna_mode_is_chat()


def decide_route(
    query: str,
    *,
    forced_tool: str | None = None,
    default_model: str | None = None,
) -> CascadeDecision:
    local = default_model or local_model_name()

    # Stage 8.5 — Dictation Loop short-circuit (keyword or GUI latch).
    try:
        from dana.management.dictation import should_handle_dictation

        if should_handle_dictation(query or ""):
            _log_cascade(
                "dictation keyword/mode -> Dictation_Handler "
                "(OCR-paired session log; LLM bypass)"
            )
            try:
                from dana.telemetry import log_router

                log_router(
                    "dictation route",
                    current_agent="Dictation_Handler",
                    active_intent="dictate",
                    payload={"query_preview": (query or "")[:160]},
                )
            except Exception:  # noqa: BLE001
                pass
            return CascadeDecision(
                complexity="low",
                backend="local",
                model=local,
                reason="dictation mode -> Dictation_Handler (OCR session log)",
            )
    except Exception:  # noqa: BLE001
        pass

    # Module 2 mailroom — absolute top of pipeline (before chat bypass / MoA / LLM).
    mailroom = fuzzy_match_command(query or "")
    if mailroom is not None:
        try:
            from dana.telemetry import log_router

            log_router(
                f"mailroom {mailroom.method} -> {mailroom.target}",
                current_agent="Mailroom",
                active_intent=mailroom.target,
                payload={
                    "raw_asr": mailroom.raw_asr,
                    "matched_command": mailroom.command,
                    "confidence": round(float(mailroom.score), 2),
                    "target_node": mailroom.target,
                    "method": mailroom.method,
                    "threshold": FUZZY_MATCH_THRESHOLD,
                    "residual": mailroom.residual or "",
                },
            )
        except Exception:  # noqa: BLE001
            pass

        target = mailroom.target
        if target in _MAILROOM_MODE_TARGETS:
            try:
                from dana.agentic import set_donna_mode

                set_donna_mode(target)
            except Exception:  # noqa: BLE001
                pass
            # Stage 3.1: residual clause rides in Handoff.intent_context.
            try:
                from dana.handoff import execute_handoff
                from dana.schema import Handoff

                _mode_to_agent = {
                    "vision": "Vision_Agent",
                    "chat": "Chat_Node",
                    "developer": "ReAct_Agent",
                    "research": "MoA_Reasoner",
                }
                residual = (mailroom.residual or "").strip()
                intent_ctx = residual or (mailroom.raw_asr or query or "").strip() or target
                execute_handoff(
                    Handoff(
                        target_agent=_mode_to_agent.get(target, "ReAct_Agent"),
                        reason=f"mailroom {mailroom.method} → {target}",
                        intent_context=intent_ctx[:500],
                    ),
                    session_id="",
                    current_agent="Mailroom",
                )
            except Exception:  # noqa: BLE001
                pass
            _log_cascade(
                f"mailroom {mailroom.method} "
                f"score={mailroom.score:.1f} '{mailroom.command}' "
                f"-> mode={target} (LLM / chat-node bypass)"
                + (
                    f" residual={mailroom.residual!r}"
                    if (mailroom.residual or "").strip()
                    else ""
                )
            )
            if target == "vision":
                vision = vision_model_name()
                reasoner = reasoner_model_name()
                return CascadeDecision(
                    complexity="high",
                    backend="moa",
                    model=f"moa:{vision}+{reasoner}",
                    reason=(
                        f"mailroom {mailroom.method} "
                        f"({mailroom.score:.0f}%) -> vision MoA pipeline (LLM bypass)"
                    ),
                    vision_model=vision,
                    reasoner_model=reasoner,
                )
            if target == "chat":
                return CascadeDecision(
                    complexity="low",
                    backend="local",
                    model=local,
                    reason=(
                        f"mailroom {mailroom.method} "
                        f"({mailroom.score:.0f}%) -> chat mode local"
                    ),
                )
            # developer / research: mode already set; continue with tool heuristics.
        elif target == "status_check":
            _log_cascade(
                f"mailroom {mailroom.method} score={mailroom.score:.1f} "
                f"-> status_check (LLM bypass)"
            )
            return CascadeDecision(
                complexity="low",
                backend="local",
                model=local,
                reason=(
                    f"mailroom {mailroom.method} "
                    f"({mailroom.score:.0f}%) -> status_check"
                ),
            )
        elif target == "mute":
            _log_cascade(
                f"mailroom {mailroom.method} score={mailroom.score:.1f} "
                f"-> mute (LLM bypass)"
            )
            return CascadeDecision(
                complexity="low",
                backend="local",
                model=local,
                reason=(
                    f"mailroom {mailroom.method} "
                    f"({mailroom.score:.0f}%) -> mute"
                ),
            )
    else:
        # Fall-through: forensic ASR breadcrumb before semantic / LLM routing.
        try:
            from dana.telemetry import log_voice_asr

            log_voice_asr(
                query or "",
                payload={"mailroom": "fallthrough", "threshold": FUZZY_MATCH_THRESHOLD},
            )
        except Exception:  # noqa: BLE001
            pass

    # Chat mode normally stays on lightweight llama — BUT system/file/code
    # intents must escalate to the tool-enabled MoA / ReAct path.
    try:
        from dana.agentic import requires_tool_graph

        tool_intent = requires_tool_graph(query or "")
    except Exception:  # noqa: BLE001
        tool_intent = False

    # Deep research / idle BACKGROUND TASKS must never stay on chat-local bypass.
    _deep_force = (forced_tool or "").strip() == "dispatch_research_swarm" or (
        query or ""
    ).lstrip().startswith("[BACKGROUND TASK]")
    if _donna_mode_is_chat() and not tool_intent and not _deep_force:
        return CascadeDecision(
            complexity="low",
            backend="local",
            model=local,
            reason="chat mode → local llama, tools/MoA bypassed",
        )
    if _donna_mode_is_chat() and (tool_intent or _deep_force):
        _log_cascade(
            "CRITICAL: system/file/code/research intent in chat mode → tool-graph "
            "(MoA/ReAct); lightweight chat bypassed"
        )
    # Scaffolded modes: log intent; keep current MoA/local heuristics for now.
    try:
        from dana.agentic import get_donna_mode

        mode = get_donna_mode()
        if mode in {"vision", "research"}:
            _log_cascade(
                f"mode={mode} (scaffolded) — using standard cascade heuristics"
            )
    except Exception:  # noqa: BLE001
        pass

    complexity = classify_complexity(query, forced_tool=forced_tool)
    vision = vision_model_name()
    reasoner = reasoner_model_name()

    if complexity == "high" and is_cascade_enabled():
        if is_visual_task(query, forced_tool=forced_tool):
            return CascadeDecision(
                complexity="high",
                backend="moa",
                model=f"moa:{vision}+{reasoner}",
                reason="high-cognitive visual → local MoA (vision→reasoner)",
                vision_model=vision,
                reasoner_model=reasoner,
            )
        # Non-visual high load: reasoner MoA stage (DeepSeek) without vision.
        return CascadeDecision(
            complexity="high",
            backend="moa",
            model=reasoner,
            reason="high-complexity → local MoA reasoner (DeepSeek/local)",
            vision_model="",
            reasoner_model=reasoner,
        )

    if complexity == "high" and not is_cascade_enabled():
        return CascadeDecision(
            complexity="high",
            backend="local",
            model=local,
            reason="high-complexity but cascade disabled — staying on local Ollama",
        )
    return CascadeDecision(
        complexity="low",
        backend="local",
        model=local,
        reason="low-complexity → local llama",
    )


def note_high_complexity_deepseek_latency(
    latency_ms: float,
    *,
    model: str = "",
) -> None:
    """Push DeepSeek high-complexity latency into live ``dashboard.md`` telemetry."""
    mid = (model or "").strip()
    if "deepseek" not in mid.lower():
        return
    try:
        from dana.telemetry import cascade_latency_threshold_ms, note_cascade_latency

        note_cascade_latency(latency_ms, model=mid)
        thr = cascade_latency_threshold_ms()
        flag = " OVER THRESHOLD" if float(latency_ms) >= thr else ""
        _log_cascade(
            f"DeepSeek latency={float(latency_ms):.0f}ms "
            f"threshold={thr:.0f}ms{flag} model={mid}"
        )
    except Exception:  # noqa: BLE001
        pass


def _log_cascade(msg: str, *, level: str = "info") -> None:
    try:
        from dana.logging import log as _log

        _log("Cascade", msg, level=level)
    except Exception:
        pass


def _http_error_detail(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        body = (body or "").strip()
        if body:
            return f"HTTP Error {exc.code}: {body[:500]}"
        return f"HTTP Error {exc.code}: {exc.reason}"
    return str(exc)


def _downscale_png_for_vision(png_bytes: bytes, *, max_side: int = 1024) -> bytes:
    """Shrink captures so vision models fit VRAM / avoid runner OOMs."""
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        img.load()
        w, h = img.size
        if max(w, h) <= max_side:
            return png_bytes
        img.thumbnail((max_side, max_side))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


def _ollama_generate(
    *,
    model: str,
    prompt: str,
    images_b64: list[str] | None = None,
    timeout: float = 90.0,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": _ollama_keep_alive_value(),
        "options": {"num_predict": 512, "num_ctx": 4096},
    }
    if images_b64:
        payload["images"] = images_b64
    req = urllib.request.Request(
        f"{ollama_base_url()}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_http_error_detail(exc)) from exc
    return str(data.get("response") or "").strip()


def _ollama_chat(
    *,
    model: str,
    system: str,
    user: str,
    timeout: float = 90.0,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": _ollama_keep_alive_value(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{ollama_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("message") or {}
    return str(msg.get("content") or data.get("response") or "").strip()


def _ollama_chat_vision(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: float = 180.0,
) -> str:
    """Ollama standard multimodal chat payload (images on the user message)."""
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": _ollama_keep_alive_value(),
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "options": {"num_predict": 512, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        f"{ollama_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_http_error_detail(exc)) from exc
    msg = data.get("message") or {}
    return str(msg.get("content") or data.get("response") or "").strip()


def _ensure_llava_installed() -> str | None:
    """If no LLaVA tag exists, silently ``ollama pull llava:7b`` and return the tag."""
    tags = _list_ollama_tags(force=True)
    for cand in ("llava:7b", "llava:latest", "llava"):
        hit = None
        tags_l = {t.lower(): t for t in tags}
        if cand.lower() in tags_l:
            hit = tags_l[cand.lower()]
        else:
            bare = cand.lower().split(":")[0]
            for full, orig in tags_l.items():
                if full == bare or full.startswith(bare + ":"):
                    hit = orig
                    break
        if hit:
            return hit

    _log_cascade("MoA: llava missing — running `ollama pull llava:7b`", level="warning")
    try:
        import subprocess

        proc = subprocess.run(
            ["ollama", "pull", "llava:7b"],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        if proc.returncode != 0:
            _log_cascade(
                f"MoA: ollama pull llava:7b failed rc={proc.returncode} "
                f"err={(proc.stderr or '')[:200]}",
                level="warning",
            )
            return None
    except Exception as exc:  # noqa: BLE001
        _log_cascade(f"MoA: ollama pull llava failed: {exc}", level="warning")
        return None

    global _ollama_tags_cache
    _ollama_tags_cache = None
    tags = _list_ollama_tags(force=True)
    tags_l = {t.lower(): t for t in tags}
    for cand in ("llava:7b", "llava:latest", "llava"):
        if cand.lower() in tags_l:
            return tags_l[cand.lower()]
        bare = cand.lower().split(":")[0]
        for full, orig in tags_l.items():
            if full == bare or full.startswith(bare + ":"):
                return orig
    return None


def extract_vision_context(
    png_bytes: bytes,
    *,
    prompt: str = "",
    model: str | None = None,
) -> str:
    """MoA stage 1 — local vision model extracts readable context from an image.

    Uses Ollama multimodal chat format; on mllama/format rejection falls back to
    LLaVA (auto-pulls ``llava:7b`` if missing). Raw base64 never leaves this stage.
    """
    vision = (model or vision_model_name()).strip()
    ask = (
        prompt
        or "Extract all readable text from this image. Note the title/heading if any, "
        "list body text, and estimate word count. Be literal and complete."
    ).strip()
    compact = _downscale_png_for_vision(png_bytes, max_side=1024)
    b64 = base64.b64encode(compact).decode("ascii")
    _log_cascade(
        f"MoA vision extract model={vision} image_bytes={len(compact)} "
        f"(from {len(png_bytes)})"
    )
    tried: list[str] = []
    errors: list[str] = []

    def _is_format_reject(detail: str) -> bool:
        d = (detail or "").lower()
        return any(
            tok in d
            for tok in (
                "mllama",
                "unknown model architecture",
                "does not support images",
                "invalid",
                "unsupported",
                "failed to load",
                "500",
            )
        )

    def _try_model(name: str) -> str:
        tried.append(name)
        # Prefer standard multimodal chat payload.
        try:
            text = _ollama_chat_vision(
                model=name, prompt=ask, image_b64=b64, timeout=180.0
            )
            if text:
                return text
        except Exception as chat_exc:  # noqa: BLE001
            detail_chat = str(chat_exc)
            if _is_format_reject(detail_chat):
                # mllama / architecture rejects will also fail /api/generate —
                # skip straight to the LLaVA fallback chain.
                raise
            _log_cascade(
                f"MoA vision chat failed ({name}): {chat_exc}; trying /api/generate",
                level="warning",
            )
        return _ollama_generate(
            model=name, prompt=ask, images_b64=[b64], timeout=180.0
        )

    def _fallback_chain(*, skip_mllama: bool) -> str:
        # Prefer installed LLaVA; pull if absent.
        llava = _ensure_llava_installed()
        order: list[str] = []
        if llava:
            order.append(llava)
        for cand in _VISION_CANDIDATES:
            bare = cand.split(":")[0].lower()
            if skip_mllama and "llama3.2-vision" in bare:
                continue
            if _is_vision_blacklisted(cand):
                continue
            alt = _pick_installed("", (cand,), fallback="")
            if alt and alt not in order and alt not in tried:
                order.append(alt)
        for alt in order:
            if alt in tried:
                continue
            try:
                _log_cascade(f"MoA vision retry model={alt}")
                text = _try_model(alt)
                if text:
                    return text
            except Exception as exc2:  # noqa: BLE001
                errors.append(f"{alt}: {exc2}")
                detail2 = str(exc2)
                if _is_format_reject(detail2):
                    _blacklist_vision(alt, reason=detail2[:160])
                continue
        return ""

    try:
        text = _try_model(vision)
        if text:
            # Drop b64 from local scope ASAP (reasoner never sees it).
            del b64
            return text
    except Exception as exc:  # noqa: BLE001
        detail = _http_error_detail(exc) if not str(exc) else str(exc)
        errors.append(f"{vision}: {detail}")
        _log_cascade(f"MoA vision failed ({vision}): {detail}", level="warning")
        skip_mllama = _is_format_reject(detail)
        if skip_mllama:
            _blacklist_vision(vision, reason=detail[:160])
        text = _fallback_chain(skip_mllama=skip_mllama)
        if text:
            del b64
            return text

    # If preferred model returned empty, still try LLaVA chain.
    if not errors:
        text = _fallback_chain(skip_mllama=True)
        if text:
            try:
                del b64
            except Exception:
                pass
            return text

    try:
        del b64
    except Exception:
        pass

    # Structural fallback when no vision model is available.
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        err_snip = ("; ".join(errors))[:240] if errors else "no response"
        return (
            f"[vision unavailable — structural fallback] Screen capture {w}x{h} PNG "
            f"({len(png_bytes)} bytes). Tried={tried or [vision]}. Detail={err_snip}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"[vision unavailable] capture {len(png_bytes)} bytes; error={exc}"


def reason_over_context(
    context: str,
    *,
    rule: str = "",
    task: str = "",
    model: str | None = None,
) -> str:
    """MoA stage 2 — local reasoner (DeepSeek) evaluates context vs rule/task."""
    reasoner = (model or reasoner_model_name()).strip()
    system = (
        "You are Donna's local MoA reasoner. Be precise and concise. "
        "When given a RULE, decide PASS/FAIL and produce a short actionable COMMENT. "
        "Use this exact format when a RULE is present:\n"
        "VERDICT: PASS|FAIL|UNCLEAR\n"
        "WORD_COUNT: <integer or -1>\n"
        "COMMENT: <one sentence, max 200 chars>\n"
        "If no RULE is given, answer the TASK directly in plain text."
    )
    user_parts = []
    if task:
        user_parts.append(f"TASK:\n{task.strip()}")
    if rule:
        user_parts.append(f"RULE:\n{rule.strip()}")
    user_parts.append(f"CONTEXT (from vision / prior stage):\n{(context or '')[:4000]}")
    user = "\n\n".join(user_parts)
    _log_cascade(f"MoA reasoner model={reasoner}")
    t0 = time.perf_counter()
    try:
        out = _ollama_chat(model=reasoner, system=system, user=user, timeout=120.0)
        note_high_complexity_deepseek_latency(
            (time.perf_counter() - t0) * 1000.0,
            model=reasoner,
        )
        return out
    except Exception as exc:  # noqa: BLE001
        note_high_complexity_deepseek_latency(
            (time.perf_counter() - t0) * 1000.0,
            model=reasoner,
        )
        _log_cascade(f"MoA reasoner failed ({reasoner}): {exc}", level="warning")
        # Fall back to local fast model.
        fallback = local_model_name()
        if fallback != reasoner:
            try:
                _log_cascade(f"MoA reasoner fallback model={fallback}")
                return _ollama_chat(
                    model=fallback, system=system, user=user, timeout=90.0
                )
            except Exception as exc2:  # noqa: BLE001
                return f"UNCLEAR: reasoner unavailable ({exc2})"
        return f"UNCLEAR: reasoner unavailable ({exc})"


def run_visual_moa(
    png_bytes: bytes,
    *,
    rule: str = "",
    task: str = "",
    vision_prompt: str = "",
) -> dict[str, Any]:
    """Full local MoA for high-cognitive visual tasks.

    Vision model extracts image text → reasoner evaluates rule → final string.
    """
    decision = decide_route(
        task or rule or "evaluate visual",
        forced_tool="evaluate_slide_and_type",
    )
    vision_text = extract_vision_context(png_bytes, prompt=vision_prompt)
    final = reason_over_context(
        vision_text,
        rule=rule,
        task=task
        or (
            "Evaluate the slide/context against the RULE and produce VERDICT/WORD_COUNT/COMMENT."
            if rule
            else "Summarize the visual context."
        ),
        model=decision.reasoner_model or None,
    )
    return {
        "vision_text": vision_text,
        "final": final,
        "route": f"moa/{decision.vision_model or vision_model_name()}+{decision.reasoner_model or reasoner_model_name()}",
        "vision_model": decision.vision_model or vision_model_name(),
        "reasoner_model": decision.reasoner_model or reasoner_model_name(),
        "decision": decision,
    }


def resolve_chat_model(
    *,
    query: str = "",
    forced_tool: str | None = None,
    default_model: str | None = None,
    temperature: float = 0.2,
    mode: ComputeMode | None = None,
) -> Any:
    """Return a LangChain chat model for the chosen Cascade / MoA route.

    Visual MoA (image bytes) should call ``run_visual_moa`` directly.
    This helper binds the **reasoner** (or local fast model) for text ReAct turns.

    ``mode`` proportional tiers:
      - ``lightweight`` — ``llama3.2:1b`` (fallback ``qwen2.5-coder:7b``) System-1 chat
      - ``react`` — standard local model for ``bind_tools``
      - ``deep_research`` — MoA reasoner plans; formatter stays local for tools
    """
    compute_mode: ComputeMode = mode or resolve_compute_mode(
        query,
        forced_tool=forced_tool,
        use_lightweight=False,
    )

    # System-1: skip MoA / complexity escalate entirely.
    if compute_mode == "lightweight":
        model_id = default_model or lightweight_model_name()
        # If 1b was chosen optimistically but tags were empty, keep fallback ready.
        if not default_model and model_id.startswith("llama3.2:1b"):
            tags = _list_ollama_tags()
            if tags and not _ollama_tag_available(model_id, tags):
                model_id = local_model_name()
        _log_cascade(
            f"route=local complexity=low mode=lightweight model={model_id} "
            "(System-1 proportional compute)"
        )
        reasoner_id = ""
        decision = None
    else:
        decision = decide_route(
            query, forced_tool=forced_tool, default_model=default_model
        )
        if compute_mode == "deep_research" and decision.complexity != "high":
            # Force heavy reasoner foresight for swarm / background research.
            decision = decide_route(
                query,
                forced_tool=forced_tool or "dispatch_research_swarm",
                default_model=default_model,
            )
        _log_cascade(
            f"route={decision.backend} complexity={decision.complexity} "
            f"mode={compute_mode} model={decision.model} ({decision.reason})"
        )

        # Optional legacy external path (explicit opt-in only).
        if (
            decision.backend in ("cascade", "moa")
            and allow_external_cascade()
            and decision.backend == "cascade"
        ):
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(model=cascade_model_name(), temperature=temperature)
            except Exception as exc:  # noqa: BLE001
                _log_cascade(
                    f"WARNING: external Cascade unavailable ({exc}); "
                    "using local MoA reasoner",
                    level="warning",
                )

        # ReAct / tool-calling MUST stay on the fast local chat model.
        # DeepSeek-R1 does not emit reliable Ollama native tool_calls under bind_tools.
        # High-complexity MoA turns use the two-stage shim in agentic_react_graph:
        #   stage1 reasoner (no tools) → stage2 local formatter (bind_tools).
        reasoner_id = decision.reasoner_model or reasoner_model_name()
        if compute_mode == "deep_research":
            reasoner_id = deep_research_model_name()
        if decision.backend == "local" and compute_mode == "react":
            model_id = decision.model or local_model_name()
        else:
            # moa / deep_research text ReAct path → local tool-caller (formatter)
            model_id = local_model_name()
            if (
                decision.complexity == "high"
                or decision.backend == "moa"
                or compute_mode == "deep_research"
            ):
                _log_cascade(
                    f"ReAct tool-loop local={model_id} "
                    f"(MoA shim: reasoner={reasoner_id} plans, "
                    f"formatter={model_id} bind_tools; mode={compute_mode})"
                )

        # Opt-in escape hatch for experiments only (expect broken native tool_calls).
        if (os.environ.get("DONNA_REACT_USE_REASONER") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            model_id = reasoner_id or deep_research_model_name()
            _log_cascade(
                f"WARNING: DONNA_REACT_USE_REASONER → ChatOllama={model_id}",
                level="warning",
            )

    # Lightweight System-1 (llama3.2:1b) stays capped at 4096 to avoid RAM bloat.
    # ReAct / deep_research keep the larger env-tunable window (default 8192).
    if compute_mode == "lightweight":
        num_ctx = 4096
        try:
            num_ctx = max(
                2048,
                min(
                    4096,
                    int(os.environ.get("DONNA_LIGHTWEIGHT_NUM_CTX", "4096") or "4096"),
                ),
            )
        except ValueError:
            num_ctx = 4096
    else:
        num_ctx = 8192
        try:
            num_ctx = max(
                4096,
                int(os.environ.get("DONNA_REACT_NUM_CTX", "8192") or "8192"),
            )
        except ValueError:
            num_ctx = 8192

    # Tool-call JSON (large args / multi-arg schemas) must finish before the
    # generation ceiling — 512 truncates mid-JSON and crashes llama-server.
    num_predict = 4096
    try:
        num_predict = max(
            512,
            int(os.environ.get("DONNA_REACT_NUM_PREDICT", "4096") or "4096"),
        )
    except ValueError:
        num_predict = 4096

    # Thermal guardrail / zero-latency unload: keep_alive from idle monitor
    # (default 0). Never pass -1 / "-1" — Ollama returns HTTP 400 on those values.
    try:
        from dana.middleware.idle_monitor import ollama_keep_alive

        keep_alive = ollama_keep_alive()
    except Exception:  # noqa: BLE001
        keep_alive = 0

    from dana.llm_client import build_chat_ollama, draft_num_predict

    _draft_n = draft_num_predict()
    _log_cascade(
        f"ChatOllama model={model_id} num_ctx={num_ctx} "
        f"num_predict={num_predict} keep_alive={keep_alive!r}"
        + (f" draft_num_predict={_draft_n}" if _draft_n is not None else "")
    )
    # Leave format unset: native tool grammar comes from bind_tools(tools),
    # not format="json" (which would constrain spoken FINAL answers to JSON).
    # Speculative drafting: Modelfile DRAFT / llama.cpp -md; see dana.llm_client.
    return build_chat_ollama(
        model=model_id,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive=keep_alive,
    )
