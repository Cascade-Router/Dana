"""Cascade Router — local Mixture of Agents (MoA), not GPT-4o.

Low-complexity turns stay on a fast local chat model (``llama3.2``).
High-complexity / visual turns escalate to a **local MoA**:

  1. Vision agent  — Qwen-VL / Llama 3.2 Vision / LLaVA (Ollama) extracts image context
  2. Reasoner agent — DeepSeek (or local fallback) evaluates rules / returns final text

Env overrides:
  DONNA_LOCAL_MODEL       — fast chat model (default llama3.2)
  DONNA_VISION_MODEL      — preferred vision model (default auto-detect)
  DONNA_REASONER_MODEL    — preferred reasoner (default auto-detect DeepSeek)
  DONNA_CASCADE_EXTERNAL  — set 1 to optionally allow ChatOpenAI (off by default)
  DONNA_CASCADE_MODEL     — external model id if EXTERNAL=1 (legacy)
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
        from donna.settings import load_donna_settings

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
        or "llama3.2"
    )


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
        from donna.agentic import get_donna_mode

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
    # Chat mode normally stays on lightweight llama — BUT system/file/code
    # intents must escalate to the tool-enabled MoA / ReAct path.
    try:
        from donna.agentic import requires_tool_graph

        tool_intent = requires_tool_graph(query or "")
    except Exception:  # noqa: BLE001
        tool_intent = False

    if _donna_mode_is_chat() and not tool_intent:
        return CascadeDecision(
            complexity="low",
            backend="local",
            model=local,
            reason="chat mode → local llama, tools/MoA bypassed",
        )
    if _donna_mode_is_chat() and tool_intent:
        _log_cascade(
            "CRITICAL: system/file/code intent in chat mode → tool-graph "
            "(MoA/ReAct); lightweight chat bypassed"
        )
    # Scaffolded modes: log intent; keep current MoA/local heuristics for now.
    try:
        from donna.agentic import get_donna_mode

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
        from donna.telemetry import cascade_latency_threshold_ms, note_cascade_latency

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
        from donna.logging import log as _log

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
) -> Any:
    """Return a LangChain chat model for the chosen Cascade / MoA route.

    Visual MoA (image bytes) should call ``run_visual_moa`` directly.
    This helper binds the **reasoner** (or local fast model) for text ReAct turns.
    Chat mode always resolves to the local fast model (no MoA escalate).
    """
    decision = decide_route(
        query, forced_tool=forced_tool, default_model=default_model
    )
    _log_cascade(
        f"route={decision.backend} complexity={decision.complexity} "
        f"model={decision.model} ({decision.reason})"
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
                f"WARNING: external Cascade unavailable ({exc}); using local MoA reasoner",
                level="warning",
            )

    # ReAct / tool-calling MUST stay on the fast local chat model.
    # DeepSeek-R1 (MoA reasoner) does not emit Ollama native tool_calls when
    # bind_tools is used — it returns empty content/tool_calls or prose tickets.
    # Reasoner stays reserved for ``run_visual_moa`` / ``reason_over_context``.
    reasoner_id = decision.reasoner_model or reasoner_model_name()
    if decision.backend == "local":
        model_id = decision.model or local_model_name()
    else:
        # moa / cascade text ReAct path → local tool-caller
        model_id = local_model_name()
        if decision.complexity == "high" or decision.backend == "moa":
            _log_cascade(
                f"ReAct tool-loop local={model_id} "
                f"(reasoner={reasoner_id} reserved for MoA stages; "
                f"R1 bind_tools is non-functional on Ollama)"
            )

    # Opt-in escape hatch for experiments only (expect broken native tool_calls).
    if (os.environ.get("DONNA_REACT_USE_REASONER") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        model_id = reasoner_id
        _log_cascade(
            f"WARNING: DONNA_REACT_USE_REASONER → ChatOllama={model_id}",
            level="warning",
        )

    num_ctx = 8192
    try:
        num_ctx = max(
            4096,
            int(os.environ.get("DONNA_REACT_NUM_CTX", "8192") or "8192"),
        )
    except ValueError:
        num_ctx = 8192

    from langchain_ollama import ChatOllama

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

    _log_cascade(
        f"ChatOllama model={model_id} num_ctx={num_ctx} num_predict={num_predict}"
    )
    return ChatOllama(
        model=model_id,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
