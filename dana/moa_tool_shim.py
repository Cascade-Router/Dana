"""Two-stage MoA tool-binding shim (DeepSeek-R1 reason → Llama tool-call).

DeepSeek-R1 on Ollama does not emit reliable native ``tool_calls`` under
``bind_tools``. High-complexity turns therefore:

1. **Reasoner stage** — ``deepseek-r1`` (no tools) produces a textual plan.
2. **Formatter stage** — ``llama3.2`` with ``bind_tools`` converts that plan
   into a strict native tool call without inventing/truncating args.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dana.cascade_router import (
    decide_route,
    local_model_name,
    note_high_complexity_deepseek_latency,
    reasoner_model_name,
)

# Tools that must not be force-executed from thin broker args on MoA routes —
# the reasoner plan should drive complete arguments (esp. draft_cursor_prompt).
_MOA_DEFER_FORCE_TOOLS = frozenset(
    {
        "draft_cursor_prompt",
        "architect_new_tool",
        "dispatch_titan_repair",
        "delegate_to_cursor",
    }
)

_REASONER_SYSTEM = """
You are Dana's MoA reasoner (DeepSeek-R1). You do NOT call tools and you do NOT
emit JSON tool-call envelopes.

You must enclose your internal chain of thought in <think>...</think> tags before
generating your final structured response.

Given the user request (and any forced tool hint), write a precise execution plan
a formatter model will convert into a native tool call.

Required sections (use these exact headings — plain text, no markdown bold):
INTENT: <single tool id from the allowed list, or NONE>
OBJECTIVE: <full, untruncated objective / goal text — complete sentences only>
CONTEXT: <structured brief for the tool — include paths, root cause, steps, acceptance when drafting tickets>
ARGS: <one key: value per line for any other tool arguments>
NOTES: <optional short caveats>

Rules:
- Always emit <think>...</think> first, then the structured plan outside the tags.
- Copy technical details from the user verbatim when they matter (paths, APIs, IDs).
- Never invent a shorter "summary" objective when the user gave a full ticket brief.
- If the forced tool hint is set, INTENT must equal that tool id.
- Keep the plan under ~800 words. Prefer completeness of OBJECTIVE/CONTEXT over fluff.
- Do not worry about Pydantic validation here — Stage-3 tool guards reject/retry malformed tool args.
""".strip()

_FORMATTER_INJECT_PREFIX = """
MOA TOOL-BINDING SHIM (HARD):
You are the formatter / tool-caller only (Llama). A DeepSeek reasoner already
planned this turn. You MUST:
1) Call the tool named in INTENT using native tool_calls (not prose).
2) Copy OBJECTIVE / CONTEXT / ARGS into tool arguments WITHOUT truncating,
   paraphrasing, or inventing softer wording.
3) Do not answer conversationally until the required tool has executed.
4) Do not invent phantom tool names.

=== MOA REASONER PLAN (authoritative) ===
""".strip()


def moa_shim_enabled() -> bool:
    """Feature flag — default ON. Set DANA_MOA_TOOL_SHIM=0 to disable."""
    raw = (os.environ.get("DANA_MOA_TOOL_SHIM") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def should_use_moa_tool_shim(
    user_text: str,
    *,
    forced_tool_id: str | None = None,
) -> bool:
    """True for high-complexity / MoA text routes that need the two-stage shim."""
    if not moa_shim_enabled():
        return False
    # MoA-class forced tools always use the shim (even if Mode=chat would
    # otherwise short-circuit decide_route to low/local).
    if (forced_tool_id or "").strip() in _MOA_DEFER_FORCE_TOOLS:
        return True
    text_l = (user_text or "").lower()
    if any(k in text_l for k in ("draft_cursor_prompt", "deepseek", "self-improvement")):
        return True
    decision = decide_route(
        user_text or "",
        forced_tool=forced_tool_id,
    )
    if decision.backend == "moa":
        return True
    if decision.complexity == "high":
        return True
    return False


def defer_forced_tool_for_moa(tool_id: str | None) -> bool:
    """Skip thin broker force-exec so the shim can supply complete args."""
    return (tool_id or "").strip() in _MOA_DEFER_FORCE_TOOLS


def _build_reasoner_llm(*, temperature: float = 0.2) -> Any:
    from langchain_ollama import ChatOllama

    model_id = reasoner_model_name()
    num_predict = 2048
    try:
        num_predict = max(
            512,
            int(os.environ.get("DANA_MOA_REASONER_NUM_PREDICT", "2048") or "2048"),
        )
    except ValueError:
        num_predict = 2048
    return ChatOllama(
        model=model_id,
        temperature=temperature,
        num_ctx=8192,
        num_predict=num_predict,
    )


def _llm_raw_text(result: Any) -> str:
    """Pull raw string content from a ChatOllama result (think tags intact).

    DeepSeek-R1 on Ollama often parks CoT in ``thinking`` /
    ``additional_kwargs`` while ``content`` is empty — harvest those so the
    Stage-3 extractor and Pydantic path still receive a payload.
    """
    content = getattr(result, "content", None)
    content_s = ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                # Multimodal / typed blocks (text, thinking, …).
                ptype = str(part.get("type") or "").lower()
                text = str(part.get("text") or part.get("thinking") or part)
                if ptype in {"thinking", "reasoning"}:
                    parts.append(f"<think>{text}</think>")
                else:
                    parts.append(text)
            else:
                parts.append(str(part))
        content_s = "".join(parts)
    elif content is not None:
        content_s = str(content)

    think_bits: list[str] = []
    for attr in ("thinking", "reasoning", "reasoning_content"):
        extra = getattr(result, attr, None)
        if extra and str(extra).strip():
            think_bits.append(str(extra).strip())
    ak = getattr(result, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        for key in ("thinking", "reasoning", "reasoning_content"):
            extra = ak.get(key)
            if extra and str(extra).strip():
                think_bits.append(str(extra).strip())
    meta = getattr(result, "response_metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("thinking", "reasoning", "reasoning_content"):
            extra = meta.get(key)
            if extra and str(extra).strip():
                think_bits.append(str(extra).strip())

    if think_bits and "<think" not in content_s.lower():
        think_blob = "\n\n".join(dict.fromkeys(think_bits))  # dedupe, keep order
        return f"<think>\n{think_blob}\n</think>\n{content_s}".strip()
    if content_s.strip():
        return content_s
    if think_bits:
        think_blob = "\n\n".join(dict.fromkeys(think_bits))
        return f"<think>\n{think_blob}\n</think>"
    return str(result or "")


def _llm_text(result: Any) -> str:
    """Clean post-think text only (legacy helper)."""
    from dana.agentic import extract_r1_think_blocks

    return extract_r1_think_blocks(_llm_raw_text(result)).clean_text.strip()


# Stage 3.2 — how many Blackboard turns to staple onto the reasoner prompt.
_MOA_HISTORY_TURNS = 5


def format_blackboard_history_block(
    session_id: str | None,
    *,
    user_text: str = "",
    max_turns: int = _MOA_HISTORY_TURNS,
) -> str:
    """Deterministic read-only history from SQLite Blackboard for MoA hydration.

    Fetches the newest ``max_turns`` user/assistant rows for ``session_id``.
    Drops a trailing user turn that duplicates the current ``user_text`` so the
    live request is not double-listed. Returns ``""`` when nothing usable exists.
    Does not alter plan/tool schemas — history is prompt context only.
    """
    sid = (session_id or "").strip()
    if not sid:
        return ""
    try:
        from dana.memory import load_messages
    except Exception:  # noqa: BLE001
        return ""

    # Pull a little extra so we can drop the duplicate current user turn.
    try:
        rows = load_messages(sid, limit=max(int(max_turns) + 2, 4))
    except Exception:  # noqa: BLE001
        return ""

    turns: list[dict[str, str]] = []
    for row in rows or []:
        role = str((row or {}).get("role") or "").strip().lower()
        content = str((row or {}).get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append({"role": role, "content": content})

    if not turns:
        return ""

    current = (user_text or "").strip()
    if (
        current
        and turns
        and turns[-1]["role"] == "user"
        and turns[-1]["content"].strip() == current
    ):
        turns = turns[:-1]

    turns = turns[-max(1, int(max_turns)) :]
    if not turns:
        return ""

    lines = ["[RECENT CONVERSATION HISTORY]", f"(session_id={sid}, read-only)"]
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Dana"
        # Keep each turn bounded so hydration cannot drown the reasoner plan.
        body = turn["content"]
        if len(body) > 480:
            body = body[:477] + "..."
        lines.append(f"{label}: {body}")
    lines.append("[END RECENT CONVERSATION HISTORY]")
    lines.append(
        "Use the history above to resolve pronouns and named entities "
        "(e.g. project names) in the current user request. "
        "Do not invent history that is not listed."
    )
    return "\n".join(lines)


def run_moa_reasoner_stage(
    user_text: str,
    *,
    forced_tool_id: str | None = None,
    allowed_tool_ids: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    """Stage 1: DeepSeek-R1 textual plan (no tools bound).

    Module 3: ``<think>`` blocks are extracted, filed on the Blackboard under
    ``session_id``, logged as ``[REASONING_TRACE]``, and stripped so only the
    clean plan is returned to the Stage-2 formatter.

    Stage 3.2: recent Blackboard turns are appended to the system message
    (core ``_REASONER_SYSTEM`` unchanged) before the LLM call.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from dana.agentic import extract_r1_think_blocks
    from dana.logging import log

    allowed = ", ".join(allowed_tool_ids or []) or "(bound tools will be provided next)"
    forced = (forced_tool_id or "").strip() or "NONE"
    human = (
        f"Forced tool hint: {forced}\n"
        f"Allowed tool ids: {allowed}\n\n"
        f"User request:\n{(user_text or '').strip()}"
    )
    # Stage 3.2 — staple Blackboard history after core instructions (append-only).
    history_block = format_blackboard_history_block(
        session_id,
        user_text=user_text or "",
        max_turns=_MOA_HISTORY_TURNS,
    )
    system_content = _REASONER_SYSTEM
    if history_block:
        system_content = f"{_REASONER_SYSTEM}\n\n{history_block}"
    llm = _build_reasoner_llm()
    t0 = time.perf_counter()
    log(
        "MoAShim",
        f"stage1 reasoner={reasoner_model_name()} forced={forced} "
        f"(no tools bound)"
        + (
            f" hydrated_history_chars={len(history_block)}"
            if history_block
            else " hydrated_history=none"
        ),
    )
    think_text = ""
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=human),
            ]
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        note_high_complexity_deepseek_latency(
            latency_ms,
            model=reasoner_model_name(),
        )
        raw = _llm_raw_text(result)
        extracted = extract_r1_think_blocks(raw)
        think_text = extracted.think_text
        plan = extracted.clean_text.strip()
        # Stage 3.1: empty <think> extract — pass full raw string downstream
        # so Pydantic tool guards (not MoA string gates) own rejection.
        if not think_text:
            plan = plan or (raw or "").strip()
            if not plan:
                plan = (raw or "").strip()
        elif not plan:
            # Think-only / empty clean — still forward full raw for guards.
            plan = (raw or "").strip()
        # Module 3 — file CoT + emit tagged telemetry; never pass think downstream.
        if think_text:
            try:
                from dana.memory import append_reasoning_trace
                from dana.telemetry import log_reasoning_trace

                sid = (session_id or "").strip()
                if sid:
                    append_reasoning_trace(
                        sid,
                        think_text,
                        clean_text=plan,
                        source=reasoner_model_name(),
                    )
                log_reasoning_trace(
                    think_text,
                    session_id=sid,
                    clean_text=plan,
                    payload={
                        "latency_ms": round(latency_ms, 1),
                        "think_chars": len(think_text),
                        "clean_chars": len(plan),
                        "reasoner": reasoner_model_name(),
                    },
                )
            except Exception as _tel_exc:  # noqa: BLE001
                log("MoAShim", f"reasoning trace file/log skipped: {_tel_exc}")
        elif (raw or "").strip():
            # Empty extract but non-empty raw — still emit a minimal reasoning tag
            # so soak/telemetry can see R1 fired without <think>.
            try:
                from dana.telemetry import log_reasoning_trace

                log_reasoning_trace(
                    "(no <think> tags — full raw forwarded to formatter/guards)",
                    session_id=(session_id or "").strip(),
                    clean_text=plan[:500],
                    payload={
                        "latency_ms": round(latency_ms, 1),
                        "think_chars": 0,
                        "clean_chars": len(plan),
                        "reasoner": reasoner_model_name(),
                        "empty_think_fallback": True,
                    },
                )
            except Exception as _tel_exc:  # noqa: BLE001
                log("MoAShim", f"empty-think telemetry skipped: {_tel_exc}")
        log(
            "MoAShim",
            f"stage1 think_chars={len(think_text)} clean_chars={len(plan)} "
            f"latency_ms={latency_ms:.0f}",
        )
    except Exception as exc:  # noqa: BLE001
        log("MoAShim", f"stage1 reasoner failed: {exc}")
        # Fallback plan so formatter can still force the broker tool.
        plan = (
            f"INTENT: {forced if forced != 'NONE' else 'NONE'}\n"
            f"OBJECTIVE: {(user_text or '').strip()}\n"
            f"CONTEXT: Reasoner unavailable ({exc}); use the full user request.\n"
            f"ARGS:\nNOTES: fallback"
        )
    if not plan.strip():
        plan = (
            f"INTENT: {forced if forced != 'NONE' else 'NONE'}\n"
            f"OBJECTIVE: {(user_text or '').strip()}\n"
            f"CONTEXT: Empty reasoner output; use the full user request.\n"
        )
    # If the reasoner left INTENT blank/NONE but we have a forced tool, stamp it.
    if forced != "NONE":
        fields = parse_plan_fields(plan)
        intent = (fields.get("intent") or "").strip().upper()
        if intent in {"", "NONE"}:
            plan = (
                f"INTENT: {forced}\n"
                f"OBJECTIVE: {(fields.get('objective') or user_text or '').strip()}\n"
                f"CONTEXT: {(fields.get('context') or user_text or '').strip()}\n"
                f"ARGS:\n{(fields.get('args') or '').strip()}\n"
                f"NOTES: forced tool stamped after empty INTENT\n"
            )
    # Hard guarantee: never leak think tags to Stage-2.
    plan = extract_r1_think_blocks(plan).clean_text.strip() or plan
    log("MoAShim", f"stage1 plan_chars={len(plan)} preview={plan[:160]!r}")
    return plan


def formatter_system_injection(plan: str) -> str:
    """Stage-2 system appendix for Llama bind_tools."""
    body = (plan or "").strip()
    return (
        f"{_FORMATTER_INJECT_PREFIX}\n{body}\n"
        "=== END MOA REASONER PLAN ===\n"
        f"Formatter model hint: {local_model_name()} with bind_tools."
    )


def parse_plan_fields(plan: str) -> dict[str, str]:
    """Best-effort extract INTENT/OBJECTIVE/CONTEXT from a reasoner plan.

    Accepts plain ``INTENT:`` and markdown ``**INTENT**:`` / ``**INTENT:**`` variants.
    """
    text = plan or ""
    out: dict[str, str] = {}
    # Only treat MoA top-level headings as section boundaries (not "Root cause:").
    _next = r"INTENT|OBJECTIVE|CONTEXT|ARGS|NOTES"

    def _section(name: str) -> str:
        # Heading may be wrapped in markdown bold on either/both sides of the colon.
        m = re.search(
            rf"(?is)(?:\*\*\s*)?{name}(?:\s*\*\*)?\s*:\s*(?:\*\*\s*)?"
            rf"(.*?)(?=\n\s*(?:\*\*\s*)?(?:{_next})(?:\s*\*\*)?\s*:|\Z)",
            text,
        )
        raw = (m.group(1).strip() if m else "").strip()
        # Drop leftover bold markers from section bodies.
        return re.sub(r"^\*\*\s*|\s*\*\*$", "", raw).strip()

    out["intent"] = _section("INTENT")
    out["objective"] = _section("OBJECTIVE")
    out["context"] = _section("CONTEXT")
    out["args"] = _section("ARGS")
    out["notes"] = _section("NOTES")
    return out


def plan_has_structured_context(plan: str) -> bool:
    """True when CONTEXT includes Target files, Root cause, Steps, Acceptance.

    Required for ``draft_cursor_prompt`` plans. Other INTENTs are treated as OK
    (structure gate applies only to ledger tickets).
    """
    fields = parse_plan_fields(plan)
    intent = (fields.get("intent") or "").strip().lower()
    if intent and intent not in {"", "none", "draft_cursor_prompt"}:
        return True
    # draft_cursor_prompt / NONE / empty → enforce four-part CONTEXT.
    ctx = fields.get("context") or ""
    if not ctx.strip():
        return False
    has_targets = bool(
        re.search(r"(?is)\btarget\s+files?\b", ctx)
        and re.search(r"(?i)\b[\w./\\-]+\.(?:py|md|json|txt|yml|yaml)\b", ctx)
    )
    has_root = bool(re.search(r"(?is)\broot\s+cause\b", ctx))
    has_steps = bool(
        re.search(r"(?is)\bstep[- ]?by[- ]?step(?:\s+changes?)?\b|\bsteps?\s*:", ctx)
    )
    has_accept = bool(re.search(r"(?is)\bacceptance\s+criteria\b", ctx))
    return bool(has_targets and has_root and has_steps and has_accept)


def enrich_forced_tool_from_plan(forced_tool: Any, plan: str) -> Any:
    """Merge reasoner OBJECTIVE/CONTEXT into a draft_cursor_prompt ToolCall."""
    from dataclasses import replace

    from dana.tools.schema import ToolCall

    if forced_tool is None or not isinstance(forced_tool, ToolCall):
        return forced_tool
    fields = parse_plan_fields(plan)
    args = dict(forced_tool.arguments or {})
    intent = (fields.get("intent") or "").strip()
    if intent and intent.upper() != "NONE" and intent != forced_tool.tool_id:
        # Keep forced id; reasoner may still help args.
        pass
    obj = (fields.get("objective") or "").strip()
    ctx = (fields.get("context") or "").strip()
    if forced_tool.tool_id == "draft_cursor_prompt":
        junk = ("empty reasoner output", "reasoner unavailable", "use the full user request")
        if obj and not any(j in obj.lower() for j in junk) and (
            not str(args.get("objective") or "").strip()
            or len(obj) > len(str(args.get("objective") or ""))
        ):
            args["objective"] = obj
        if ctx and not any(j in ctx.lower() for j in junk) and (
            not str(args.get("context") or "").strip()
            or len(ctx) > len(str(args.get("context") or ""))
        ):
            args["context"] = ctx
        return replace(forced_tool, arguments=args)
    if forced_tool.tool_id == "architect_new_tool" and obj:
        if not str(args.get("goal") or args.get("tool_description") or "").strip():
            args["goal"] = obj
        return replace(forced_tool, arguments=args)
    return forced_tool
