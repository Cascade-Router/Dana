"""Pillar 3 — local-GPU context distillation.

Audit finding this module addresses: NO cross-turn conversation history is
currently replayed into the LLM at all. ``dana.api.server._process_user_text``
starts a brand-new ``messages`` list (system prompt + this one user turn) on
every NEW user turn — the prior turn's assistant reply/tool trace is never
resent (only WITHIN one turn's multi-step ReAct chain, capped at
``_MAX_REACT_ITERATIONS``, does ``messages`` grow). That already keeps
per-turn token cost flat — there is no unbounded-history problem to solve
here — but it also means the model has zero memory of turn N-1 unless the
agent explicitly chose to write it into Core Memory
(``dana.plugins.memory.core_memory``) or the Task Planner
(``dana.plugins.planning.task_board``), both of which require the model to
decide, on its own initiative, that something was worth persisting.

This module closes that continuity gap for free rather than solving the
(nonexistent) unbounded-history problem the original ask assumed: after
every turn finishes, the LOCAL Ollama model — the user's RTX 2080, zero
cloud cost, off the request's hot path — folds the just-completed turn into
a rolling, word-capped "working memory" summary kept in the session dict.
``dana.core.react_dispatch.build_system_prompt`` threads that summary into
every subsequent turn's system prompt as a dense stand-in for "resend the
whole conversation" — the cloud model gains continuity without ever seeing
more than one short paragraph of it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dana.core.model_provider import ModelProvider
from dana.system_health import llm_lock

_MAX_SUMMARY_WORDS = 150
# Defense in depth alongside the word cap: a model that ignores "150 words"
# and answers in a script with no whitespace to split on (dense CJK output,
# a stray code block, ...) would sail through a word-count trim untouched —
# a hard character ceiling bounds session["working_memory"] regardless of
# what the local model actually returns. ~8 chars/word average English
# prose, so this only ever bites when the word cap alone wouldn't have.
_MAX_SUMMARY_CHARS = 1200
_DISTILL_MODEL_ENV = "DANA_DISTILL_MODEL"
# Same default the rest of the local-agent stack already assumes is present
# (dana.core.model_provider._DEFAULT_LOCAL_MODEL) — no new model pull required.
_DEFAULT_DISTILL_MODEL = "qwen2.5-coder:7b"
# Hard ceiling on the WHOLE distillation attempt (lock wait + HTTP call) —
# same rationale/order of magnitude as dana.core.react_dispatch's own
# _LOCAL_TOOL_CALL_TIMEOUT_SEC for the primary ReAct turn: a stalling or
# VRAM-fragmented Ollama must never be allowed to run this background task
# indefinitely, even though nothing is waiting on its result.
_DISTILL_TIMEOUT_SEC = 20.0


def distillation_enabled() -> bool:
    raw = (os.environ.get("DANA_CONTEXT_DISTILL") or "").strip().lower()
    return raw not in {"0", "false", "off"}


def _distill_model() -> str:
    return (os.environ.get(_DISTILL_MODEL_ENV) or "").strip() or _DEFAULT_DISTILL_MODEL


def _build_distill_messages(prior_summary: str, user_text: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You maintain a terse rolling memory of an ongoing assistant session for "
                "another AI to read as context on its NEXT turn. Given the CURRENT SUMMARY "
                "(may be empty) and the LATEST EXCHANGE, write an UPDATED summary in plain "
                f"prose, {_MAX_SUMMARY_WORDS} words or fewer. Keep durable facts (objects "
                "created, decisions made, open goals, user preferences); drop resolved or "
                "no-longer-relevant detail. Output ONLY the updated summary, no preamble."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CURRENT SUMMARY:\n{prior_summary or '(none yet)'}\n\n"
                f"LATEST EXCHANGE:\nUser: {user_text}\nAssistant: {assistant_text}\n\n"
                "UPDATED SUMMARY:"
            ),
        },
    ]


def _trim_summary(text: str, *, max_words: int, max_chars: int) -> str:
    """Deterministic post-hoc cap — applied regardless of whether the local
    model actually honored the prompt's own word-count instruction, so
    ``session["working_memory"]`` can never grow past this bound no matter
    how long a session runs (50+ turns) or how the model misbehaves.
    """
    cleaned = (text or "").strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip() + "…"
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def _local_gpu_busy() -> bool:
    """Best-effort, non-blocking check for whether something else (almost
    always a LIVE ReAct turn's own local-model call) currently holds
    ``dana.system_health.llm_lock`` — the same lock ``ModelProvider``'s local
    path serializes every Ollama generation behind, specifically to avoid
    doubling VRAM usage / crashing the GPU with concurrent generations.

    Distillation must NEVER make a live turn wait for the GPU — that would
    turn a best-effort background summary into a real user-facing latency
    regression. Rather than blocking (like every other local call in this
    codebase deliberately does), this takes a non-blocking probe: if the
    lock is free, acquire-then-immediately-release it and proceed (the
    normal blocking call right after will re-acquire it near-instantly);
    if it's already held, skip this round's distillation entirely — the
    working-memory summary simply stays one turn stale, which is a fully
    acceptable trade for a best-effort feature, unlike blocking the user's
    next real reply behind a background summarization call.
    """
    acquired = llm_lock.acquire(blocking=False)
    if acquired:
        llm_lock.release()
        return False
    return True


async def distill_turn(session: dict[str, Any], user_text: str, assistant_text: str) -> None:
    """Fire-and-forget: update ``session["working_memory"]`` from the turn
    that just finished. Never raises — a distillation failure (Ollama not
    running, the RTX 2080 unreachable/under external load, a timeout, a
    malformed response, anything) must never surface to the user or block
    the next turn; the session simply keeps whatever summary (possibly
    none) it already had.
    """
    if not distillation_enabled() or not (user_text or "").strip():
        return
    if _local_gpu_busy():
        return
    prior = (session.get("working_memory") or {}).get("summary", "")
    try:
        provider = ModelProvider(local_model=_distill_model())
        messages = _build_distill_messages(prior, user_text, assistant_text)
        summary = await asyncio.wait_for(
            asyncio.to_thread(
                provider.complete,
                messages,
                num_predict=220,
                temperature=0.1,
                allow_cloud=False,  # local RTX 2080 only — must never spend cloud TPM budget
            ),
            timeout=_DISTILL_TIMEOUT_SEC,
        )
        summary = _trim_summary(summary, max_words=_MAX_SUMMARY_WORDS, max_chars=_MAX_SUMMARY_CHARS)
        if summary:
            session["working_memory"] = {"summary": summary, "turn": session.get("turn_counter", 0)}
    except Exception:  # noqa: BLE001 — background distillation is best-effort only; never propagate
        pass


def schedule_distillation(session: dict[str, Any], user_text: str, assistant_text: str) -> None:
    """Schedules ``distill_turn`` as a background asyncio task — never
    awaited by the caller, so a slow/unreachable local model can never delay
    the reply the user is already looking at (``_finish_turn`` has already
    sent it by the time this runs).
    """
    try:
        asyncio.create_task(distill_turn(session, user_text, assistant_text))
    except RuntimeError:  # noqa: BLE001 — no running event loop (e.g. a sync test harness)
        pass


__all__ = ("distill_turn", "distillation_enabled", "schedule_distillation")
