"""Fire-and-forget research-swarm dispatcher (in-process background thread).

Keeps the ReAct voice loop free: starts ``dana.swarm.swarm_main.run_research``
(PlannerAgent → Search Agent + WebSearchTool → WriterAgent) on a daemon thread
and returns immediately.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

OnComplete = Callable[[str, str], None]  # (topic, spoken_summary)


def _default_on_complete(topic: str, summary: str) -> None:
    """Best-effort TTS handoff — lazy import avoids circular import at module load."""
    try:
        from dana.core_agent import enqueue_speech

        text = f"My research is complete. Here is what I found: {summary}"
        enqueue_speech(text)
    except Exception:
        pass


def _swarm_worker(topic: str, on_complete: OnComplete | None) -> None:
    try:
        from dana.swarm.swarm_main import run_research

        summary = run_research(topic)
    except Exception as exc:  # noqa: BLE001
        summary = f"The background research on {topic} failed: {exc}"
    cb = on_complete if on_complete is not None else _default_on_complete
    try:
        cb(topic, summary)
    except Exception:
        pass


def dispatch_research_swarm(
    topic: str,
    *,
    on_complete: OnComplete | None = None,
) -> str:
    """Spin a background research thread; never wait for it to finish.

    Returns immediately with a fixed OK observation for the ReAct loop.
    """
    q = (topic or "").strip()
    if not q:
        return "ERROR: missing topic"

    thread = threading.Thread(
        target=_swarm_worker,
        args=(q, on_complete),
        name="DonnaResearchSwarm",
        daemon=True,
    )
    thread.start()
    return f"OK: Background research swarm dispatched for topic: {q}."


def _titan_repair_worker(query: str, on_complete: OnComplete | None) -> None:
    try:
        from dana.swarm.titan_repair import run_titan_repair

        summary = run_titan_repair(query=query)
    except Exception as exc:  # noqa: BLE001
        summary = f"Titan Repair failed: {exc}"
    cb = on_complete if on_complete is not None else _default_titan_repair_complete
    try:
        cb(query or "bug_tracker", summary)
    except Exception:
        pass


def _default_titan_repair_complete(topic: str, summary: str) -> None:
    try:
        from dana.core_agent import enqueue_speech

        text = (summary or "").strip() or (
            "Titan Repair finished. Drafts are in CAMGRASPER/tracker/pending_patches/ "
            "for your review — nothing was applied to source."
        )
        enqueue_speech(text)
    except Exception:
        pass


def dispatch_titan_repair(
    query: str = "",
    *,
    on_complete: OnComplete | None = None,
) -> str:
    """Fire-and-forget Titan Repair swarm over docs/bug_tracker.json."""
    thread = threading.Thread(
        target=_titan_repair_worker,
        args=((query or "").strip(), on_complete),
        name="DonnaTitanRepair",
        daemon=True,
    )
    thread.start()
    return (
        "OK: Titan Repair swarm dispatched — reading PENDING bugs from "
        "bug_tracker.json and drafting patches into CAMGRASPER/tracker/pending_patches/ "
        "(will ask you to review; will not hot-patch core source)."
    )


def handle_tool_call(call: Any) -> str:
    """Broker/plugin-compatible handler."""
    args = getattr(call, "arguments", None) or {}
    if not isinstance(args, dict):
        args = {}
    topic = args.get("topic")
    if topic is None or not str(topic).strip():
        topic = args.get("query")
    if topic is None or not str(topic).strip():
        return "ERROR: missing topic"
    return dispatch_research_swarm(str(topic).strip())
