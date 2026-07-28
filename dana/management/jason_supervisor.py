"""Stage 6.2 — Jason CTO supervisor (bulk slide evaluation pipeline).

LangGraph workflow ``bulk_evaluate_slides``:
  ingest (.pptx via slide_parser) → per-slide MoA reasoner → enqueue
  ``type_stealth_text`` on the Blackboard action_queue.

Progress is persisted under sensor key ``jason_bulk_slide_progress`` so
interrupted runs skip already-queued slide_ids.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

from dana.memory.blackboard import (
    enqueue_action,
    get_sensor_state,
    set_sensor_state,
)
from dana.tools.slide_parser import parse_slides_in_directory

PROGRESS_KEY = "jason_bulk_slide_progress"

# Physical actions Jason must not enqueue during Shadow Run (audit mode).
_AUDIT_BLOCKED_TOOLS: frozenset[str] = frozenset({"type_stealth_text", "press_key"})


class BulkSlideState(TypedDict, total=False):
    directory: str
    session_id: str
    slides: list[dict[str, Any]]
    index: int
    evaluations: list[dict[str, Any]]
    enqueued: list[dict[str, Any]]
    skipped: list[str]
    status: str
    history: list[dict[str, Any]]


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("JasonCTO", msg)
    except Exception:  # noqa: BLE001
        print(f"[JasonCTO] {msg}", flush=True)


# Stage 8.2 — Feather grading rubric injected before DeepSeek slide eval.
_FEATHER_RULES_REL = Path("dana") / "knowledge" / "feather_project_rules.md"


def feather_project_rules_path() -> Path:
    """Absolute path to ``dana/knowledge/feather_project_rules.md``."""
    return Path(__file__).resolve().parents[2] / _FEATHER_RULES_REL


def load_feather_project_rules() -> str:
    """Read Feather project rules markdown (empty file → empty string)."""
    path = feather_project_rules_path()
    try:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        _log(f"feather rules read failed: {exc}")
        return ""


def feather_rules_system_preamble(rules: str | None = None) -> str:
    """System-prompt prefix Jason injects before routing a slide to DeepSeek."""
    body = rules if rules is not None else load_feather_project_rules()
    body = (body or "").strip() or "(no project rules on file)"
    return (
        "Strictly evaluate this slide against the following project rules: "
        f"{body}."
    )


def audit_mode_enabled() -> bool:
    """True when ``DONNA_AUDIT_MODE=1`` — no physical actuator enqueue."""
    return os.environ.get("DONNA_AUDIT_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def shadow_run_audit_path() -> Path:
    """``logs/shadow_run_audit.txt`` under the Donna workspace."""
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR) / "shadow_run_audit.txt"
    except Exception:  # noqa: BLE001
        return Path("logs") / "shadow_run_audit.txt"


def append_shadow_run_audit(
    *,
    slide_id: str,
    evaluation: str,
    tool_name: str = "type_stealth_text",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one Shadow Run evaluation block; return the audit file path."""
    path = shadow_run_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bits = [
        f"===== SHADOW RUN {ts} =====",
        f"session_id: {session_id or '-'}",
        f"slide_id: {slide_id}",
        f"would_enqueue: {tool_name}",
        "evaluation:",
        (evaluation or "").strip() or "(empty)",
    ]
    if extra:
        bits.append(f"extra: {json.dumps(extra, ensure_ascii=False)}")
    bits.append("")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(bits) + "\n")
    return path


def enqueue_jason_physical(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    session_id: str = "",
    db_path: Path | str | None = None,
    audit_slide_id: str = "",
    audit_text: str = "",
) -> int:
    """Enqueue a physical tool, or write Shadow Run audit when audit mode is on."""
    name = (tool_name or "").strip()
    args = dict(arguments or {})
    if audit_mode_enabled() and name in _AUDIT_BLOCKED_TOOLS:
        path = append_shadow_run_audit(
            slide_id=audit_slide_id or str(args.get("slide_id") or "n/a"),
            evaluation=audit_text or str(args.get("text") or args.get("key_name") or ""),
            tool_name=name,
            session_id=session_id,
            extra=args,
        )
        _log(f"AUDIT skip enqueue tool={name!r} wrote={path}")
        return 0
    return enqueue_action(
        name,
        args,
        session_id=session_id,
        db_path=db_path,
    )


def _dry_reasoner() -> bool:
    return os.environ.get("DONNA_JASON_DRY_REASONER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_progress(*, db_path: Path | str | None = None) -> dict[str, Any]:
    row = get_sensor_state(PROGRESS_KEY, db_path=db_path)
    if not row:
        return {"completed_slide_ids": [], "queued_actions": {}}
    raw = row.get("value") or "{}"
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("completed_slide_ids", [])
    data.setdefault("queued_actions", {})
    return data


def save_progress(
    progress: dict[str, Any],
    *,
    db_path: Path | str | None = None,
) -> None:
    set_sensor_state(
        PROGRESS_KEY,
        json.dumps(progress, ensure_ascii=False),
        meta={"publisher": "jason_supervisor", "pipeline": "bulk_evaluate_slides"},
        db_path=db_path,
    )


def strip_evaluation_text(raw: str) -> str:
    """Strip markdown / conversational filler for Ghost Typist raw text."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Drop fenced code blocks / think tags.
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"(?is)```.*?```", " ", text)
    # Drop markdown emphasis / headings / bullets.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"[*`_~]", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    # Drop common chat wrappers.
    text = re.sub(
        r"(?is)^\s*(sure[,!]?\s+|here(?:'s| is)\s+|evaluation:\s*|verdict:\s*)",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def reason_slide_evaluation(
    instructions: str,
    content: str,
    *,
    reasoner_fn: Callable[[str], str] | None = None,
    session_id: str = "",
) -> str:
    """Ask MoA/DeepSeek for a concise evaluation string (no markdown)."""
    # Stage 8.2 — inject Feather project rules at the top of the system prompt.
    rules_preamble = feather_rules_system_preamble()
    # Stage 8.5 — Behavior Mixer weights from Blackboard persona_mixer.
    try:
        from dana.memory.blackboard import behavior_mixer_prompt_weights

        behavior = behavior_mixer_prompt_weights()
    except Exception:  # noqa: BLE001
        behavior = ""
    prompt = (
        f"{rules_preamble}\n\n"
        f"{behavior}\n\n"
        "You are Donna's MoA slide reasoner. Evaluate the slide CONTENT against "
        "the INSTRUCTIONS. Reply with ONE concise plain-text evaluation only — "
        "no markdown, no bullet lists, no greetings, no JSON.\n\n"
        f"INSTRUCTIONS:\n{(instructions or '').strip() or '(none)'}\n\n"
        f"CONTENT:\n{(content or '').strip() or '(empty)'}\n"
    )
    if reasoner_fn is not None:
        return strip_evaluation_text(reasoner_fn(prompt))

    if _dry_reasoner():
        inst = (instructions or "").strip() or "rule"
        body = (content or "").strip() or "empty"
        words = len(re.findall(r"[A-Za-z0-9']+", body))
        return strip_evaluation_text(
            f"Slide follows '{inst[:80]}' with approximately {words} words; "
            f"review notes: content present={'yes' if body else 'no'}; "
            f"rules_applied={rules_preamble[:120]}"
        )

    try:
        from dana.cascade_router import reasoner_model_name
        from dana.core_agent import ask_ollama_messages

        raw = ask_ollama_messages(
            [
                {
                    "role": "system",
                    "content": (
                        f"{rules_preamble}\n\n"
                        "Output only a short plain-text slide evaluation. "
                        "No markdown. No preamble."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=reasoner_model_name(),
        )
        return strip_evaluation_text(str(raw or ""))
    except Exception as exc:  # noqa: BLE001
        _log(f"reasoner fallback: {exc}")
        return strip_evaluation_text(
            f"Unable to reach reasoner ({exc}); manual review required for "
            f"instructions={instructions[:60]!r}."
        )


def enqueue_stealth_evaluation(
    evaluation: str,
    *,
    slide_id: str,
    session_id: str = "",
    db_path: Path | str | None = None,
) -> int:
    """INSERT ``type_stealth_text`` pending action; return action_id.

    When ``DONNA_AUDIT_MODE=1``, skip the physical queue and append DeepSeek's
    evaluation to ``logs/shadow_run_audit.txt`` (returns ``0``).
    """
    if audit_mode_enabled():
        path = append_shadow_run_audit(
            slide_id=slide_id,
            evaluation=evaluation,
            tool_name="type_stealth_text",
            session_id=session_id,
        )
        _log(f"AUDIT shadow write slide_id={slide_id} path={path}")
        return 0
    return enqueue_action(
        "type_stealth_text",
        {
            "text": evaluation,
            "wait_hotkey": True,
            "hotkey": "f9",
            "slide_id": slide_id,
        },
        session_id=session_id or "jason-bulk-slides",
        db_path=db_path,
    )


def ingest_node(state: BulkSlideState) -> dict[str, Any]:
    directory = str(state.get("directory") or "")
    slides = parse_slides_in_directory(directory)
    # Drop parse-error stubs without content.
    usable = [
        s
        for s in slides
        if not s.get("error")
        and (s.get("instructions") or s.get("content"))
    ]
    _log(f"ingest directory={directory!r} slides={len(usable)}")
    return {
        "slides": usable,
        "index": 0,
        "evaluations": list(state.get("evaluations") or []),
        "enqueued": list(state.get("enqueued") or []),
        "skipped": list(state.get("skipped") or []),
        "status": "ingested",
        "history": list(state.get("history") or [])
        + [{"event": "ingest", "slide_count": len(usable)}],
    }


def evaluate_one_node(
    state: BulkSlideState,
    *,
    reasoner_fn: Callable[[str], str] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    slides = list(state.get("slides") or [])
    idx = int(state.get("index") or 0)
    if idx >= len(slides):
        return {"status": "complete"}

    slide = slides[idx]
    slide_id = str(slide.get("slide_id") or f"slide#{idx}")
    progress = load_progress(db_path=db_path)
    completed = {str(x) for x in (progress.get("completed_slide_ids") or [])}
    skipped = list(state.get("skipped") or [])
    evaluations = list(state.get("evaluations") or [])
    enqueued = list(state.get("enqueued") or [])
    history = list(state.get("history") or [])
    session_id = str(state.get("session_id") or "jason-bulk-slides")

    if slide_id in completed:
        _log(f"skip already-queued slide_id={slide_id}")
        skipped.append(slide_id)
        history.append({"event": "skip", "slide_id": slide_id})
        return {
            "index": idx + 1,
            "skipped": skipped,
            "history": history,
            "status": "evaluating",
        }

    evaluation = reason_slide_evaluation(
        str(slide.get("instructions") or ""),
        str(slide.get("content") or ""),
        reasoner_fn=reasoner_fn,
        session_id=session_id,
    )
    action_id = enqueue_stealth_evaluation(
        evaluation,
        slide_id=slide_id,
        session_id=session_id,
        db_path=db_path,
    )
    evaluations.append(
        {
            "slide_id": slide_id,
            "evaluation": evaluation,
            "action_id": action_id,
            "audit": bool(audit_mode_enabled() and action_id == 0),
        }
    )
    enqueued.append(
        {
            "slide_id": slide_id,
            "action_id": action_id,
            "audit": bool(audit_mode_enabled() and action_id == 0),
        }
    )
    completed.add(slide_id)
    progress["completed_slide_ids"] = sorted(completed)
    queued = dict(progress.get("queued_actions") or {})
    queued[slide_id] = action_id
    progress["queued_actions"] = queued
    progress["directory"] = str(state.get("directory") or "")
    save_progress(progress, db_path=db_path)
    history.append(
        {
            "event": "enqueue",
            "slide_id": slide_id,
            "action_id": action_id,
            "evaluation_chars": len(evaluation),
        }
    )
    _log(f"enqueued type_stealth_text slide_id={slide_id} action_id={action_id}")
    return {
        "index": idx + 1,
        "evaluations": evaluations,
        "enqueued": enqueued,
        "skipped": skipped,
        "history": history,
        "status": "evaluating",
    }


def _should_continue(state: BulkSlideState) -> str:
    slides = state.get("slides") or []
    idx = int(state.get("index") or 0)
    if idx < len(slides):
        return "evaluate"
    return "done"


def build_bulk_evaluate_slides_graph(
    *,
    reasoner_fn: Callable[[str], str] | None = None,
    db_path: Path | str | None = None,
) -> Any:
    """Compile Jason's bulk_evaluate_slides StateGraph."""
    from langgraph.graph import END, START, StateGraph

    def _eval(state: BulkSlideState) -> dict[str, Any]:
        return evaluate_one_node(state, reasoner_fn=reasoner_fn, db_path=db_path)

    def _finalize(state: BulkSlideState) -> dict[str, Any]:
        gc_stats: dict[str, Any] = {}
        try:
            from dana.memory.garbage_collector import run_blackboard_gc

            gc_stats = run_blackboard_gc(db_path=db_path)
            _log(
                f"blackboard GC deleted={gc_stats.get('deleted')} "
                f"vacuumed={gc_stats.get('vacuumed')}"
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"blackboard GC skipped: {exc}")
            gc_stats = {"ok": False, "error": str(exc)}
        return {
            "status": "complete",
            "history": list(state.get("history") or [])
            + [
                {
                    "event": "done",
                    "enqueued": len(state.get("enqueued") or []),
                    "skipped": len(state.get("skipped") or []),
                    "gc": gc_stats,
                }
            ],
        }

    g = StateGraph(BulkSlideState)
    g.add_node("ingest", ingest_node)
    g.add_node("evaluate", _eval)
    g.add_node("finalize", _finalize)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        _should_continue,
        {"evaluate": "evaluate", "done": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def bulk_evaluate_slides(
    directory: Path | str,
    *,
    session_id: str = "jason-bulk-slides",
    reasoner_fn: Callable[[str], str] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run Jason's bulk slide evaluation pipeline; return final state."""
    graph = build_bulk_evaluate_slides_graph(
        reasoner_fn=reasoner_fn,
        db_path=db_path,
    )
    initial: BulkSlideState = {
        "directory": str(Path(directory).resolve()),
        "session_id": session_id,
        "slides": [],
        "index": 0,
        "evaluations": [],
        "enqueued": [],
        "skipped": [],
        "status": "start",
        "history": [],
    }
    final = graph.invoke(initial)
    return dict(final)


def reset_bulk_progress(*, db_path: Path | str | None = None) -> None:
    """Clear Jason bulk-slide progress (tests / fresh batches)."""
    save_progress(
        {"completed_slide_ids": [], "queued_actions": {}},
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Stage 6.5 — Andon Cord / recovery_mode
# ---------------------------------------------------------------------------

OPERATOR_ANDON_TOOLS: frozenset[str] = frozenset(
    {"navigate_and_click", "type_stealth_text"}
)

MAX_ANDON_DEPTH = 2


class RecoveryState(TypedDict, total=False):
    failed_action_id: int
    failed_tool: str
    error_context: str
    failed_arguments: dict[str, Any]
    session_id: str
    visual_context: str
    ticket: str
    plan: list[dict[str, Any]]
    enqueued: list[dict[str, Any]]
    status: str


def format_wake_cto_ticket(task_name: str, error_context: str) -> str:
    """Canonical Andon wake ticket for Jason."""
    return (
        f"Operator failed on task {task_name}. "
        f"Reason: {error_context}. "
        "Review perception.ocr (Florence) and generate a recovery plan."
    )


def enqueue_wake_cto_ticket(
    *,
    task_name: str,
    error_context: str,
    failed_action_id: int,
    failed_arguments: dict[str, Any] | None = None,
    session_id: str = "",
    db_path: Path | str | None = None,
) -> int:
    """Enqueue a ``wake_cto`` action; return action_id (0 if already woken)."""
    wake_key = f"andon_wake_{int(failed_action_id)}"
    prior = get_sensor_state(wake_key, db_path=db_path)
    if prior and str(prior.get("value") or "").strip():
        return 0

    ticket = format_wake_cto_ticket(task_name, error_context)
    wake_id = enqueue_action(
        "wake_cto",
        {
            "ticket": ticket,
            "task_name": task_name,
            "error_context": error_context,
            "failed_action_id": int(failed_action_id),
            "failed_arguments": dict(failed_arguments or {}),
        },
        session_id=session_id or "andon-cord",
        db_path=db_path,
    )
    set_sensor_state(
        wake_key,
        str(wake_id),
        meta={"publisher": "andon_cord", "task_name": task_name},
        db_path=db_path,
    )
    return wake_id


def trigger_andon_cord(
    *,
    task_name: str,
    error_context: str,
    failed_action_id: int,
    failed_arguments: dict[str, Any] | None = None,
    session_id: str = "",
    db_path: Path | str | None = None,
    run_recovery: bool = True,
) -> dict[str, Any]:
    """Log Andon wake ticket and optionally run Jason recovery_mode immediately."""
    wake_id = enqueue_wake_cto_ticket(
        task_name=task_name,
        error_context=error_context,
        failed_action_id=failed_action_id,
        failed_arguments=failed_arguments,
        session_id=session_id,
        db_path=db_path,
    )
    _log(
        f"ANDON wake_cto action_id={wake_id} failed_action_id={failed_action_id} "
        f"task={task_name!r}"
    )
    out: dict[str, Any] = {
        "wake_cto_action_id": wake_id,
        "failed_action_id": int(failed_action_id),
        "ticket": format_wake_cto_ticket(task_name, error_context),
    }
    if run_recovery and wake_id:
        # Mark wake ticket running→completed via recovery (or claim later).
        recovery = recovery_mode(
            failed_action_id=int(failed_action_id),
            failed_tool=task_name,
            error_context=error_context,
            failed_arguments=dict(failed_arguments or {}),
            session_id=session_id,
            db_path=db_path,
            ticket=out["ticket"],
        )
        out["recovery"] = recovery
        try:
            from dana.memory.blackboard import resolve_action

            resolve_action(
                wake_id,
                status="completed",
                result=json.dumps(
                    {
                        "ok": True,
                        "enqueued": recovery.get("enqueued") or [],
                        "plan": recovery.get("plan") or [],
                    },
                    ensure_ascii=False,
                ),
                db_path=db_path,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"ANDON wake resolve skipped: {exc}")
    return out


def plan_recovery_sequence(
    *,
    failed_tool: str,
    error_context: str,
    failed_arguments: dict[str, Any],
    visual_context: str,
) -> list[dict[str, Any]]:
    """Heuristic recovery plan (popup clear → retry original operator)."""
    err = (error_context or "").lower()
    visual = (visual_context or "").lower()
    depth = int(failed_arguments.get("_andon_depth") or 0)
    plan: list[dict[str, Any]] = []

    popup_hint = any(
        k in err or k in visual
        for k in ("popup", "dialog", "modal", "overlay", "blocked", "close")
    )
    missing_target = any(
        k in err
        for k in ("not found", "disappeared", "never arrived", "target box")
    )

    # Always attempt a close/dismiss when UI may be blocking, or on nav miss.
    if popup_hint or missing_target or failed_tool == "navigate_and_click":
        plan.append(
            {
                "tool_name": "click_close_button",
                "arguments": {
                    "reason": "clear_blocking_ui",
                    "failed_tool": failed_tool,
                    "_andon_depth": depth,
                },
            }
        )

    if depth < MAX_ANDON_DEPTH:
        retry_args = dict(failed_arguments or {})
        retry_args["_andon_depth"] = depth + 1
        retry_args["_andon_retry_of"] = failed_tool
        plan.append(
            {
                "tool_name": failed_tool or "navigate_and_click",
                "arguments": retry_args,
            }
        )
    return plan


def announce_jason_andon_override(*, block: bool = False) -> dict[str, Any]:
    """Stage 8.1 — synthesize Jason's Andon line and play with ducking."""
    try:
        from dana.audio.multi_voice_tts import (
            JASON_ANDON_LINE,
            synthesize_jason_andon_line,
        )
        from dana.ui.audio_mixer import play_jason

        wav = synthesize_jason_andon_line()
        if block:
            ok = play_jason(wav, block=True)
        else:
            # Non-blocking: generate then play on a daemon so recovery continues.
            def _run() -> None:
                try:
                    play_jason(wav, block=True)
                except Exception as exc:  # noqa: BLE001
                    _log(f"jason andon playback failed: {exc}")

            threading.Thread(target=_run, name="JasonAndonVoice", daemon=True).start()
            ok = True
        _log(f"Jason Andon voice queued ok={ok} line={JASON_ANDON_LINE!r}")
        return {"ok": bool(ok), "path": str(wav), "line": JASON_ANDON_LINE}
    except Exception as exc:  # noqa: BLE001
        _log(f"Jason Andon voice skipped: {exc}")
        return {"ok": False, "error": str(exc)}


def recovery_mode(
    *,
    failed_action_id: int = 0,
    failed_tool: str = "",
    error_context: str = "",
    failed_arguments: dict[str, Any] | None = None,
    session_id: str = "",
    db_path: Path | str | None = None,
    ticket: str = "",
    visual_context: str | None = None,
) -> dict[str, Any]:
    """Jason recovery_mode: read screen state, enqueue close + retry sequence."""
    from dana.memory.blackboard import get_action, read_perception_ocr_text

    # Stage 8.1 — Jason talks over Donna as Andon wakes.
    voice_meta = announce_jason_andon_override()

    args = dict(failed_arguments or {})
    tool = (failed_tool or "").strip()
    err = (error_context or "").strip()

    if failed_action_id and (not tool or not err):
        row = get_action(int(failed_action_id), db_path=db_path)
        if row:
            tool = tool or str(row.get("tool_name") or "")
            err = err or str(row.get("error_context") or row.get("result") or "")
            if not args:
                args = dict(row.get("arguments") or {})

    # Fail closed on YOLO-only Blackboard: recovery needs OCR grounding.
    visual = (
        visual_context
        if visual_context is not None
        else read_perception_ocr_text(db_path=db_path)
    )
    if visual_context is None and not (visual or "").strip():
        visual = (
            "(no OCR: perception.ocr missing — recovery limited; "
            "run ocr_with_region for grounded UI state)"
        )
    ticket_text = ticket or format_wake_cto_ticket(tool or "operator", err or "unknown")
    plan = plan_recovery_sequence(
        failed_tool=tool or "navigate_and_click",
        error_context=err,
        failed_arguments=args,
        visual_context=visual or "",
    )
    enqueued: list[dict[str, Any]] = []
    sid = session_id or "andon-recovery"
    for step in plan:
        tool_n = str(step["tool_name"])
        step_args = dict(step.get("arguments") or {})
        aid = enqueue_jason_physical(
            tool_n,
            step_args,
            session_id=sid,
            db_path=db_path,
            audit_slide_id=f"recovery:{failed_action_id}",
            audit_text=str(
                step_args.get("text")
                or step_args.get("key_name")
                or tool_n
            ),
        )
        enqueued.append({"tool_name": tool_n, "action_id": aid})
        _log(
            f"recovery enqueue tool={tool_n!r} action_id={aid} "
            f"failed_action_id={failed_action_id}"
        )

    try:
        set_sensor_state(
            "jason_andon_last_recovery",
            json.dumps(
                {
                    "failed_action_id": int(failed_action_id),
                    "failed_tool": tool,
                    "error_context": err,
                    "ticket": ticket_text,
                    "enqueued": enqueued,
                    "visual_preview": (visual or "")[:240],
                    "voice": voice_meta,
                },
                ensure_ascii=False,
            ),
            meta={"publisher": "jason_supervisor", "mode": "recovery"},
            db_path=db_path,
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "status": "recovery_enqueued",
        "failed_action_id": int(failed_action_id),
        "failed_tool": tool,
        "error_context": err,
        "ticket": ticket_text,
        "visual_context": visual or "",
        "plan": plan,
        "enqueued": enqueued,
        "voice": voice_meta,
    }


def handle_wake_cto(
    arguments: dict[str, Any] | None = None,
    *,
    db_path: Path | str | None = None,
) -> str:
    """Actuator entry for ``wake_cto`` → Jason recovery_mode."""
    args = dict(arguments or {})
    result = recovery_mode(
        failed_action_id=int(args.get("failed_action_id") or 0),
        failed_tool=str(args.get("task_name") or args.get("failed_tool") or ""),
        error_context=str(args.get("error_context") or ""),
        failed_arguments=(
            dict(args["failed_arguments"])
            if isinstance(args.get("failed_arguments"), dict)
            else {}
        ),
        session_id=str(args.get("session_id") or "andon-recovery"),
        db_path=db_path,
        ticket=str(args.get("ticket") or ""),
    )
    return (
        f"OK: wake_cto recovery enqueued={len(result.get('enqueued') or [])} "
        f"failed_action_id={result.get('failed_action_id')}"
    )
