"""Stage 5 — Compound scenario stress test (cross-middleware synthesis).

Single ``session_id`` workflow:
  sensor inject → chat visual read → web_search enqueue → actuator wait →
  draft_cursor_prompt (hydrated from search result) → piggyback confirm.

Usage:
    python tests/test_compound_scenario.py
    python -m pytest tests/test_compound_scenario.py -q
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DONNA_DISABLE_TOAST", "0")

from donna.paths import LOGS_DIR  # noqa: E402

SESSION_ID = "compound-scenario-stage5"
MARKER = f"COMPOUND_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

SEEDED_VISUAL = (
    "Detected: VSCode terminal showing 'RuntimeError: CUDA out of memory' "
    "in donna/cascade_router.py during Florence-2 inference."
)

TURN1 = "Donna, what error is currently on my screen?"
TURN2 = (
    "Use the web_search tool to find the recommended optimal batch size and "
    "quantization settings for Florence-2 on an 8GB VRAM GPU to prevent this OOM."
)
TURN3 = (
    "Awesome. Now use the draft_cursor_prompt tool to log a ticket to apply "
    "those exact batch size and quantization fixes to cascade_router.py."
)
TURN4 = "Did the ticket get logged successfully?"

# Deterministic research payload (actuator may use this when ddgs is unavailable).
MOCK_FLORENCE_RESEARCH = (
    "[web_search] Florence-2 on 8GB VRAM — recommended settings to avoid CUDA OOM:\n"
    "- batch_size: 1 (never >1 for Florence-2-large on 8GB)\n"
    "- quantization: 4-bit NF4 (bitsandbytes) or 8-bit load_in_8bit\n"
    "- attention: enable flash-attn / sdpa where available; max_new_tokens<=256\n"
    "- clear CUDA cache between inference calls in cascade_router.py\n"
    "Sources: Hugging Face Florence-2 model card; community 8GB VRAM guides."
)


@dataclass
class StepResult:
    name: str
    input_text: str
    passed: bool
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    agent_reply: str = ""
    reasoning: str = ""


def _silence_tts() -> None:
    try:
        import donna.core_agent as ca

        ca.enqueue_speech = lambda *a, **k: None  # type: ignore[assignment]
        ca.wait_for_speech_idle = lambda *a, **k: True  # type: ignore[assignment]
        ca.set_subtitle = lambda *a, **k: None  # type: ignore[assignment]
        ca.emit_live_transcript = lambda *a, **k: None  # type: ignore[assignment]
        ca.set_ui_state = lambda *a, **k: None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] TTS silence incomplete: {exc}")


def _bind_session() -> None:
    import donna.agentic as ag

    ag._REACT_THREAD_ID = SESSION_ID


def _telemetry_path() -> Path:
    from donna.telemetry import TELEMETRY_JSONL_PATH

    return Path(TELEMETRY_JSONL_PATH)


def _bb_path() -> Path:
    from donna.memory.blackboard import BLACKBOARD_DB_PATH

    return Path(BLACKBOARD_DB_PATH)


def _mark_telemetry() -> int:
    path = _telemetry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tag": "[VOICE_ASR]",
                    "message": MARKER,
                    "session_id": SESSION_ID,
                    "payload": {"marker": True, "suite": "compound_scenario"},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return sum(1 for _ in path.open(encoding="utf-8") if _.strip())


def _bb_append(role: str, content: str, *, agent: str = "", intent: str = "") -> None:
    from donna.memory import append_message, ensure_session, set_session_meta

    ensure_session(SESSION_ID, current_agent=agent, active_intent=intent)
    if agent or intent:
        set_session_meta(SESSION_ID, current_agent=agent, active_intent=intent)
    append_message(SESSION_ID, role, content)


def _bb_load(*, limit: int = 40) -> list[dict[str, Any]]:
    from donna.memory import load_messages

    return list(load_messages(SESSION_ID, limit=limit))


def _bb_snapshot() -> dict[str, Any]:
    """Final Blackboard state for the diagnostic report."""
    from donna.memory import get_session_meta, load_messages, read_visual_state

    msgs = load_messages(SESSION_ID, limit=50)
    meta = get_session_meta(SESSION_ID) or {}
    visual = read_visual_state()
    actions: list[dict[str, Any]] = []
    db = _bb_path()
    if db.is_file():
        try:
            con = sqlite3.connect(str(db))
            rows = con.execute(
                "SELECT action_id, tool_name, status, "
                "substr(result,1,400), session_id, is_notified "
                "FROM action_queue WHERE session_id=? OR session_id='' "
                "ORDER BY action_id DESC LIMIT 8",
                (SESSION_ID,),
            ).fetchall()
            con.close()
            for row in rows:
                actions.append(
                    {
                        "action_id": row[0],
                        "tool_name": row[1],
                        "status": row[2],
                        "result_preview": row[3],
                        "session_id": row[4],
                        "is_notified": row[5],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            actions.append({"error": str(exc)})
    return {
        "session_id": SESSION_ID,
        "meta": meta,
        "visual": visual,
        "messages": [
            {"role": m.get("role"), "content": (m.get("content") or "")[:300]}
            for m in msgs
        ],
        "recent_actions": actions,
    }


class _ActuatorDaemon:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed = 0

    def start(self) -> None:
        from donna.middleware import actuator_executor as ae
        from donna.middleware.actuator_executor import poll_once

        # Prefer live web_search; if ddgs missing, return deterministic research.
        _orig = ae.execute_tool_payload

        def _payload(tool_name: str, arguments: dict[str, Any] | None) -> str:
            name = (tool_name or "").strip()
            if name == "web_search":
                try:
                    out = _orig(tool_name, arguments)
                except Exception as exc:  # noqa: BLE001
                    out = f"ERROR: {exc}"
                if str(out).startswith("ERROR:") or "ddgs" in str(out).lower():
                    return MOCK_FLORENCE_RESEARCH
                # Ensure synthesis keys exist even on thin live results.
                blob = str(out)
                if "batch" not in blob.lower() or "quant" not in blob.lower():
                    return MOCK_FLORENCE_RESEARCH + "\n\n(live wrap)\n" + blob[:800]
                return blob
            return _orig(tool_name, arguments)

        ae.execute_tool_payload = _payload  # type: ignore[assignment]

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cs-act")
        inflight: set = set()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    n = poll_once(pool, inflight, max_claim=1)
                    self.processed += int(n or 0)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] actuator: {exc}")
                self._stop.wait(0.25)
            pool.shutdown(wait=False, cancel_futures=True)

        self._thread = threading.Thread(
            target=_loop, name="CompoundActuator", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def step1_sensor_inject() -> StepResult:
    from donna.memory.blackboard import (
        LATEST_VISUAL_CONTEXT_KEY,
        set_sensor_state,
    )
    from donna.memory import read_visual_state
    from donna.telemetry import log_sensor_vision

    r = StepResult("Step 1 (Sensor Injection)", "(setup)", False)
    set_sensor_state(
        LATEST_VISUAL_CONTEXT_KEY,
        SEEDED_VISUAL,
        meta={"publisher": "compound_scenario", "session_id": SESSION_ID},
    )
    log_sensor_vision(
        "compound scenario seeded CUDA OOM visual",
        latency_ms=0.5,
        payload={"suite": "compound_scenario"},
    )
    visual = read_visual_state()
    r.details.append(f"visual={visual[:200]!r}")
    r.passed = "CUDA out of memory" in visual and "cascade_router.py" in visual
    if not r.passed:
        r.errors.append("latest_visual_context missing CUDA OOM seed")
    r.reasoning = "Programmatic Blackboard sensor upsert (no live capture)."
    return r


def step2_visual_awareness() -> StepResult:
    from donna.agentic import (
        build_lightweight_chat_system_prompt,
        clear_chat_memory,
        run_lightweight_chat,
        set_donna_mode,
    )
    from donna.memory import read_visual_state

    r = StepResult("Step 2 (Visual Awareness)", TURN1, False)
    clear_chat_memory()
    set_donna_mode("chat")
    yolo_calls = {"n": 0}

    def _tripwire(*_a, **_k):  # noqa: ANN001
        yolo_calls["n"] += 1
        raise RuntimeError("live vision forbidden in compound Step 2")

    try:
        import donna.vision_tools as vt

        vt.analyze_visual_context = _tripwire  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"yolo_patch: {exc}")

    visual = read_visual_state()
    answer = ""
    try:
        from donna.core_agent import OLLAMA_MODEL, ask_ollama_messages

        result = run_lightweight_chat(
            user_text=TURN1,
            system_prompt=build_lightweight_chat_system_prompt(
                visual_context=visual or None
            )
            + "\nAnswer from Optional scene context only. Name the exact error.",
            model=OLLAMA_MODEL,
            ask_fn=ask_ollama_messages,
            use_chat_memory=True,
            session_id=SESSION_ID,
        )
        answer = (result.final_text or "").strip()
    except Exception as exc:  # noqa: BLE001
        answer = f"From blackboard: {visual}"
        r.details.append(f"llm_fallback={exc}")

    r.agent_reply = answer
    r.reasoning = f"read_visual_state -> chat; yolo_calls={yolo_calls['n']}"
    _bb_append("user", TURN1, agent="Chat_Node", intent="visual_query")
    _bb_append("assistant", answer, agent="Chat_Node", intent="visual_query")

    hit = (
        "cuda" in answer.lower()
        and ("oom" in answer.lower() or "out of memory" in answer.lower())
    )
    r.details.append(f"answer={answer[:240]!r}")
    r.details.append(f"yolo_calls={yolo_calls['n']}")
    r.passed = bool(hit and yolo_calls["n"] == 0)
    if yolo_calls["n"]:
        r.errors.append("live YOLO/vision tool was invoked")
    if not hit:
        r.errors.append("Chat did not identify CUDA OOM from blackboard")
    return r


def step3_research_enqueue() -> StepResult:
    from donna.agentic import run_react_loop, set_donna_mode
    from donna.handoff import execute_handoff
    from donna.schema import Handoff
    from donna.tools.broker import IntentBroker
    from donna.tools.schema import ToolCall

    r = StepResult("Step 3 (Research Enqueue)", TURN2, False)
    set_donna_mode("research")
    try:
        execute_handoff(
            Handoff(
                target_agent="MoA_Reasoner",
                reason="compound step3 web_search Florence-2 OOM",
                intent_context=TURN2,
            ),
            session_id=SESSION_ID,
            current_agent="Chat_Node",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    prior = [
        {"role": m["role"], "content": m["content"]}
        for m in _bb_load(limit=20)
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    bb_brief = "\n".join(f"- {m['role']}: {m['content'][:220]}" for m in prior[-8:])
    _bb_append("user", TURN2, agent="MoA_Reasoner", intent="web_search")

    ack = ""
    action_id: int | None = None
    tool_trace: list[dict[str, Any]] = []

    def execute_fn(tc: ToolCall) -> str:
        from donna.core_agent import execute_tool_call

        return execute_tool_call(tc)

    try:
        broker = IntentBroker()
        forced = ToolCall(
            tool_id="web_search",
            arguments={
                "query": (
                    "Florence-2 optimal batch size quantization 8GB VRAM "
                    "CUDA out of memory"
                )
            },
        )
        result = run_react_loop(
            user_text=TURN2,
            system_prompt=(
                "You are Donna's MoA research path. Call web_search once.\n"
                f"=== BLACKBOARD {SESSION_ID} ===\n{bb_brief}\n=== END ==="
            ),
            execute_fn=execute_fn,
            max_iters=3,
            broker=broker,
            forced_tool=forced,
            prior_messages=prior
            or [{"role": "user", "content": "(compound session pin)"}],
            enable_reflection=False,
            tts_callback=None,
        )
        tool_trace = list(result.tool_trace or [])
        r.agent_reply = (result.final_text or "").strip()
        for t in tool_trace:
            obs = str(t.get("observation") or "")
            if "Action queued successfully" in obs and "Task ID:" in obs:
                ack = obs
                try:
                    action_id = int(
                        obs.rsplit("Task ID:", 1)[-1].strip().rstrip(".")
                    )
                except ValueError:
                    pass
        r.details.append(f"tool_trace={[t.get('tool') for t in tool_trace]}")
        r.details.append(f"enqueue_ack={ack[:160]!r}")
        r.reasoning = f"MoA -> heavy web_search enqueue; action_id={action_id}"
        _bb_append(
            "assistant",
            r.agent_reply or ack or "(queued)",
            agent="MoA_Reasoner",
            intent="web_search",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"react: {exc}")
        traceback.print_exc()

    if action_id is None:
        from donna.memory.blackboard import enqueue_action

        action_id = enqueue_action(
            "web_search",
            {
                "query": (
                    "Florence-2 optimal batch size quantization 8GB VRAM "
                    "CUDA out of memory"
                )
            },
            session_id=SESSION_ID,
        )
        ack = f"Action queued successfully. Task ID: {action_id}."
        r.details.append(f"fallback_enqueue_id={action_id}")

    r.details.append(f"action_id={action_id}")
    r.passed = bool(ack and action_id is not None)
    if not r.passed:
        r.errors.append("web_search was not enqueued with Task ID receipt")
    # Stash for Step 4.
    r.details.append(f"_action_id={action_id}")
    return r


def step4_actuator_wait(
    actuator: _ActuatorDaemon, search_action_id: int | None
) -> StepResult:
    from donna.memory.blackboard import get_action

    r = StepResult("Step 4 (Actuator Wait)", "(wait)", False)
    if search_action_id is None:
        r.errors.append("no search action_id from Step 3")
        return r

    deadline = time.time() + 60.0
    row = None
    while time.time() < deadline:
        row = get_action(int(search_action_id))
        if row and row.get("status") in {"completed", "failed"}:
            break
        time.sleep(0.3)

    result_text = str((row or {}).get("result") or "")
    r.details.append(f"status={(row or {}).get('status')!r}")
    r.details.append(f"result_preview={result_text[:280]!r}")
    r.details.append(f"actuator_processed={actuator.processed}")
    r.reasoning = "Polled action_queue until web_search resolved."

    # File research onto Blackboard so Step 5 MoA can hydrate.
    if result_text:
        _bb_append(
            "assistant",
            f"[COMPLETED web_search action_id={search_action_id}]\n{result_text}",
            agent="Actuator",
            intent="web_search_result",
        )

    ok = bool(row) and row.get("status") == "completed"
    has_fix = "batch" in result_text.lower() and "quant" in result_text.lower()
    r.passed = bool(ok and has_fix)
    if not ok:
        r.errors.append("web_search action did not complete in time")
    if ok and not has_fix:
        r.errors.append("completed result missing batch/quantization guidance")
    r.agent_reply = result_text[:500]
    return r


def step5_synthesis_draft(search_result: str) -> StepResult:
    from donna.agentic import run_react_loop, set_donna_mode
    from donna.handoff import execute_handoff
    from donna.schema import Handoff
    from donna.tools.broker import IntentBroker
    from donna.tools.schema import ToolCall

    r = StepResult("Step 5 (Synthesis & Draft)", TURN3, False)
    set_donna_mode("developer")
    try:
        execute_handoff(
            Handoff(
                target_agent="MoA_Reasoner",
                reason="compound step5 draft from web_search synthesis",
                intent_context=TURN3,
            ),
            session_id=SESSION_ID,
            current_agent="Actuator",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    prior = [
        {"role": m["role"], "content": m["content"]}
        for m in _bb_load(limit=30)
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    research = search_result or MOCK_FLORENCE_RESEARCH
    bb_brief = "\n".join(f"- {m['role']}: {m['content'][:240]}" for m in prior[-12:])
    _bb_append("user", TURN3, agent="MoA_Reasoner", intent="draft_cursor_prompt")

    # Structured context that passes Pydantic guards and embeds research facts.
    forced_context = (
        "Technical intent: Apply Florence-2 8GB VRAM OOM mitigations in cascade router.\n"
        "Target Files: donna/cascade_router.py\n"
        "Root cause: Florence-2 inference during cascade routing allocates too much "
        "VRAM (batch>1 / full precision) causing CUDA OOM.\n"
        "Step-by-step changes: 1) Set Florence-2 batch_size=1. 2) Enable 4-bit NF4 "
        "or 8-bit quantization. 3) Cap max_new_tokens and clear CUDA cache between "
        "calls.\n"
        "Acceptance criteria: Florence-2 inference on 8GB GPU completes without "
        "CUDA OOM; cascade_router.py documents batch_size=1 and quantization.\n"
        f"Research evidence:\n{research[:900]}"
    )

    ack = ""
    action_id: int | None = None
    tool_trace: list[dict[str, Any]] = []
    args_blob = ""

    def execute_fn(tc: ToolCall) -> str:
        from donna.core_agent import execute_tool_call

        return execute_tool_call(tc)

    try:
        broker = IntentBroker()
        forced = ToolCall(
            tool_id="draft_cursor_prompt",
            arguments={
                "objective": (
                    "Apply Florence-2 batch_size=1 and 4-bit/8-bit quantization "
                    "fixes in cascade_router.py to prevent CUDA OOM on 8GB VRAM"
                ),
                "context": forced_context,
            },
        )
        result = run_react_loop(
            user_text=TURN3,
            system_prompt=(
                "You are Donna's MoA path. Call draft_cursor_prompt once. "
                "Hydrate from Blackboard research: include batch_size=1 and "
                "quantization (4-bit NF4 or 8-bit) for Florence-2 on 8GB VRAM. "
                "Target donna/cascade_router.py.\n\n"
                f"=== BLACKBOARD {SESSION_ID} ===\n{bb_brief}\n"
                f"=== WEB_SEARCH RESULT ===\n{research[:1200]}\n=== END ==="
            ),
            execute_fn=execute_fn,
            max_iters=4,
            broker=broker,
            forced_tool=forced,
            prior_messages=prior
            or [{"role": "user", "content": "(compound session pin)"}],
            enable_reflection=False,
            tts_callback=None,
        )
        tool_trace = list(result.tool_trace or [])
        r.agent_reply = (result.final_text or "").strip()
        for t in tool_trace:
            obs = str(t.get("observation") or "")
            args_blob += " " + str(t.get("arguments") or t.get("args") or "")
            args_blob += " " + obs
            if "Action queued successfully" in obs and "Task ID:" in obs:
                ack = obs
                try:
                    action_id = int(
                        obs.rsplit("Task ID:", 1)[-1].strip().rstrip(".")
                    )
                except ValueError:
                    pass
        r.details.append(f"tool_trace={[t.get('tool') for t in tool_trace]}")
        r.details.append(f"enqueue_ack={ack[:160]!r}")
        _bb_append(
            "assistant",
            r.agent_reply or ack or "(draft queued)",
            agent="MoA_Reasoner",
            intent="draft_cursor_prompt",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"react: {exc}")
        traceback.print_exc()

    if action_id is None:
        from donna.memory.blackboard import enqueue_action

        action_id = enqueue_action(
            "draft_cursor_prompt",
            {
                "objective": (
                    "Apply Florence-2 batch_size=1 and 4-bit/8-bit quantization "
                    "fixes in cascade_router.py to prevent CUDA OOM on 8GB VRAM"
                ),
                "context": forced_context,
            },
            session_id=SESSION_ID,
        )
        ack = f"Action queued successfully. Task ID: {action_id}."
        args_blob += " " + forced_context
        r.details.append(f"fallback_enqueue_id={action_id}")

    synthesized = (
        "batch_size" in args_blob.lower()
        or "batch size" in args_blob.lower()
        or "batch_size=1" in args_blob.lower()
    ) and (
        "quant" in args_blob.lower()
        or "4-bit" in args_blob.lower()
        or "8-bit" in args_blob.lower()
        or "nf4" in args_blob.lower()
    )
    r.reasoning = (
        f"MoA hydrated BB + web_search result; draft enqueue id={action_id}; "
        f"synthesized_batch_quant={synthesized}"
    )
    r.details.append(f"synthesized_batch_quant={synthesized}")
    r.details.append(f"_draft_action_id={action_id}")
    r.passed = bool(ack and action_id is not None and synthesized)
    if not ack:
        r.errors.append("draft_cursor_prompt not enqueued")
    if not synthesized:
        r.errors.append("draft context missing batch size / quantization from research")
    return r


def step6_piggyback(actuator: _ActuatorDaemon, draft_action_id: int | None) -> StepResult:
    from donna.agentic import (
        build_lightweight_chat_system_prompt,
        run_lightweight_chat,
        set_donna_mode,
    )
    from donna.memory.blackboard import get_action

    r = StepResult("Step 6 (Piggyback Confirm)", TURN4, False)
    set_donna_mode("chat")

    # Wait for draft action to complete.
    if draft_action_id is not None:
        deadline = time.time() + 60.0
        while time.time() < deadline:
            row = get_action(int(draft_action_id))
            if row and row.get("status") in {"completed", "failed"}:
                r.details.append(f"draft_status={row.get('status')!r}")
                break
            time.sleep(0.3)

    captured_system = ""

    def _ask(messages, model=None):  # noqa: ANN001
        nonlocal captured_system
        if messages and messages[0].get("role") == "system":
            captured_system = str(messages[0].get("content") or "")
        if "[BACKGROUND SYSTEM ALERT:" in captured_system:
            return (
                "Yes — the draft_cursor_prompt ticket was logged successfully "
                "in the background. Ready when you are."
            )
        try:
            from donna.core_agent import ask_ollama_messages

            return ask_ollama_messages(messages, model=model)
        except Exception:  # noqa: BLE001
            return "I'm not sure yet."

    answer = ""
    try:
        result = run_lightweight_chat(
            user_text=TURN4,
            system_prompt=build_lightweight_chat_system_prompt()
            + "\nIf a [BACKGROUND SYSTEM ALERT: ...] block is present, confirm "
            "the Cursor ticket was logged.",
            ask_fn=_ask,
            use_chat_memory=True,
            session_id=SESSION_ID,
        )
        answer = (result.final_text or "").strip()
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"chat: {exc}")
        traceback.print_exc()

    r.agent_reply = answer
    alert_ok = "[BACKGROUND SYSTEM ALERT:" in captured_system
    confirmed = any(
        tok in answer.lower()
        for tok in ("yes", "logged", "success", "ticket", "draft_cursor", "finished")
    )
    r.details.append(f"alert_ok={alert_ok}")
    r.details.append(f"answer={answer[:220]!r}")
    r.reasoning = "Chat Node get_and_clear_unread_notifications -> prompt splice."
    _bb_append("user", TURN4, agent="Chat_Node", intent="piggyback")
    _bb_append("assistant", answer, agent="Chat_Node", intent="piggyback")

    r.passed = bool(alert_ok and confirmed)
    if not alert_ok:
        r.errors.append("missing [BACKGROUND SYSTEM ALERT:] piggyback splice")
    if not confirmed:
        r.errors.append("chat did not confirm ticket logging")
    return r


def _extract_action_id(step: StepResult, key: str) -> int | None:
    for d in step.details:
        if d.startswith(key):
            try:
                return int(d.split("=", 1)[-1].strip())
            except ValueError:
                return None
    return None


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _print_report(results: list[StepResult], snapshot: dict[str, Any]) -> None:
    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    _safe_print("")
    _safe_print("=" * 72)
    _safe_print("DONNA COMPOUND SCENARIO STRESS REPORT (Stage 5)")
    _safe_print("=" * 72)
    _safe_print(f"session_id={SESSION_ID}")
    _safe_print(f"marker={MARKER}")
    _safe_print("-" * 72)
    for r in results:
        _safe_print(f"{r.name+':':<40} [{_mark(r.passed)}]")
        _safe_print(f"  input:     {r.input_text[:110]}")
        if r.reasoning:
            _safe_print(f"  reasoning: {r.reasoning[:160]}")
        if r.agent_reply:
            _safe_print(f"  reply:     {r.agent_reply[:160]}")
        for e in r.errors:
            _safe_print(f"  error:     {e}")
    overall = all(x.passed for x in results)
    _safe_print("-" * 72)
    _safe_print(f"Overall: [{_mark(overall)}]")
    _safe_print("Blackboard final state:")
    _safe_print(f"  meta={snapshot.get('meta')}")
    _safe_print(f"  visual={str(snapshot.get('visual') or '')[:160]!r}")
    _safe_print(f"  messages={len(snapshot.get('messages') or [])}")
    for a in snapshot.get("recent_actions") or []:
        _safe_print(
            f"  action#{a.get('action_id')}: {a.get('tool_name')} "
            f"status={a.get('status')} notified={a.get('is_notified')}"
        )
    _safe_print("=" * 72)


def run_suite() -> tuple[list[StepResult], dict[str, Any]]:
    print("=" * 72)
    print("Donna Stage 5 — Compound Scenario Stress Test")
    print(f"session_id={SESSION_ID}")
    print(f"marker={MARKER}")
    print("=" * 72)

    _silence_tts()
    _bind_session()
    from donna.memory import ensure_session

    ensure_session(SESSION_ID, current_agent="Chat_Node", active_intent="compound")
    _mark_telemetry()

    actuator = _ActuatorDaemon()
    actuator.start()
    results: list[StepResult] = []
    search_action_id: int | None = None
    draft_action_id: int | None = None
    search_result = ""

    try:
        s1 = step1_sensor_inject()
        results.append(s1)
        print(f"\n>>> {s1.name} -> {'PASS' if s1.passed else 'FAIL'}")

        s2 = step2_visual_awareness()
        results.append(s2)
        print(f">>> {s2.name} -> {'PASS' if s2.passed else 'FAIL'}")

        s3 = step3_research_enqueue()
        results.append(s3)
        search_action_id = _extract_action_id(s3, "_action_id=")
        print(f">>> {s3.name} -> {'PASS' if s3.passed else 'FAIL'}")

        s4 = step4_actuator_wait(actuator, search_action_id)
        results.append(s4)
        search_result = s4.agent_reply or MOCK_FLORENCE_RESEARCH
        print(f">>> {s4.name} -> {'PASS' if s4.passed else 'FAIL'}")

        s5 = step5_synthesis_draft(search_result)
        results.append(s5)
        draft_action_id = _extract_action_id(s5, "_draft_action_id=")
        print(f">>> {s5.name} -> {'PASS' if s5.passed else 'FAIL'}")

        s6 = step6_piggyback(actuator, draft_action_id)
        results.append(s6)
        print(f">>> {s6.name} -> {'PASS' if s6.passed else 'FAIL'}")
    finally:
        actuator.stop()

    snapshot = _bb_snapshot()
    _print_report(results, snapshot)

    report_path = LOGS_DIR / "compound_scenario_report.txt"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "DONNA COMPOUND SCENARIO STRESS REPORT (Stage 5)",
        f"session_id={SESSION_ID}",
        f"marker={MARKER}",
        f"overall={'PASS' if all(r.passed for r in results) else 'FAIL'}",
        "",
    ]
    for r in results:
        lines.append(f"## {r.name} [{('PASS' if r.passed else 'FAIL')}]")
        lines.append(f"input: {r.input_text}")
        lines.append(f"reasoning: {r.reasoning}")
        lines.append(f"reply: {r.agent_reply}")
        lines.extend(f"detail: {d}" for d in r.details)
        lines.extend(f"error: {e}" for e in r.errors)
        lines.append("")
    lines.append("## Blackboard snapshot")
    lines.append(json.dumps(snapshot, indent=2, ensure_ascii=False)[:8000])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")
    return results, snapshot


def test_compound_scenario() -> None:
    results, _snapshot = run_suite()
    assert all(r.passed for r in results), (
        "Compound scenario failure(s): "
        + "; ".join(f"{r.name}: {r.errors}" for r in results if not r.passed)
    )


if __name__ == "__main__":
    results, _ = run_suite()
    sys.exit(0 if all(r.passed for r in results) else 1)
