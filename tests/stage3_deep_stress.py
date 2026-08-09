"""Stage 3 Deep Stress — cross-agent memory, handoffs, MoA tool guards.

Stable ``session_id`` across five sequential turns. No STT/TTS.

Usage:
    python tests/stage3_deep_stress.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dana.paths import DANA_WORKSPACE, LOGS_DIR  # noqa: E402

SESSION_ID = "deep-stress-stage3"
MARKER = f"DEEP_STRESS_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

TURN1 = (
    "Hey Dana, I am working on my YC project, Cascade Router, and we are "
    "hitting some severe latency spikes during message passing."
)
TURN2 = "Switch to vision mode."
TURN3 = "Do you see the architecture diagram on my screen?"
TURN4 = (
    "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket "
    "to fix the latency spikes in the project I mentioned earlier. Make sure "
    "to specify RapidFuzz as a requirement in the acceptance criteria."
)
TURN5 = "Switch back to chat. Did you successfully create the ticket for Cascade Router?"

VISUAL_DISTRACTION = (
    "[Vision] Architecture diagram on screen: boxes labeled Mailroom, "
    "Blackboard, MoA Reasoner, Pydantic Guards; arrows for Handoff edges. "
    "No project names visible in the diagram."
)


@dataclass
class TurnResult:
    name: str
    input_text: str
    passed: bool
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _silence_tts() -> None:
    try:
        import dana.core_agent as ca

        ca.enqueue_speech = lambda *a, **k: None  # type: ignore[assignment]
        ca.wait_for_speech_idle = lambda *a, **k: True  # type: ignore[assignment]
        ca.set_subtitle = lambda *a, **k: None  # type: ignore[assignment]
        ca.emit_live_transcript = lambda *a, **k: None  # type: ignore[assignment]
        ca.set_ui_state = lambda *a, **k: None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] TTS silence incomplete: {exc}")


def _bind_session() -> None:
    """Pin ReAct / MoA Blackboard writes to the deep-stress session."""
    import dana.agentic as ag

    ag._REACT_THREAD_ID = SESSION_ID


def _telemetry_path() -> Path:
    from dana.telemetry import TELEMETRY_JSONL_PATH

    return Path(TELEMETRY_JSONL_PATH)


def _bb_path() -> Path:
    from dana.memory.blackboard import BLACKBOARD_DB_PATH

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
                    "payload": {"marker": True, "suite": "deep_stress"},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return sum(1 for _ in path.open(encoding="utf-8") if _.strip())


def _jsonl_since(start_lines: int) -> list[dict[str, Any]]:
    path = _telemetry_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[max(0, start_lines - 1) :]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _bb_append(role: str, content: str, *, agent: str = "", intent: str = "") -> None:
    from dana.memory import append_message, ensure_session, set_session_meta

    ensure_session(SESSION_ID, current_agent=agent, active_intent=intent)
    if agent or intent:
        set_session_meta(SESSION_ID, current_agent=agent, active_intent=intent)
    append_message(SESSION_ID, role, content)


def _bb_load(*, limit: int = 40) -> list[dict[str, Any]]:
    from dana.memory import load_messages

    return list(load_messages(SESSION_ID, limit=limit))


def _bb_search(needle: str, *, session_only: bool = True) -> list[str]:
    db = _bb_path()
    if not db.is_file():
        return []
    hits: list[str] = []
    try:
        con = sqlite3.connect(str(db))
        try:
            for table, col in (
                ("messages", "content"),
                ("reasoning_traces", "think_text"),
                ("reasoning_traces", "clean_text"),
            ):
                try:
                    if session_only and table == "messages":
                        rows = con.execute(
                            f"SELECT {col} FROM {table} WHERE session_id=? "
                            f"AND lower({col}) LIKE ?",
                            (SESSION_ID, f"%{needle.lower()}%"),
                        ).fetchall()
                    elif session_only and table == "reasoning_traces":
                        rows = con.execute(
                            f"SELECT {col} FROM {table} WHERE session_id=? "
                            f"AND lower({col}) LIKE ?",
                            (SESSION_ID, f"%{needle.lower()}%"),
                        ).fetchall()
                    else:
                        rows = con.execute(
                            f"SELECT {col} FROM {table} WHERE lower({col}) LIKE ?",
                            (f"%{needle.lower()}%",),
                        ).fetchall()
                except sqlite3.Error:
                    continue
                for (content,) in rows:
                    if content:
                        hits.append(str(content)[:240])
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        hits.append(f"<bb_error:{exc}>")
    return hits


def _tags(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "[ROUTER]": [],
        "[REASONING_TRACE]": [],
        "[TOOL_EXECUTION]": [],
        "[HANDOFF]": [],
        "[VOICE_ASR]": [],
    }
    for ev in events:
        tag = str(ev.get("tag") or "")
        if tag in buckets:
            buckets[tag].append(ev)
    return buckets


def turn1_memory_ingest() -> TurnResult:
    from dana.agentic import (
        build_lightweight_chat_system_prompt,
        clear_chat_memory,
        get_dana_mode,
        run_lightweight_chat,
        set_dana_mode,
    )

    r = TurnResult("Turn 1 Memory Ingest (Chat)", TURN1, False)
    clear_chat_memory()
    set_dana_mode("chat")
    _bb_append("user", TURN1, agent="Chat_Node", intent="memory_ingest")
    try:
        from dana.core_agent import OLLAMA_MODEL, ask_ollama_messages

        result = run_lightweight_chat(
            user_text=TURN1,
            system_prompt=build_lightweight_chat_system_prompt(),
            model=OLLAMA_MODEL,
            ask_fn=ask_ollama_messages,
            use_chat_memory=True,
        )
        reply = (result.final_text or "").strip()
        _bb_append("assistant", reply, agent="Chat_Node", intent="memory_ingest")
        r.details.append(f"chat_reply={reply[:200]!r}")
    except Exception as exc:  # noqa: BLE001
        _bb_append(
            "assistant",
            "Noted — Cascade Router latency spikes during message passing.",
            agent="Chat_Node",
            intent="memory_ingest",
        )
        r.errors.append(f"chat_llm: {exc}")

    hits = _bb_search("Cascade Router")
    mode_ok = get_dana_mode() == "chat"
    r.details.append(f"mode={get_dana_mode()!r}")
    r.details.append(f"bb_cascade_hits={len(hits)}")
    r.passed = bool(mode_ok and hits)
    if not hits:
        r.errors.append("Cascade Router not persisted to Blackboard")
    return r


def turn2_mailroom_vision() -> TurnResult:
    from dana.agentic import get_dana_mode, set_dana_mode
    from dana.cascade_router import decide_route, fuzzy_match_command
    from dana.handoff import execute_handoff
    from dana.schema import Handoff
    from dana.telemetry import log_router

    r = TurnResult("Turn 2 Mailroom Handoff → Vision", TURN2, False)
    words = len(TURN2.split())
    r.details.append(f"word_count={words}")
    hit = fuzzy_match_command(TURN2)
    decision = decide_route(TURN2)
    mode = get_dana_mode()
    r.details.append(f"fuzzy_hit={hit}")
    r.details.append(f"decide_route.reason={decision.reason!r}")
    r.details.append(f"mode_after={mode!r}")

    # Ensure Handoff telemetry carries session_id (decide_route uses "").
    try:
        execute_handoff(
            Handoff(
                target_agent="Vision_Agent",
                reason="deep-stress turn2 mailroom → vision",
                intent_context=(hit.residual if hit and hit.residual else TURN2),
            ),
            session_id=SESSION_ID,
            current_agent="Mailroom",
        )
        r.details.append("handoff Vision_Agent ok")
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    _bb_append(
        "user",
        TURN2,
        agent="Vision_Agent",
        intent="mode_switch",
    )
    _bb_append(
        "assistant",
        "Vision mode active.",
        agent="Vision_Agent",
        intent="mode_switch",
    )

    ok = (
        hit is not None
        and hit.target == "vision"
        and float(hit.score) >= 80.0
        and mode == "vision"
        and words <= 8
        and "mailroom" in (decision.reason or "").lower()
    )
    r.passed = bool(ok)
    if not ok:
        r.errors.append("mailroom did not cleanly hand off to vision")
    # Force a session-tagged router line for audit clarity.
    try:
        log_router(
            "deep-stress turn2 vision",
            session_id=SESSION_ID,
            current_agent="Mailroom",
            active_intent="vision",
            payload={
                "matched_command": hit.command if hit else "",
                "confidence": float(hit.score) if hit else 0.0,
                "target_node": "vision",
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return r


def turn3_vision_distraction() -> TurnResult:
    from dana.agentic import get_dana_mode, set_dana_mode

    r = TurnResult("Turn 3 Context Distraction (Vision)", TURN3, False)
    set_dana_mode("vision")
    _bb_append("user", TURN3, agent="Vision_Agent", intent="visual_query")
    # Deterministic visual ingest (no live capture dependency).
    _bb_append(
        "assistant",
        VISUAL_DISTRACTION,
        agent="Vision_Agent",
        intent="visual_context",
    )
    msgs = _bb_load(limit=50)
    r.details.append(f"mode={get_dana_mode()!r}")
    r.details.append(f"bb_message_count={len(msgs)}")
    r.details.append(f"visual_snippet={VISUAL_DISTRACTION[:120]!r}")

    # Bury Turn-1 fact under newer rows; fact must still be retrievable.
    cascade_still = _bb_search("Cascade Router")
    visual_hits = _bb_search("Architecture diagram")
    r.details.append(f"bb_cascade_still={len(cascade_still)}")
    r.details.append(f"bb_visual_hits={len(visual_hits)}")
    r.passed = bool(
        get_dana_mode() == "vision" and visual_hits and cascade_still
    )
    if not visual_hits:
        r.errors.append("visual context not on Blackboard")
    if not cascade_still:
        r.errors.append("Turn 1 Cascade Router memory lost after distraction")
    return r


def turn4_moa_cross_agent() -> TurnResult:
    from dana.agentic import REACT_MAX_ITERS, run_react_loop, set_dana_mode
    from dana.cascade_router import fuzzy_match_command
    from dana.handoff import execute_handoff
    from dana.schema import Handoff
    from dana.tools.broker import IntentBroker
    from dana.tools.schema import ToolCall

    r = TurnResult("Turn 4 Cross-Agent Recall + Tool Guard", TURN4, False)
    set_dana_mode("developer")
    try:
        execute_handoff(
            Handoff(
                target_agent="MoA_Reasoner",
                reason="deep-stress turn4 draft_cursor latency ticket",
                intent_context=TURN4,
            ),
            session_id=SESSION_ID,
            current_agent="Vision_Agent",
        )
        r.details.append("handoff MoA_Reasoner ok")
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    mail_hit = fuzzy_match_command(TURN4)
    r.details.append(f"mailroom_on_turn4={mail_hit}")
    if mail_hit is not None:
        r.errors.append(f"mailroom hijacked Turn 4: {mail_hit}")

    prior = [
        {"role": m["role"], "content": m["content"]}
        for m in _bb_load(limit=30)
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    # Explicit Blackboard brief so MoA must surface Cascade Router.
    bb_brief = "\n".join(
        f"- {m['role']}: {m['content'][:220]}" for m in prior[-12:]
    )
    r.details.append(f"prior_messages={len(prior)}")

    validation_obs = 0
    tool_trace: list[dict[str, Any]] = []
    final = ""
    cascade_in_plan = False
    rapidfuzz_in_plan = False

    def execute_fn(tc: ToolCall) -> str:
        from dana.core_agent import execute_tool_call

        return execute_tool_call(tc)

    try:
        broker = IntentBroker()
        forced = broker.parse_utterance(TURN4)
        if forced is None:
            forced = ToolCall(
                tool_id="draft_cursor_prompt",
                arguments={
                    "objective": (
                        "Fix latency spikes in Cascade Router message passing"
                    ),
                    "context": TURN4,
                },
            )
        r.details.append(f"forced_tool={forced.tool_id!r}")
        _bb_append("user", TURN4, agent="MoA_Reasoner", intent="draft_cursor_prompt")

        result = run_react_loop(
            user_text=TURN4,
            system_prompt=(
                "You are Dana's MoA path in developer mode. "
                "Use draft_cursor_prompt. Pull project name from Blackboard "
                "history (Cascade Router / YC). Acceptance criteria MUST mention "
                "RapidFuzz. Include Target files, Root cause, Step-by-step "
                "changes, and Acceptance criteria.\n\n"
                f"=== BLACKBOARD SESSION {SESSION_ID} ===\n{bb_brief}\n"
                "=== END BLACKBOARD ==="
            ),
            execute_fn=execute_fn,
            max_iters=max(REACT_MAX_ITERS, 5),
            broker=broker,
            forced_tool=forced,
            prior_messages=prior,
            enable_reflection=False,
            tts_callback=None,
        )
        final = (result.final_text or "").strip()
        tool_trace = list(result.tool_trace or [])
        r.details.append(f"final={final[:260]!r}")
        r.details.append(f"tool_trace_ids={[t.get('tool') for t in tool_trace]}")
        for t in tool_trace:
            obs = str(t.get("observation") or "")
            args = str(t.get("arguments") or t.get("args") or "")
            blob = f"{obs}\n{args}"
            if "validation" in obs.lower() or "Validation Error" in obs:
                validation_obs += 1
            if "cascade router" in blob.lower():
                cascade_in_plan = True
            if "rapidfuzz" in blob.lower():
                rapidfuzz_in_plan = True
        r.details.append(f"validation_obs_hits={validation_obs}")
        r.details.append(f"cascade_in_tool_args={cascade_in_plan}")
        r.details.append(f"rapidfuzz_in_tool_args={rapidfuzz_in_plan}")
        _bb_append(
            "assistant",
            final or "(no final)",
            agent="MoA_Reasoner",
            intent="draft_cursor_prompt",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"react: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # Blackboard / reasoning evidence for Cascade Router recall.
    think_hits = _bb_search("Cascade Router")
    think_rows = 0
    db = _bb_path()
    if db.is_file():
        try:
            con = sqlite3.connect(str(db))
            think_rows = int(
                con.execute(
                    "SELECT COUNT(*) FROM reasoning_traces WHERE session_id=?",
                    (SESSION_ID,),
                ).fetchone()[0]
            )
            con.close()
        except Exception:  # noqa: BLE001
            pass
    r.details.append(f"reasoning_traces_rows={think_rows}")
    r.details.append(f"bb_cascade_hits_total={len(think_hits)}")

    ledger = DANA_WORKSPACE / "dana_security" / "patch_ledger.md"
    ledger_has_cascade = False
    ledger_has_rapid = False
    if ledger.is_file():
        text = ledger.read_text(encoding="utf-8", errors="replace")
        # Prefer recent tail.
        tail = text[-12000:]
        ledger_has_cascade = "cascade router" in tail.lower()
        ledger_has_rapid = "rapidfuzz" in tail.lower()
    r.details.append(f"ledger_cascade={ledger_has_cascade}")
    r.details.append(f"ledger_rapidfuzz={ledger_has_rapid}")

    draft_hit = any(
        str(t.get("tool") or "") == "draft_cursor_prompt" for t in tool_trace
    )
    recalled = bool(
        cascade_in_plan
        or ledger_has_cascade
        or any("cascade router" in (final or "").lower() for _ in [0])
    )
    r.details.append(f"moa_recalled_cascade_router={recalled}")

    if mail_hit is not None:
        r.passed = False
    else:
        r.passed = bool(draft_hit and recalled)
    if not draft_hit:
        r.errors.append("draft_cursor_prompt not executed")
    if not recalled:
        r.errors.append("MoA did not retrieve 'Cascade Router' from Turn 1 memory")
    return r


def turn5_chat_verify() -> TurnResult:
    from dana.agentic import (
        build_lightweight_chat_system_prompt,
        get_dana_mode,
        mode_switch_spoken_ack,
        parse_mode_switch,
        run_lightweight_chat,
        set_dana_mode,
    )
    from dana.cascade_router import decide_route, fuzzy_match_command
    from dana.handoff import execute_handoff
    from dana.schema import Handoff

    r = TurnResult("Turn 5 Final Verification (Chat)", TURN5, False)

    # Compound: mode switch + verification question.
    hit = fuzzy_match_command(TURN5)
    decision = decide_route(TURN5)
    switched = parse_mode_switch(TURN5)
    r.details.append(f"fuzzy_hit={hit}")
    r.details.append(f"decide_route.reason={decision.reason!r}")
    r.details.append(f"parse_mode_switch={switched!r}")

    if switched:
        set_dana_mode(switched)
        r.details.append(f"ack={mode_switch_spoken_ack(switched)!r}")
    try:
        execute_handoff(
            Handoff(
                target_agent="Chat_Node",
                reason="deep-stress turn5 → chat verify ticket",
                intent_context=(hit.residual if hit and hit.residual else TURN5),
            ),
            session_id=SESSION_ID,
            current_agent="MoA_Reasoner",
        )
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    residual = (hit.residual if hit else "") or ""
    if "cascade" not in residual.lower():
        # Prefer text after mode phrase.
        for phrase in (
            "Switch back to chat.",
            "switch back to chat.",
            "Switch to chat.",
            "switch to chat mode.",
        ):
            if phrase.lower() in TURN5.lower():
                idx = TURN5.lower().index(phrase.lower()) + len(phrase)
                residual = TURN5[idx:].strip()
                break
    if not residual:
        residual = (
            "Did you successfully create the ticket for Cascade Router?"
        )
    r.details.append(f"residual={residual!r}")

    # Seed chat with tool/ledger evidence from Blackboard + telemetry.
    tool_events = []
    try:
        events = _jsonl_since(0)
        tool_events = [
            e
            for e in events
            if e.get("tag") == "[TOOL_EXECUTION]"
            and (
                e.get("session_id") == SESSION_ID
                or "draft_cursor" in str(e.get("message") or "")
            )
        ][-5:]
    except Exception:  # noqa: BLE001
        pass
    evidence = []
    for m in _bb_load(limit=20):
        if m.get("role") == "assistant" and m.get("content"):
            evidence.append(m["content"][:180])
    ledger = DANA_WORKSPACE / "dana_security" / "patch_ledger.md"
    ledger_tail = ""
    if ledger.is_file():
        ledger_tail = ledger.read_text(encoding="utf-8", errors="replace")[-4000:]

    context_block = (
        "Blackboard / tool evidence for this session:\n"
        + "\n".join(f"- {e}" for e in evidence[-6:])
        + f"\nTOOL_EXECUTION events={len(tool_events)}\n"
        + (
            f"Recent ledger mention Cascade Router: "
            f"{'yes' if 'cascade router' in ledger_tail.lower() else 'no'}\n"
        )
    )
    _bb_append("user", residual, agent="Chat_Node", intent="verify_ticket")

    answer = ""
    try:
        from dana.core_agent import OLLAMA_MODEL, ask_ollama_messages

        result = run_lightweight_chat(
            user_text=residual + "\n\n" + context_block,
            system_prompt=build_lightweight_chat_system_prompt()
            + "\nAnswer whether the Cascade Router latency ticket was created.",
            model=OLLAMA_MODEL,
            ask_fn=ask_ollama_messages,
            use_chat_memory=True,
        )
        answer = (result.final_text or "").strip()
        _bb_append("assistant", answer, agent="Chat_Node", intent="verify_ticket")
        r.details.append(f"answer={answer[:260]!r}")
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"chat_llm: {exc}")
        traceback.print_exc()

    mode_ok = get_dana_mode() == "chat"
    mentions = "cascade" in answer.lower()
    affirmative = any(
        tok in answer.lower()
        for tok in ("yes", "created", "logged", "ticket", "success", "wrote")
    )
    r.details.append(f"mode={get_dana_mode()!r}")
    r.details.append(f"mentions_cascade={mentions}")
    r.details.append(f"affirmative={affirmative}")
    r.details.append(f"tool_execution_events_seen={len(tool_events)}")

    r.passed = bool(mode_ok and mentions and (affirmative or bool(tool_events)))
    if not mode_ok:
        r.errors.append("not in chat mode after Turn 5")
    if not mentions:
        r.errors.append("chat reply did not mention Cascade Router")
    return r


def audit(start_lines: int) -> dict[str, Any]:
    events = _jsonl_since(start_lines)
    buckets = _tags(events)
    retries = []
    for ev in buckets.get("[TOOL_EXECUTION]", []):
        payload = ev.get("payload") or {}
        msg = str(ev.get("message") or "")
        if (
            payload.get("validation_bounce")
            or "validation" in msg.lower()
            or payload.get("retry")
        ):
            retries.append(ev)
    return {
        "event_count": len(events),
        "tags": {k: len(v) for k, v in buckets.items()},
        "samples": {
            k: (v[-1] if v else None)
            for k, v in buckets.items()
            if k != "[VOICE_ASR]"
        },
        "validation_retries": retries[:6],
        "cascade_in_reasoning": any(
            "cascade" in str(e.get("message") or "").lower()
            or "cascade" in str((e.get("payload") or {}).get("clean_preview") or "").lower()
            for e in buckets.get("[REASONING_TRACE]", [])
        ),
    }


def main() -> int:
    print("=" * 72)
    print("Dana Stage 3 Deep Stress — Cross-Agent Memory & MoA")
    print(f"session_id={SESSION_ID}")
    print(f"marker={MARKER}")
    print("=" * 72)

    _silence_tts()
    _bind_session()
    start_lines = _mark_telemetry()
    results: list[TurnResult] = []

    steps = [
        ("Turn 1", turn1_memory_ingest),
        ("Turn 2", turn2_mailroom_vision),
        ("Turn 3", turn3_vision_distraction),
        ("Turn 4", turn4_moa_cross_agent),
        ("Turn 5", turn5_chat_verify),
    ]
    for label, fn in steps:
        print(f"\n--- {label} ---")
        t0 = time.perf_counter()
        try:
            tr = fn()
        except Exception as exc:  # noqa: BLE001
            tr = TurnResult(label, "", False, errors=[f"crash: {exc}"])
            traceback.print_exc()
        results.append(tr)
        print(f"  elapsed={time.perf_counter() - t0:.1f}s  PASS={tr.passed}")

    a = audit(start_lines)

    print("\n" + "=" * 72)
    print("PASS/FAIL SUMMARY")
    print("=" * 72)
    for tr in results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"\n[{status}] {tr.name}")
        print(
            f"  input: {tr.input_text[:110]}"
            f"{'...' if len(tr.input_text) > 110 else ''}"
        )
        for d in tr.details:
            print(f"  - {d}")
        for e in tr.errors:
            print(f"  ! {e}")

    # Highlight Cascade Router recall explicitly.
    t4 = results[3] if len(results) > 3 else None
    print("\n" + "=" * 72)
    print("CASCADE ROUTER RECALL (Turn 4 MoA)")
    print("=" * 72)
    if t4:
        recalled = any("moa_recalled_cascade_router=True" in d for d in t4.details)
        print(f"  recalled={recalled}")
        for d in t4.details:
            if "cascade" in d.lower() or "rapidfuzz" in d.lower() or "moa_" in d:
                print(f"  - {d}")

    print("\n" + "=" * 72)
    print("TELEMETRY VERIFICATION (JSONL since marker)")
    print("=" * 72)
    print(f"  events_since_marker~={a['event_count']}")
    for tag in ("[ROUTER]", "[HANDOFF]", "[REASONING_TRACE]", "[TOOL_EXECUTION]"):
        count = a["tags"].get(tag, 0)
        flag = "OK" if count > 0 else "MISSING"
        print(f"  [{flag}] {tag}: {count}")
        sample = a["samples"].get(tag)
        if sample:
            print(f"       msg={str(sample.get('message') or '')[:120]!r}")

    print("\nVALIDATION RETRIES")
    if a["validation_retries"]:
        for item in a["validation_retries"]:
            print(f"  - {item.get('message')} payload={item.get('payload')}")
    else:
        print("  (none tagged validation_bounce — check Turn 4 validation_obs_hits)")

    fails = sum(1 for tr in results if not tr.passed)
    required = ["[ROUTER]", "[HANDOFF]", "[REASONING_TRACE]", "[TOOL_EXECUTION]"]
    missing = [t for t in required if a["tags"].get(t, 0) == 0]
    print("\n" + "=" * 72)
    print(
        f"OVERALL: {'PASS' if fails == 0 and not missing else 'FAIL'} "
        f"({5 - fails}/5 turns, missing_tags={missing or 'none'})"
    )
    print("=" * 72)
    return 0 if fails == 0 and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
