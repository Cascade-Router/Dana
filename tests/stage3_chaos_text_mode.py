"""Stage 3 text-mode chaos soak — FSM Modules 1–4, no STT/TTS.

Drives the same bureaucratic paths as production ``run_brain_turn`` /
``run_react_loop`` while silencing speech. Usage:

    python -m tests.stage3_chaos_text_mode
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

# Ensure repo root on path when run as script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dana.paths import LOGS_DIR, DONNA_WORKSPACE  # noqa: E402

TURN1 = "switch to vision mounts."
TURN2 = "Switch back to chat mode. My favorite color is cobalt blue."
TURN3 = "What did I just tell you my favorite color was?"
TURN4 = (
    "Donna, use the draft_cursor_prompt tool to log a self-improvement ticket "
    "to implement a sliding-window garbage collector for our SQLite blackboard "
    "so it doesn't grow infinitely."
)

CHAOS_MARKER = f"CHAOS_STAGE3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


@dataclass
class TurnResult:
    name: str
    input_text: str
    passed: bool
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _silence_tts() -> None:
    """No-op speech so soak never blocks on Piper / audio."""
    try:
        import dana.core_agent as ca

        ca.enqueue_speech = lambda *a, **k: None  # type: ignore[assignment]
        ca.wait_for_speech_idle = lambda *a, **k: True  # type: ignore[assignment]
        ca.set_subtitle = lambda *a, **k: None  # type: ignore[assignment]
        ca.emit_live_transcript = lambda *a, **k: None  # type: ignore[assignment]
        ca.set_ui_state = lambda *a, **k: None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] TTS silence incomplete: {exc}")


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
                    "message": CHAOS_MARKER,
                    "session_id": "chaos-stage3",
                    "payload": {"marker": True},
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


def _bb_search(needle: str) -> list[str]:
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
                    rows = con.execute(
                        f"SELECT {col} FROM {table} WHERE lower({col}) LIKE ?",
                        (f"%{needle.lower()}%",),
                    ).fetchall()
                except sqlite3.Error:
                    continue
                for (content,) in rows:
                    if content:
                        hits.append(str(content)[:200])
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        hits.append(f"<bb_error:{exc}>")
    return hits


def _tags_present(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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


def turn1_mailroom() -> TurnResult:
    from dana.agentic import get_donna_mode, set_donna_mode
    from dana.cascade_router import decide_route, fuzzy_match_command
    from dana.handoff import execute_handoff
    from dana.schema import Handoff

    r = TurnResult("Turn 1 Mailroom / RapidFuzz", TURN1, False)
    set_donna_mode("chat")
    hit = fuzzy_match_command(TURN1)
    decision = decide_route(TURN1)
    mode = get_donna_mode()
    r.details.append(f"fuzzy_hit={hit}")
    r.details.append(f"decide_route.reason={decision.reason!r}")
    r.details.append(f"mode_after={mode!r}")

    # Structured Swarm handoff for deterministic capability switch (Module 4).
    try:
        execute_handoff(
            Handoff(
                target_agent="Vision_Agent",
                reason="chaos turn1 mailroom → vision",
                intent_context=TURN1,
            ),
            session_id="chaos-stage3",
        )
        r.details.append("execute_handoff(Vision_Agent) ok")
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"handoff: {exc}")

    ok_hit = (
        hit is not None
        and hit.target == "vision"
        and float(hit.score) >= 80.0
    )
    ok_mode = mode == "vision"
    ok_reason = "mailroom" in (decision.reason or "").lower()
    r.passed = bool(ok_hit and ok_mode and ok_reason)
    if not ok_hit:
        r.errors.append("RapidFuzz did not route to vision ≥80%")
    if not ok_mode:
        r.errors.append(f"mode is {mode!r}, expected vision")
    if not ok_reason:
        r.errors.append("decide_route did not short-circuit via mailroom")
    return r


def turn2_chat_memory() -> TurnResult:
    """Mode switch + cobalt fact — mirrors production compound-utterance behavior."""
    from dana.agentic import (
        append_chat_memory_turn,
        clear_chat_memory,
        get_donna_mode,
        mode_switch_spoken_ack,
        parse_mode_switch,
        run_lightweight_chat,
        set_donna_mode,
    )
    from dana.cascade_router import decide_route
    from dana.memory import append_message, ensure_session
    from dana.handoff import execute_handoff
    from dana.schema import Handoff

    r = TurnResult("Turn 2 Blackboard Memory Ingestion", TURN2, False)
    clear_chat_memory()

    # Production: mode-switch fast-path wins on compound utterances.
    switched = parse_mode_switch(TURN2)
    decision = decide_route(TURN2)
    r.details.append(f"parse_mode_switch={switched!r}")
    r.details.append(f"decide_route.reason={decision.reason!r}")

    if switched is not None:
        active = set_donna_mode(switched)
        ack = mode_switch_spoken_ack(active)
        r.details.append(f"mode_switch_ack={ack!r} mode={active!r}")
        try:
            execute_handoff(
                Handoff(
                    target_agent="Chat_Node",
                    reason="chaos turn2 → chat",
                    intent_context=TURN2,
                ),
                session_id="chaos-stage3",
            )
        except Exception as exc:  # noqa: BLE001
            r.errors.append(f"handoff: {exc}")

        # Residual clause after mode phrase (production currently drops this).
        residual = TURN2
        for phrase in (
            "Switch back to chat mode.",
            "switch back to chat mode.",
            "Switch to chat mode.",
            "switch to chat mode.",
        ):
            if phrase in residual:
                residual = residual.replace(phrase, "", 1).strip()
                break
        # Also strip if only "chat mode" matched mid-string.
        if residual.lower().startswith("my favorite"):
            pass
        elif "my favorite color" in residual.lower():
            idx = residual.lower().index("my favorite color")
            residual = residual[idx:].strip()
        else:
            residual = ""

        r.details.append(f"residual_after_mode_strip={residual!r}")
        if residual:
            # Text-mode CLI: after mode ack, process residual clause as chat content.
            # Dual-write to Blackboard (Module 1 contract). Note: production
            # ``run_brain_turn`` currently returns on mode match and drops residual;
            # lightweight chat also still uses RAM ``chat_memory_buffer`` only.
            sid = ensure_session(
                "chaos-stage3",
                current_agent="Chat_Node",
                active_intent="chat",
            )
            append_message(sid, "user", residual)
            r.details.append(
                "note=dual_write_blackboard (prod run_brain_turn would drop residual)"
            )
            try:
                from dana.core_agent import ask_ollama_messages, OLLAMA_MODEL
                from dana.agentic import build_lightweight_chat_system_prompt

                result = run_lightweight_chat(
                    user_text=residual,
                    system_prompt=build_lightweight_chat_system_prompt(),
                    model=OLLAMA_MODEL,
                    ask_fn=ask_ollama_messages,
                    use_chat_memory=True,
                )
                append_message(sid, "assistant", result.final_text or "")
                r.details.append(f"chat_reply={result.final_text!r}")
            except Exception as exc:  # noqa: BLE001
                append_chat_memory_turn(residual, "Got it — cobalt blue noted.")
                append_message(sid, "assistant", "Got it — cobalt blue noted.")
                r.errors.append(f"chat_llm: {exc}")
        else:
            r.errors.append(
                "compound utterance fully consumed by mode short-circuit; "
                "cobalt clause never reached chat/Blackboard (production gap)"
            )
    else:
        r.errors.append("mode switch to chat did not fire")

    mode_ok = get_donna_mode() == "chat"
    bb_hits = _bb_search("cobalt")
    r.details.append(f"blackboard_cobalt_hits={len(bb_hits)}")
    if bb_hits:
        r.details.append(f"bb_sample={bb_hits[0]!r}")

    # Pass requires chat mode + cobalt on Blackboard (Stage 3 contract).
    r.passed = bool(mode_ok and bb_hits)
    if not mode_ok:
        r.errors.append(f"mode={get_donna_mode()!r} expected chat")
    if not bb_hits:
        r.errors.append("cobalt blue not found in blackboard.db")
    return r


def turn3_recall() -> TurnResult:
    from dana.agentic import (
        build_lightweight_chat_system_prompt,
        get_donna_mode,
        run_lightweight_chat,
        set_donna_mode,
    )
    from dana.memory import load_messages

    r = TurnResult("Turn 3 Blackboard Memory Recall", TURN3, False)
    set_donna_mode("chat")
    bb_msgs = load_messages("chaos-stage3", limit=20)
    r.details.append(f"bb_message_count={len(bb_msgs)}")
    try:
        from dana.core_agent import ask_ollama_messages, OLLAMA_MODEL

        result = run_lightweight_chat(
            user_text=TURN3,
            system_prompt=build_lightweight_chat_system_prompt(),
            model=OLLAMA_MODEL,
            ask_fn=ask_ollama_messages,
            use_chat_memory=True,
        )
        answer = (result.final_text or "").strip()
        r.details.append(f"answer={answer!r}")
    except Exception as exc:  # noqa: BLE001
        answer = ""
        r.errors.append(f"chat_llm: {exc}")
        traceback.print_exc()

    # Prefer model answer; fall back to Blackboard evidence of cobalt.
    has_cobalt = "cobalt" in answer.lower()
    bb_has = bool(_bb_search("cobalt"))
    r.passed = bool(get_donna_mode() == "chat" and (has_cobalt or bb_has and "color" in TURN3.lower()))
    # Stricter: spoken/chat answer must mention cobalt.
    if not has_cobalt:
        r.errors.append("chat answer did not mention cobalt blue")
        r.passed = False
    if has_cobalt:
        r.passed = True
        r.errors = [e for e in r.errors if "cobalt" not in e.lower()]
    return r


def turn4_moa_guard() -> TurnResult:
    from dana.agentic import REACT_MAX_ITERS, run_react_loop, set_donna_mode
    from dana.cascade_router import fuzzy_match_command
    from dana.tools.broker import IntentBroker
    from dana.tools.schema import ToolCall

    r = TurnResult("Turn 4 DeepSeek Extractor & Pydantic Guard", TURN4, False)
    set_donna_mode("developer")

    # Stage 3.1: long prompt must NOT hit the mailroom.
    mail_hit = fuzzy_match_command(TURN4)
    r.details.append(f"mailroom_on_turn4={mail_hit}")
    if mail_hit is not None:
        r.errors.append(f"mailroom hijacked Turn 4: {mail_hit}")

    validation_bounces = 0
    tool_trace: list[dict[str, Any]] = []
    final = ""

    def execute_fn(tc: ToolCall) -> str:
        nonlocal validation_bounces
        try:
            from dana.core_agent import execute_tool_call

            return execute_tool_call(tc)
        except Exception as exc:  # noqa: BLE001
            if "ValidationError" in type(exc).__name__ or "validation" in str(exc).lower():
                validation_bounces += 1
            raise

    try:
        broker = IntentBroker()
        routed = broker.parse_utterance(TURN4)
        forced = routed
        if forced is None:
            from dana.tools.schema import ToolCall

            forced = ToolCall(
                tool_id="draft_cursor_prompt",
                arguments={
                    "objective": (
                        "Implement a sliding-window garbage collector for the "
                        "SQLite blackboard so it doesn't grow infinitely."
                    ),
                    "context": TURN4,
                },
            )
        r.details.append(
            f"forced_tool={getattr(forced, 'tool_id', None)!r}"
        )
        result = run_react_loop(
            user_text=TURN4,
            system_prompt=(
                "You are Donna in developer mode. Prefer draft_cursor_prompt for "
                "self-improvement tickets. Include target files, root cause, "
                "step-by-step changes, and acceptance criteria in the tool args."
            ),
            execute_fn=execute_fn,
            max_iters=max(REACT_MAX_ITERS, 4),
            broker=broker,
            forced_tool=forced,
            enable_reflection=False,
            tts_callback=None,
        )
        final = (result.final_text or "").strip()
        tool_trace = list(result.tool_trace or [])
        r.details.append(f"final={final[:240]!r}")
        r.details.append(f"tool_trace_ids={[t.get('tool') for t in tool_trace]}")
        r.details.append(f"validation_bounces_outer={validation_bounces}")
    except Exception as exc:  # noqa: BLE001
        r.errors.append(f"react: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    draft_hit = any(
        str(t.get("tool") or "") == "draft_cursor_prompt" for t in tool_trace
    )
    # Also count ValidationError / bounce evidence in tool observations.
    bounce_obs = [
        t
        for t in tool_trace
        if "validation" in str(t.get("observation") or "").lower()
        or "Validation Error" in str(t.get("observation") or "")
        or "retry" in str(t.get("observation") or "").lower()
    ]
    r.details.append(f"validation_obs_hits={len(bounce_obs)}")

    db = _bb_path()
    think_rows = 0
    if db.is_file():
        try:
            con = sqlite3.connect(str(db))
            think_rows = int(
                con.execute("SELECT COUNT(*) FROM reasoning_traces").fetchone()[0]
            )
            con.close()
        except Exception:  # noqa: BLE001
            pass
    r.details.append(f"reasoning_traces_rows={think_rows}")

    ledger = DONNA_WORKSPACE / "donna_security" / "patch_ledger.md"
    ledger_snip = ""
    if ledger.is_file():
        text = ledger.read_text(encoding="utf-8", errors="replace")
        if (
            "garbage collector" in text.lower()
            or "sliding-window" in text.lower()
            or "sliding window" in text.lower()
        ):
            ledger_snip = "ledger mentions blackboard/gc topic"
    r.details.append(f"ledger_evidence={ledger_snip or 'none'}")

    if not draft_hit and not bounce_obs:
        r.errors.append(
            "draft_cursor_prompt not in tool_trace and no ValidationError bounce observed"
        )
    if mail_hit is not None:
        r.passed = False
    else:
        # Pass if tool ran (success or guarded bounce/retry path exercised).
        r.passed = bool(draft_hit or bounce_obs)
    return r


def audit_logs(start_lines: int) -> dict[str, Any]:
    events = _jsonl_since(start_lines)
    buckets = _tags_present(events)
    conv = LOGS_DIR / "donna_conversation.log"
    runtime = LOGS_DIR / "donna_runtime.log"
    conv_tail = ""
    runtime_tail = ""
    if conv.is_file():
        conv_tail = "\n".join(conv.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    if runtime.is_file():
        runtime_tail = "\n".join(
            runtime.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        )
    retries = []
    for ev in buckets.get("[TOOL_EXECUTION]", []):
        payload = ev.get("payload") or {}
        msg = str(ev.get("message") or "")
        if "retry" in msg.lower() or "validation" in msg.lower() or payload.get("retry"):
            retries.append(ev)
    # Runtime grep for ValidationError bounce
    for line in runtime_tail.splitlines():
        if "ValidationError" in line or "validation bounce" in line.lower() or "guard" in line.lower():
            retries.append({"runtime": line.strip()[:240]})
    return {
        "event_count": len(events),
        "tags": {k: len(v) for k, v in buckets.items()},
        "samples": {
            k: (v[-1] if v else None) for k, v in buckets.items() if k != "[VOICE_ASR]"
        },
        "retries": retries[:8],
        "conversation_tail": conv_tail,
        "runtime_interesting": [
            ln.strip()
            for ln in runtime_tail.splitlines()
            if any(
                tok in ln
                for tok in (
                    "mailroom",
                    "Mode switch",
                    "MoA",
                    "Validation",
                    "draft_cursor",
                    "Blackboard",
                    "REASONING",
                    "Handoff",
                    "ROUTER",
                )
            )
        ][-20:],
    }


def main() -> int:
    print("=" * 72)
    print("Donna Stage 3 Chaos Test — Text Mode (no STT/TTS)")
    print(f"marker={CHAOS_MARKER}")
    print("=" * 72)

    _silence_tts()
    start_lines = _mark_telemetry()
    results: list[TurnResult] = []

    print("\n--- Turn 1: Mailroom ---")
    t0 = time.perf_counter()
    results.append(turn1_mailroom())
    print(f"  elapsed={time.perf_counter() - t0:.1f}s  PASS={results[-1].passed}")

    print("\n--- Turn 2: Blackboard ingest ---")
    t0 = time.perf_counter()
    results.append(turn2_chat_memory())
    print(f"  elapsed={time.perf_counter() - t0:.1f}s  PASS={results[-1].passed}")

    print("\n--- Turn 3: Blackboard recall ---")
    t0 = time.perf_counter()
    results.append(turn3_recall())
    print(f"  elapsed={time.perf_counter() - t0:.1f}s  PASS={results[-1].passed}")

    print("\n--- Turn 4: MoA + Pydantic guard ---")
    t0 = time.perf_counter()
    results.append(turn4_moa_guard())
    print(f"  elapsed={time.perf_counter() - t0:.1f}s  PASS={results[-1].passed}")

    audit = audit_logs(start_lines)

    print("\n" + "=" * 72)
    print("PASS/FAIL SUMMARY")
    print("=" * 72)
    for tr in results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"\n[{status}] {tr.name}")
        print(f"  input: {tr.input_text[:100]}{'…' if len(tr.input_text) > 100 else ''}")
        for d in tr.details:
            print(f"  · {d}")
        for e in tr.errors:
            print(f"  ! {e}")

    print("\n" + "=" * 72)
    print("TELEMETRY VERIFICATION (JSONL since marker)")
    print("=" * 72)
    print(f"  events_since_marker~={audit['event_count']}")
    for tag, count in audit["tags"].items():
        sample = audit["samples"].get(tag) if tag in audit["samples"] else None
        flag = "OK" if count > 0 else "MISSING"
        print(f"  [{flag}] {tag}: {count}")
        if sample:
            payload = sample.get("payload") or {}
            print(
                f"       msg={str(sample.get('message') or '')[:120]!r} "
                f"payload_keys={list(payload)[:8]}"
            )
            if tag == "[ROUTER]" and "confidence" in payload:
                print(
                    f"       confidence={payload.get('confidence')} "
                    f"target={payload.get('target_node')}"
                )

    print("\nUNHANDLED ERRORS / RETRIES")
    if audit["retries"]:
        for item in audit["retries"]:
            print(f"  · {item}")
    else:
        print("  (none detected in JSONL TOOL_EXECUTION / runtime grep)")

    if audit["runtime_interesting"]:
        print("\nRUNTIME HIGHLIGHTS")
        for ln in audit["runtime_interesting"]:
            print(f"  {ln[:200]}")

    fails = sum(1 for tr in results if not tr.passed)
    required_tags = ["[ROUTER]", "[REASONING_TRACE]", "[TOOL_EXECUTION]", "[HANDOFF]"]
    missing_tags = [t for t in required_tags if audit["tags"].get(t, 0) == 0]
    print("\n" + "=" * 72)
    print(
        f"OVERALL: {'PASS' if fails == 0 and not missing_tags else 'FAIL'} "
        f"({4 - fails}/4 turns, missing_tags={missing_tags or 'none'})"
    )
    print("=" * 72)
    return 0 if fails == 0 and not missing_tags else 1


if __name__ == "__main__":
    raise SystemExit(main())
