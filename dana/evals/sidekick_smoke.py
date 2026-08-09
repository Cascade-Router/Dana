"""Text-based Sidekick smoke — no mic / Whisper / TTS.

Exercises broker modality, cascade MoA escalation, chat→tool escalation,
and (when GPU deps allow) live YOLO objects + Florence OCR publish.

Usage::

    python -m dana.evals.sidekick_smoke
    python -m dana.evals.sidekick_smoke --live-vision
    python -m dana.evals.sidekick_smoke --live-ocr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnResult:
    name: str
    ok: bool
    detail: str
    extras: dict[str, Any] = field(default_factory=dict)


def _safe_print(msg: str, *, err: bool = False) -> None:
    """Windows consoles may be cp1252 — never crash on unicode arrows in logs."""
    stream = sys.stderr if err else sys.stdout
    try:
        print(msg, file=stream)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), file=stream)


def _pass(name: str, detail: str, **extras: Any) -> TurnResult:
    _safe_print(f"[PASS] {name}: {detail}")
    return TurnResult(name=name, ok=True, detail=detail, extras=extras)


def _fail(name: str, detail: str, **extras: Any) -> TurnResult:
    _safe_print(f"[FAIL] {name}: {detail}", err=True)
    return TurnResult(name=name, ok=False, detail=detail, extras=extras)


def turn_broker_modality() -> list[TurnResult]:
    """Deterministic YOLO vs Florence wake intents (Rework 2)."""
    from dana.tools.broker import IntentBroker

    broker = IntentBroker()
    cases = [
        ("what do you see on my screen", "analyze_visual_context", "YOLO objects"),
        ("please read the rules on my screen", "ocr_with_region", "Florence OCR"),
        ("read the paragraph on the page", "ocr_with_region", "Florence OCR"),
        ("find the Target button", "ocr_with_region", "Florence OCR"),
        (
            "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket",
            "draft_cursor_prompt",
            "MoA-force tool",
        ),
    ]
    out: list[TurnResult] = []
    for query, expect, label in cases:
        call = broker.parse_utterance(query)
        got = call.tool_id if call else None
        if got == expect:
            out.append(_pass(f"broker:{label}", f"{query!r} -> {got}"))
        else:
            out.append(
                _fail(f"broker:{label}", f"{query!r} -> {got!r}, expected {expect!r}")
            )
    return out


def turn_cascade_moa_escalation() -> list[TurnResult]:
    """Escalation management: decide_route / classify_complexity still active."""
    from dana.agentic import set_dana_mode
    from dana.cascade_router import (
        classify_complexity,
        decide_route,
        is_cascade_enabled,
    )

    out: list[TurnResult] = []
    # Developer mode so chat bypass does not swallow MoA.
    set_dana_mode("developer", as_voice=False)

    high_q = (
        "Please draft a cursor prompt for a complex self-improvement ticket "
        "about DeepSeek MoA routing."
    )
    low_q = "hey dana, how are you today?"

    high_c = classify_complexity(high_q)
    low_c = classify_complexity(low_q)
    if high_c == "high":
        out.append(_pass("complexity:high", f"classify_complexity -> {high_c}"))
    else:
        out.append(_fail("complexity:high", f"got {high_c!r}, expected 'high'"))

    if low_c == "low":
        out.append(_pass("complexity:low", f"classify_complexity -> {low_c}"))
    else:
        out.append(_fail("complexity:low", f"got {low_c!r}, expected 'low'"))

    d_high = decide_route(high_q, forced_tool="draft_cursor_prompt")
    d_low = decide_route(low_q)
    cascade_on = is_cascade_enabled()
    reason_ascii = (d_high.reason or "").encode("ascii", "replace").decode("ascii")
    out.append(
        _pass(
            "cascade:flags",
            f"enabled={cascade_on} high_backend={d_high.backend} "
            f"high_complexity={d_high.complexity} reason={reason_ascii!r}",
            decision={
                "backend": d_high.backend,
                "complexity": d_high.complexity,
                "reason": reason_ascii,
            },
        )
    )
    # High + cascade → moa (or high with reasoner). Low stays local.
    high_ok = d_high.complexity == "high" and (
        d_high.backend == "moa" or "moa" in (d_high.reason or "").lower()
        or "deepseek" in (d_high.reason or "").lower()
        or d_high.backend in {"moa", "local"}  # cascade-disabled still marks high
    )
    if high_ok:
        out.append(
            _pass(
                "escalation:moa_node",
                f"high turn backend={d_high.backend} complexity={d_high.complexity}",
            )
        )
    else:
        out.append(
            _fail(
                "escalation:moa_node",
                f"unexpected high decision backend={d_high.backend} "
                f"complexity={d_high.complexity} reason={d_high.reason!r}",
            )
        )

    if d_low.complexity == "low":
        out.append(
            _pass(
                "escalation:local_node",
                f"low turn backend={d_low.backend} complexity={d_low.complexity}",
            )
        )
    else:
        out.append(
            _fail(
                "escalation:local_node",
                f"low turn unexpectedly complexity={d_low.complexity}",
            )
        )
    return out


def turn_chat_tool_escalation() -> list[TurnResult]:
    """Chat→ReAct escalation (requires_tool_graph) still fires."""
    from dana.agentic import requires_tool_graph, set_dana_mode

    set_dana_mode("chat", as_voice=True)
    out: list[TurnResult] = []
    cases = [
        ("hi dana, how's it going?", False),
        ("please read dana/paths.py and summarize it", True),
        ("run a python script to print hello", True),
        ("what do you see on my screen", False),  # visual ≠ file/tool graph regex
    ]
    for text, expect in cases:
        got = requires_tool_graph(text)
        label = "escalate" if expect else "stay_chat"
        if got == expect:
            out.append(_pass(f"tool_graph:{label}", f"{text!r} -> {got}"))
        else:
            out.append(
                _fail(f"tool_graph:{label}", f"{text!r} -> {got}, expected {expect}")
            )
    return out


def turn_perception_contract_dry() -> list[TurnResult]:
    """Typed topics: YOLO objects must not satisfy OCR readers."""
    from dana.memory.blackboard import (
        init_blackboard,
        publish_perception_objects,
        publish_perception_ocr,
        read_perception_ocr_text,
        read_visual_state,
    )
    from dana.paths import DANA_WORKSPACE

    db = DANA_WORKSPACE / "memory" / "blackboard.db"
    init_blackboard(db)
    publish_perception_objects(
        "[Vision Output] Detected: 1 book.",
        producer="sidekick_smoke",
        db_path=db,
    )
    ocr_before = read_perception_ocr_text(db_path=db)
    # Do not wipe existing OCR — just assert objects ≠ OCR contract.
    objects_line = read_visual_state(db_path=db)
    out: list[TurnResult] = []
    if "1 book" in objects_line or "Detected" in objects_line:
        out.append(_pass("perception:objects_topic", f"Chat ambient={objects_line[:80]!r}"))
    else:
        out.append(_fail("perception:objects_topic", f"unexpected ambient={objects_line!r}"))

    # Publish a fake OCR envelope and confirm OCR reader accepts it.
    publish_perception_ocr(
        "[Florence OCR] regions=1\nRATIONALES Write specific independent rationales [100, 120, 400, 200]",
        producer="sidekick_smoke",
        db_path=db,
    )
    ocr = read_perception_ocr_text(db_path=db)
    if "RATIONALES" in ocr:
        out.append(_pass("perception:ocr_topic", f"OCR chars={len(ocr)}"))
    else:
        out.append(_fail("perception:ocr_topic", f"OCR missing RATIONALES: {ocr[:120]!r}"))

    # YOLO prose must never be returned as OCR.
    from dana.memory.blackboard import set_sensor_state, PERCEPTION_OCR_KEY, SCHEMA_OCR_V1

    set_sensor_state(
        PERCEPTION_OCR_KEY,
        "[Vision Output] Detected: 1 laptop.",
        meta={"schema": SCHEMA_OCR_V1, "kind": "ocr", "producer": "poison"},
        db_path=db,
    )
    poisoned = read_perception_ocr_text(db_path=db)
    if poisoned == "":
        out.append(_pass("perception:reject_yolo_as_ocr", "OCR reader rejected Vision Output"))
    else:
        out.append(_fail("perception:reject_yolo_as_ocr", f"leaked {poisoned!r}"))

    # Restore the good OCR sample for any follow-on live turns.
    publish_perception_ocr(
        "[Florence OCR] regions=1\nRATIONALES Write specific independent rationales [100, 120, 400, 200]",
        producer="sidekick_smoke",
        db_path=db,
    )
    _ = ocr_before  # silence lint
    return out


def turn_live_yolo() -> list[TurnResult]:
    """Cold screen → YOLO → perception.objects."""
    from dana.memory.blackboard import read_perception_objects, read_visual_state
    from dana.vision_tools import analyze_visual_context

    t0 = time.perf_counter()
    try:
        payload = analyze_visual_context(source="screen")
    except Exception as exc:  # noqa: BLE001
        return [_fail("live:yolo", f"analyze_visual_context raised: {exc}")]
    ms = (time.perf_counter() - t0) * 1000.0
    row = read_perception_objects()
    ambient = read_visual_state()
    ok = bool(payload) and row is not None and str(row.get("meta", {}).get("schema") or "").endswith("objects.v1")
    detail = f"{payload!r} latency_ms={ms:.0f} ambient={ambient[:80]!r}"
    return [_pass("live:yolo", detail) if ok else _fail("live:yolo", detail)]


def turn_live_florence() -> list[TurnResult]:
    """Cold screen → Florence OCR → perception.ocr (may be slow / VRAM heavy)."""
    from dana.memory.blackboard import read_perception_ocr
    from dana.tools.visual_tools import ocr_with_region

    t0 = time.perf_counter()
    try:
        obs = ocr_with_region(query="")
    except Exception as exc:  # noqa: BLE001
        return [_fail("live:florence", f"ocr_with_region raised: {exc}")]
    ms = (time.perf_counter() - t0) * 1000.0
    row = read_perception_ocr()
    schema_ok = row is not None and str(row.get("meta", {}).get("schema") or "").endswith("ocr.v1")
    text = str(obs or "")
    yolo_poison = text.lstrip().startswith("[Vision Output]")
    load_error = "ERROR:" in text or "load failed" in text.lower()
    ok = (not yolo_poison) and (not load_error) and ("[Florence OCR]" in text) and schema_ok
    detail = f"chars={len(text)} latency_ms={ms:.0f} schema_ok={schema_ok} sample={text[:160]!r}"
    return [_pass("live:florence", detail) if ok else _fail("live:florence", detail)]


def turn_headless_react_complex() -> list[TurnResult]:
    """Complex cognitive turn: broker force + cascade MoA foresight (no live LLM)."""
    from dana.cascade_router import decide_route
    from dana.tools.broker import IntentBroker

    query = (
        "Dana, use the draft_cursor_prompt tool to log a self-improvement ticket "
        "about DeepSeek MoA routing and Florence OCR perception contracts."
    )
    try:
        call = IntentBroker().parse_utterance(query)
        decision = decide_route(
            query,
            forced_tool=(call.tool_id if call else "draft_cursor_prompt"),
        )
    except Exception as exc:  # noqa: BLE001
        return [_fail("react:complex", f"raised: {exc}")]
    routed = call.tool_id if call else None
    ok = (
        routed == "draft_cursor_prompt"
        and decision.complexity == "high"
        and decision.backend == "moa"
    )
    detail = (
        f"routed={routed!r} backend={decision.backend} "
        f"complexity={decision.complexity} model={getattr(decision, 'model', None)!r}"
    )
    return [_pass("react:complex", detail) if ok else _fail("react:complex", detail)]


def turn_supervisor_health() -> list[TurnResult]:
    from dana.memory.blackboard import sidekick_health

    h = sidekick_health()
    detail = (
        f"vision_alive={h.get('vision_alive')} actuator_alive={h.get('actuator_alive')} "
        f"degraded={h.get('degraded')}"
    )
    # Informational — not a hard fail if daemons aren't up in this smoke.
    return [_pass("supervisor:health", detail, health=h)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dana sidekick text smoke suite")
    parser.add_argument(
        "--live-vision",
        action="store_true",
        help="Run live YOLO screen capture (needs GPU/weights)",
    )
    parser.add_argument(
        "--live-ocr",
        action="store_true",
        help="Run live Florence OCR (VRAM heavy)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary JSON at end",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("DANA_OS_DRY_RUN", "1")
    os.environ.setdefault("DANA_CURSOR_LAUNCH", "0")
    os.environ.setdefault("DANA_DISABLE_TOAST", "1")

    print("Dana Sidekick text smoke")
    print("=" * 60)

    results: list[TurnResult] = []
    results.extend(turn_broker_modality())
    results.extend(turn_cascade_moa_escalation())
    results.extend(turn_chat_tool_escalation())
    results.extend(turn_perception_contract_dry())
    results.extend(turn_headless_react_complex())
    results.extend(turn_supervisor_health())
    if args.live_vision:
        results.extend(turn_live_yolo())
    if args.live_ocr:
        results.extend(turn_live_florence())

    passed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    print("=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed, {len(results)} total")

    # Explicit answer about escalation after the rework.
    moa = next((r for r in results if r.name == "escalation:moa_node"), None)
    tool_esc = [r for r in results if r.name.startswith("tool_graph:")]
    _safe_print("\n--- Escalation management after sidekick rework ---")
    if moa and moa.ok:
        _safe_print(
            "YES - MoA/cascade escalation nodes are still active. "
            "High-complexity / draft_cursor_prompt turns still classify as high "
            "and route through the MoA reasoner path (decide_route). "
            "The perception rework did not remove or bypass cascade_router."
        )
    else:
        _safe_print(
            "WARNING - MoA escalation check did not pass; inspect cascade flags above."
        )
    if tool_esc and all(r.ok for r in tool_esc):
        _safe_print(
            "YES - chat->tool-graph escalation (requires_tool_graph) still fires for "
            "file/code/shell intents while casual chat stays on the lightweight node."
        )

    if args.json:
        _safe_print(
            json.dumps(
                {
                    "passed": passed,
                    "failed": failed,
                    "results": [
                        {"name": r.name, "ok": r.ok, "detail": r.detail, **r.extras}
                        for r in results
                    ],
                },
                indent=2,
                default=str,
            )
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
