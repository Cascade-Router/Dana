#!/usr/bin/env python3
"""Headless text-integration harness for Dana (no mic / Whisper / VAD).

Bypasses InputStream + STT and injects raw strings into the broker / tool_router
and deep-research swarm orchestration. Also enforces the 100-line runtime log
clip before the suite starts.

Usage:
  python test_text_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 0) Runtime log clip (max 100 lines) before any suite I/O
# ---------------------------------------------------------------------------


def _ensure_runtime_log_clipped() -> None:
    from dana.logging import (
        RUNTIME_LOG_MAX_LINES,
        RUNTIME_LOG_PATH,
        _runtime_log_lock,
        _trim_runtime_log_to_last_lines,
        append_runtime_log,
        enable_runtime_file_logging,
    )

    enable_runtime_file_logging()
    with _runtime_log_lock:
        _trim_runtime_log_to_last_lines(RUNTIME_LOG_PATH)
    append_runtime_log(
        f"[test_text_pipeline] session start (max_lines={RUNTIME_LOG_MAX_LINES})\n"
    )
    path = Path(RUNTIME_LOG_PATH)
    if path.is_file():
        n = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        assert n <= RUNTIME_LOG_MAX_LINES, (
            f"runtime log has {n} lines; expected <= {RUNTIME_LOG_MAX_LINES}"
        )
    print(f"[OK] runtime log clipped to last {RUNTIME_LOG_MAX_LINES} lines")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broker_route(text: str):
    from dana.tools.broker import IntentBroker
    from dana.tools.stt_corrector import correct_stt

    cleaned = correct_stt(text)
    broker = IntentBroker()
    call = broker.parse_utterance(cleaned)
    return cleaned, call


def _pass(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Turn 1 — Titan / Watchdog (no JSON / read_local_file)
# ---------------------------------------------------------------------------


def turn1_titan_watchdog() -> None:
    print("\n=== Turn 1: Titan initiative -> dispatch_watchdog ===")
    raw = "Dana, activate the Titan initiative and watch for Notepad."
    cleaned, call = _broker_route(raw)
    print(f"  cleaned={cleaned!r}")
    print(f"  routed={None if call is None else (call.tool_id, call.arguments)}")

    if call is None:
        _fail("broker returned no tool for Titan/watchdog utterance")
    assert call is not None
    if call.tool_id != "dispatch_watchdog":
        _fail(f"expected dispatch_watchdog, got {call.tool_id!r}")
    if call.tool_id == "read_local_file":
        _fail("must not route Titan initiative to read_local_file")
    blob = json.dumps(call.arguments, ensure_ascii=False).lower()
    if ".json" in blob and "titan" not in cleaned.lower():
        _fail("arguments appear to hunt for a .json file")
    task = str(call.arguments.get("task") or "")
    if "notepad" not in task.lower() and "titan" not in task.lower():
        # Still OK if task is the full raw utterance
        if "notepad" not in (call.raw_text or "").lower():
            _fail("watchdog task lost Notepad / Titan context")

    # Mirror production fast-path: tool_router must agree after STT middleware.
    from dana.core_agent import tool_router

    with patch("dana.core.agent_loop.speak_tool_working_ack", lambda *_a, **_k: None), patch(
        "dana.core_agent.SPATIAL_AGGREGATOR"
    ) as _agg:
        _agg.update_transcript = MagicMock()
        text_out, deferred = tool_router(raw)
    print(f"  tool_router deferred={None if deferred is None else deferred.tool_id}")
    if deferred is not None and deferred.tool_id == "read_local_file":
        _fail("tool_router deferred read_local_file for Titan utterance")
    if deferred is not None and deferred.tool_id != "dispatch_watchdog":
        _fail(f"tool_router deferred unexpected tool {deferred.tool_id!r}")
    if "json" in text_out.lower() and "titan" not in text_out.lower():
        # Whisper-collision residue without Titan repair
        _fail(f"STT path left bare JSON collision: {text_out!r}")

    _pass("Titan initiative -> dispatch_watchdog (not read_local_file / .json)")


# ---------------------------------------------------------------------------
# Turn 2 — Deep research Planner -> Search -> Scratchpad -> Writer
# ---------------------------------------------------------------------------


def turn2_deep_research_swarm() -> None:
    print("\n=== Turn 2: Deep research swarm layout ===")
    raw = "Dana, write a comprehensive report on robots navigating a maze."
    cleaned, call = _broker_route(raw)
    print(f"  cleaned={cleaned!r}")
    print(f"  routed={None if call is None else (call.tool_id, call.arguments)}")

    if call is None or call.tool_id != "dispatch_research_swarm":
        _fail(
            f"expected dispatch_research_swarm, got "
            f"{None if call is None else call.tool_id!r}"
        )
    query = str(call.arguments.get("query") or call.arguments.get("topic") or "").strip()
    if not query:
        _fail("dispatch_research_swarm missing populated query/topic")

    # Instrument the multi-agent path with deterministic mocks (no live OpenAI).
    path_trace: list[str] = []
    bound_tools: list[Any] = []

    class _PlannerLLM:
        def invoke(self, _messages):
            path_trace.append("planner_agent")
            return MagicMock(
                content=json.dumps(
                    [
                        "robots maze navigation algorithms",
                        "SLAM path planning maze robots",
                        "recent research robot maze solving",
                    ]
                )
            )

    class _SearchMsg:
        def __init__(self, *, tool_calls=None, content=""):
            self.tool_calls = tool_calls or []
            self.content = content

    class _SearchLLM:
        def bind_tools(self, tools):
            path_trace.append("search_agent_bind_tools")
            bound_tools.extend(list(tools))
            return self

        def invoke(self, messages):
            path_trace.append("search_agent")
            # First call: emit a populated web_search tool call.
            user_blob = ""
            for m in messages:
                user_blob += str(getattr(m, "content", "") or "")
            # Extract objective hint from the human message.
            query_arg = "robots navigating a maze algorithms"
            for line in user_blob.splitlines():
                if line.strip() and "objective" not in line.lower():
                    if len(line.strip()) > 8:
                        query_arg = line.strip()[:120]
                        break
            if any(getattr(m, "type", None) == "tool" for m in messages) or any(
                type(m).__name__ == "ToolMessage" for m in messages
            ):
                return _SearchMsg(content="Findings noted.")
            return _SearchMsg(
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": query_arg},
                        "id": "call_maze_1",
                        "type": "tool_call",
                    }
                ]
            )

    class _WriterLLM:
        def invoke(self, prompt):
            path_trace.append("writer_agent")
            return MagicMock(
                content=(
                    "# Robots Navigating a Maze\n\n"
                    "Key findings summarize path planning and SLAM from the scratchpad.\n\n"
                    "## Caveats\nSynthetic test report.\n"
                )
            )

    llm_cycle = iter([_PlannerLLM(), _SearchLLM(), _WriterLLM()])

    def _fake_build_llm(**_kwargs):
        try:
            return next(llm_cycle)
        except StopIteration:
            return _WriterLLM()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / "research_scratchpad.db"
        from dana.swarm import scratchpad as sp
        from dana.swarm import swarm_main as sm
        from dana.swarm.web_search_tool import SearchSummary, WebSearchTool

        with patch.object(sm, "_build_llm", _fake_build_llm), patch.object(
            sm, "open_session", lambda q: sp.open_session(q, db_path=db)
        ), patch.object(
            sm,
            "write_finding",
            lambda sid, **kw: sp.write_finding(sid, db_path=db, **kw),
        ), patch.object(
            sm,
            "read_findings",
            lambda sid: sp.read_findings(sid, db_path=db),
        ), patch.object(
            sm,
            "search_once",
            lambda q, **_k: SearchSummary(
                query=q,
                ok=True,
                hit_count=1,
                findings_text=f"OK: web_search q={q!r} hits=1\n1. Maze bots — A*",
                error=None,
            ),
        ), patch.object(
            WebSearchTool,
            "_run",
            lambda self, query="", **kw: (
                f"OK: web_search q={query!r} hits=1\n1. Maze bots — A*"
                if str(query).strip()
                else "ERROR: web_search refused: empty query"
            ),
        ):
            # Session opened by run_swarm / planner
            session_id = sp.open_session(query, db_path=db)
            state = {
                "query": query,
                "plan": [],
                "session_id": session_id,
                "findings": [],
                "report": "",
            }
            state.update(sm.planner_agent(state))
            if not state.get("plan"):
                _fail("PlannerAgent produced empty JSON objectives")
            print(f"  plan={state['plan']!r}")

            state.update(sm.search_agent(state))
            findings = sp.read_findings(session_id, db_path=db)
            print(f"  scratchpad_rows={len(findings)}")
            if not findings:
                _fail("Search Agent did not write Scratchpad SQLite findings")
            for row in findings:
                if not str(row.get("query_used") or "").strip():
                    _fail("Scratchpad row missing populated query_used")

            state.update(sm.writer_agent(state))
            if not (state.get("report") or "").strip():
                _fail("WriterAgent produced empty report")

        # Path assertions
        print(f"  path_trace={path_trace}")
        print(f"  bound_tool_names={[getattr(t, 'name', type(t).__name__) for t in bound_tools]}")
        if "planner_agent" not in path_trace:
            _fail("PlannerAgent node never invoked")
        if "search_agent_bind_tools" not in path_trace:
            _fail("Search Agent never called llm.bind_tools([...])")
        if not any(getattr(t, "name", "") == "web_search" for t in bound_tools):
            _fail("WebSearchTool was not bound to Search Agent")
        if "search_agent" not in path_trace:
            _fail("Search Agent node never invoked")
        if "writer_agent" not in path_trace:
            _fail("WriterAgent node never invoked")

    _pass(
        "PlannerAgent -> Search Agent (WebSearchTool bound) -> Scratchpad -> WriterAgent"
    )


# ---------------------------------------------------------------------------
# Turn 3 — Ambiguous chat must NOT force read_local_file
# ---------------------------------------------------------------------------


def turn3_over_strict_broker_fallback() -> None:
    print("\n=== Turn 3: Over-strict broker fallback (chat, not file) ===")
    raw = "I'm thinking about a few relationship things right now."
    cleaned, call = _broker_route(raw)
    print(f"  cleaned={cleaned!r}")
    print(f"  routed={None if call is None else (call.tool_id, call.arguments)}")

    if call is not None and call.tool_id == "read_local_file":
        _fail("broker panic-routed relationship chat to read_local_file")
    filepath = ""
    if call is not None:
        filepath = str(call.arguments.get("filepath") or "")
    if "relationship_reflection" in filepath.lower():
        _fail("broker invented relationship_reflection.txt filepath")

    # Production entry: tool_router should leave this as chat (no deferred file tool).
    from dana.core_agent import tool_router

    with patch("dana.core.agent_loop.speak_tool_working_ack", lambda *_a, **_k: None), patch(
        "dana.core_agent.SPATIAL_AGGREGATOR"
    ) as _agg:
        _agg.update_transcript = MagicMock()
        _text, deferred = tool_router(raw)
    if deferred is not None and deferred.tool_id == "read_local_file":
        _fail("tool_router deferred read_local_file for relationship chat")

    # Core Ollama chat node: run_react_loop with no forced file tool.
    from dana.agentic import REACT_MAX_ITERS, run_react_loop
    from dana.prompts.spatial_synthesis import build_agent_system_prompt
    from dana.tools.broker import IntentBroker
    from dana.tools.schema import ToolCall

    executed: list[str] = []

    def execute_fn(tc: ToolCall) -> str:
        executed.append(tc.tool_id)
        return f"ERROR: unexpected tool in chat turn: {tc.tool_id}"

    class _ChatOnlyLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            return AIMessage(
                content="That sounds like a lot to hold — I'm here if you want to talk it through."
            )

    system = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )
    with patch("langchain_ollama.ChatOllama", lambda **_k: _ChatOnlyLLM()):
        result = run_react_loop(
            user_text=raw,
            system_prompt=system,
            execute_fn=execute_fn,
            max_iters=REACT_MAX_ITERS,
            broker=IntentBroker(),
            enable_reflection=False,
            forced_tool=None,
        )

    if "read_local_file" in executed:
        _fail("chat turn executed read_local_file")
    if any(t.get("tool") == "read_local_file" for t in (result.tool_trace or [])):
        _fail("chat turn tool_trace contains read_local_file")
    final = (result.final_text or "").strip()
    if not final:
        _fail("Ollama chat node returned empty final text")
    print(f"  final={final[:160]!r}")
    _pass("relationship/things chat stayed conversational (no file reader)")


# ---------------------------------------------------------------------------
# Turns 4 & 5 — Tool Forge on ToolNotFound (synthesis unlocked for this run)
# ---------------------------------------------------------------------------

# Deterministic function bodies for the Tool Forge template assembler.
_DESCRIBE_TOOL_BODY = (
    "with Image.open(resolve_safe_path(filepath)) as img:\n"
    "    width, height = img.size\n"
    "    mode = img.mode\n"
    "return (\n"
    '    f"The image at {filepath} is {width}x{height} pixels in {mode} mode; "\n'
    '    "primary subject routed through the local vision/YOLO pipeline."\n'
    ")"
)

_AESTHETICS_TOOL_BODY = (
    "with Image.open(resolve_safe_path(filepath)) as img:\n"
    '    sample = img.convert("RGB").resize((16, 16))\n'
    "    pixels = list(sample.getdata())\n"
    "count = max(1, len(pixels))\n"
    "r = sum(p[0] for p in pixels) // count\n"
    "g = sum(p[1] for p in pixels) // count\n"
    "b = sum(p[2] for p in pixels) // count\n"
    'hex_color = "#%02x%02x%02x" % (r, g, b)\n'
    "return (\n"
    '    f"Aesthetics grasp: dominant color {hex_color} across {count} sampled "\n'
    '    "pixels; balanced composition and harmonious palette."\n'
    ")"
)


class _ForgeLLM:
    """Scripted coder + zero-trust security reviewer for the Tool Forge subgraph."""

    def invoke(self, messages: Any):
        from unittest.mock import MagicMock as _MM

        sys_blob = ""
        user_blob = ""
        for m in messages:
            if isinstance(m, dict):
                role, content = m.get("role"), m.get("content")
            else:
                role, content = getattr(m, "type", ""), getattr(m, "content", "")
            if role == "system":
                sys_blob += str(content or "")
            else:
                user_blob += str(content or "")

        low_sys = sys_blob.lower()
        if "security auditor" in low_sys or "zero-trust" in low_sys:
            return _MM(
                content=json.dumps(
                    {
                        "status": "APPROVED",
                        "threat_assessment": "No egress; image path is sandbox-resolved.",
                        "violations": [],
                        "required_remediation": "",
                    }
                )
            )

        low_user = user_blob.lower()
        if "aesthetic" in low_user or "color palette" in low_user:
            return _MM(
                content=json.dumps(
                    {
                        "tool_name": "aesthetics_grasp_forge",
                        "description": "Evaluate image aesthetics and dominant color palette.",
                        "docstring": "Evaluate the visual aesthetics and dominant color palette of an image.",
                        "python_code": _AESTHETICS_TOOL_BODY,
                    }
                )
            )
        return _MM(
            content=json.dumps(
                {
                    "tool_name": "describe_images_forge",
                    "description": "Describe the contents of an image.",
                    "docstring": "Describe the contents of an image at a sandboxed path.",
                    "python_code": _DESCRIBE_TOOL_BODY,
                }
            )
        )


def _make_sample_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _run_forge_turn(
    *,
    raw: str,
    made_up_tool_name: str,
    filepath_arg: str,
    expected_tool_id: str,
    image_color: tuple[int, int, int],
    description_needle: str,
) -> None:
    from dana.agentic import REACT_MAX_ITERS, run_react_loop
    from dana.paths import DOCS_DIR, GENERATED_TOOLS_DIR
    from dana.prompts.spatial_synthesis import build_agent_system_prompt
    from dana.tools.broker import IntentBroker
    from dana.tools.registry import get_tool_registry
    from dana.tools.schema import ToolCall
    import dana_security

    image_path = DOCS_DIR / filepath_arg.split("/", 1)[-1]
    _make_sample_image(image_path, image_color)

    # Fresh registry so ToolNotFound is the true precondition.
    reg = get_tool_registry(reload=True)
    if expected_tool_id in reg.tools:
        _fail(f"precondition failed: {expected_tool_id} already registered")

    class _ForgeReactLLM:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": made_up_tool_name,
                            "args": {"filepath": filepath_arg},
                            "id": "forge_call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="FINAL: I forged and loaded that tool.")

    def execute_fn(tc: ToolCall) -> str:
        return f"OK: {tc.tool_id}"

    system = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )

    try:
        with patch("langchain_ollama.ChatOllama", lambda **_k: _ForgeReactLLM()), patch(
            "dana.swarm.tool_forge_graph._chat_ollama", lambda **_k: _ForgeLLM()
        ), patch(
            "dana.settings.is_dynamic_tool_synthesis_enabled", lambda: True
        ), patch.object(
            dana_security, "register_tool_schema", lambda *a, **k: {"id": expected_tool_id}
        ), patch(
            "dana.tools.broker.reload_broker_registry", lambda *a, **k: None
        ):
            result = run_react_loop(
                user_text=raw,
                system_prompt=system,
                execute_fn=execute_fn,
                max_iters=REACT_MAX_ITERS,
                broker=IntentBroker(),
                enable_reflection=False,
                forced_tool=None,
            )

        forge_seen = any(
            "toolnotfound" in str(t.get("observation") or "").lower()
            or "forged" in str(t.get("observation") or "").lower()
            for t in (result.tool_trace or [])
        )
        if not forge_seen:
            _fail(f"Tool Forge was not triggered on ToolNotFound (trace={result.tool_trace})")

        gen_file = GENERATED_TOOLS_DIR / f"{expected_tool_id}.py"
        if not gen_file.is_file():
            _fail(f"generated tool not written to {gen_file}")

        reg_after = get_tool_registry()
        entry = reg_after.get(expected_tool_id)
        if entry is None:
            _fail(f"{expected_tool_id} was not hot-loaded into ToolRegistry")
        assert entry is not None
        if entry.callable is None:
            _fail(f"{expected_tool_id} registered without a callable")

        description = str(reg_after.execute(expected_tool_id, filepath=filepath_arg))
        print(f"  forged {expected_tool_id} -> {description!r}")
        if description_needle.lower() not in description.lower():
            _fail(f"{expected_tool_id} description missing {description_needle!r}: {description!r}")
    finally:
        try:
            image_path.unlink()
        except OSError:
            pass

    _pass(f"ToolNotFound -> Tool Forge -> AST+Security -> hot-loaded `{expected_tool_id}`")


def turn4_describe_images_forge() -> None:
    print("\n=== Turn 4: Describe Images Tool Forge ===")
    _run_forge_turn(
        raw="Dana, describe the contents of the image located at docs/sample_photo.jpg.",
        made_up_tool_name="describe_image_contents",
        filepath_arg="docs/sample_photo.jpg",
        expected_tool_id="describe_images_forge",
        image_color=(200, 30, 30),
        description_needle="pixels",
    )


def turn5_aesthetics_grasp_forge() -> None:
    print("\n=== Turn 5: Aesthetics Grasp Tool Forge ===")
    _run_forge_turn(
        raw="Dana, evaluate the visual aesthetics and color palette of docs/design_concept.jpg.",
        made_up_tool_name="evaluate_visual_aesthetics",
        filepath_arg="docs/design_concept.jpg",
        expected_tool_id="aesthetics_grasp_forge",
        image_color=(40, 90, 180),
        description_needle="dominant color",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Dana headless text-integration suite")
    print("=" * 60)
    try:
        _ensure_runtime_log_clipped()
        turn1_titan_watchdog()
        turn2_deep_research_swarm()
        turn3_over_strict_broker_fallback()
        turn4_describe_images_forge()
        turn5_aesthetics_grasp_forge()
    except AssertionError as exc:
        print(f"\nSUITE FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\nSUITE ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("ALL TURNS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
