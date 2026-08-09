"""Text-only Dana agentic loop test (no mic / wake word / TTS).

Feeds a research query through run_react_loop with live Ollama + real web_search.
"""

from __future__ import annotations

import json
import re
import sys

from dana.agentic import REACT_MAX_ITERS, run_react_loop
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.web_search import format_search_observation, web_search
from dana.prompts.spatial_synthesis import build_agent_system_prompt

QUERY = "Research the latest updates on the LangGraph library."

# Pollution markers from sports/kickoff routing (should NOT appear in research obs).
POLLUTION_MARKERS = (
    "Speak kickoffs in that local timezone",
    "Speak the start time in the USER'S local timezone",
    "kickoff time ET",
    "opening match kickoff",
    "World Cup 2026",
)

# Final answer must not leak ReAct protocol or sports schedule noise.
LEAK_MARKERS = (
    "TOOL:",
    "FINAL:",
    "Observation:",
    "kickoff",
    "FIFA",
    "World Cup",
    "p.m. ET",
    "pm ET",
)


_vault_stub: dict[str, str] = {}


def execute_tool_call(tc: ToolCall) -> str:
    if tc.tool_id == "web_search":
        query = str(tc.arguments.get("query") or "").strip()
        if not query:
            return "ERROR: missing query"
        payload = web_search(query)
        return format_search_observation(payload)
    if tc.tool_id == "write_vault_memory":
        key = str(tc.arguments.get("key") or "").strip()
        value = str(tc.arguments.get("value") or "")
        if key:
            _vault_stub[key] = value
        return f"OK: saved {key}={value!r}"
    if tc.tool_id == "read_vault_memory":
        key = str(tc.arguments.get("key") or "").strip()
        return f"OK: {key}={_vault_stub.get(key)!r}"
    return f"ERROR: unsupported tool {tc.tool_id}"


def main() -> int:
    try:
        import dana.core_agent as _agent  # noqa: F401 — ensure agent package loads (Ollama host)
    except ImportError as exc:
        print(f"ERROR: could not import dana.core_agent: {exc}", file=sys.stderr)
        return 1

    broker = IntentBroker()
    intent = broker.parse_utterance(QUERY)
    print("=== Broker intent ===")
    if intent:
        print(f"  tool_id={intent.tool_id!r}  args={intent.arguments!r}")
    else:
        print("  (no fast-path intent — LLM will choose tool)")

    system = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )

    print("\n=== Running ReAct loop ===")
    print(f"Query: {QUERY!r}\n")

    result = run_react_loop(
        user_text=QUERY,
        system_prompt=system,
        execute_fn=execute_tool_call,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        enable_reflection=False,
    )

    print("=== Tool trace ===")
    print(json.dumps(result.tool_trace, indent=2, ensure_ascii=False))

    print("\n=== Final response ===")
    print(result.final_text)
    print(f"\n(iterations={result.iterations}, reply_lang={result.reply_lang})")

    # --- Verification ---
    obs_blob = " ".join(
        str(step.get("observation") or "") for step in result.tool_trace if step.get("tool")
    )
    pollution_hits = [m for m in POLLUTION_MARKERS if m.lower() in obs_blob.lower()]
    leak_hits = [m for m in LEAK_MARKERS if m.lower() in (result.final_text or "").lower()]

    print("\n=== Verification ===")
    if pollution_hits:
        print(f"FAIL: web_search observation polluted by: {pollution_hits}")
    else:
        print("PASS: web_search observation is clean (no kickoff/sports prompt pollution).")

    if leak_hits:
        print(f"FAIL: final answer leaked: {leak_hits}")
    elif not re.search(r"langgraph|lang\s*graph", result.final_text or "", re.I):
        print("WARN: final answer may not mention LangGraph — review manually.")
    else:
        print("PASS: final answer looks like a clean LangGraph summary.")

    if pollution_hits or leak_hits:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
