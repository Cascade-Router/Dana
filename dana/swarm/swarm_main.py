#!/usr/bin/env python3
"""Donna deep-research swarm — Planner → Search (WebSearchTool) → Writer.

Architecture:
  PlannerAgent  — decompose topic into discrete search objectives
  Search Agent  — ``llm.bind_tools([WebSearchTool])``; cache via Scratchpad
  WriterAgent   — synthesize multi-paragraph report from scratchpad only
  write_node    — persist ``docs/latest_swarm_report.txt``

Usage:
  python -m dana.swarm.swarm_main "research topic here"

Auth:
  Loads CAMGRASPER/.env via python-dotenv.
  ChatOpenAI reads OPENAI_API_KEY from the environment automatically.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from dana.paths import DOCS_DIR, ENV_PATH, PROJECT_ROOT
from dana.swarm.scratchpad import (
    format_findings_for_writer,
    open_session,
    read_findings,
    write_finding,
)
from dana.swarm.web_search_tool import WebSearchTool, build_web_search_tool, search_once

_PROJECT_ROOT = PROJECT_ROOT
load_dotenv(ENV_PATH)
load_dotenv()

_DOCS_DIR = DOCS_DIR
_REPORT_PATH = _DOCS_DIR / "latest_swarm_report.txt"

_MAX_SEARCH_TOOL_ROUNDS = 3
_MAX_PLAN_OBJECTIVES = 5

PLANNER_SYSTEM = """
You are the PlannerAgent for Donna's deep-research multi-agent layer.
Your only job is query decomposition:
1. Break the user's complex research topic into 2–5 discrete search objectives.
2. Each objective must be a concrete, self-contained web search query string
   the Search Agent can pass directly to WebSearchTool.
3. Cover distinct facets (definitions, recent developments, numbers, caveats)
   — do not duplicate near-identical queries.

Output EXACTLY one JSON array of strings. No markdown fences. No commentary.
Example: ["objective one", "objective two", "objective three"]
""".strip()

SEARCH_SYSTEM = """
You are the Search Agent for Donna's deep-research swarm.
Protocol (strict):
1. You MUST use the bound `web_search` tool (WebSearchTool / web.run wrapper)
   to collect outside information. Never invent facts or URLs.
2. Always populate a non-empty `query` argument before calling the tool.
   Empty or placeholder queries are forbidden and will be rejected.
3. Prefer one focused search per objective. After the tool returns, briefly
   acknowledge what was found — do not call tools out-of-turn once you have
   usable results for this objective.
4. Raw findings are written to the Scratchpad cache by the runtime; you do
   not need to format a final report.
""".strip()

WRITER_SYSTEM = """
You are the WriterAgent for Donna's deep-research swarm.
Protocol (strict):
1. Compile a comprehensive multi-paragraph Markdown synthesis using ONLY the
   Scratchpad / cache findings provided below.
2. Do NOT call tools. Do NOT invent facts, statistics, or URLs absent from
   the scratchpad.
3. Structure:
   - Title
   - Multi-paragraph Key Findings (prose, not only bullets)
   - Caveats / Uncertainty
   - Recommended Next Steps
4. Keep under 600 words. If the scratchpad is empty, say so plainly.
""".strip()


class SwarmState(TypedDict):
    query: str
    plan: list[str]
    session_id: str
    findings: list[str]
    report: str


def _build_llm(*, temperature: float = 0.2) -> ChatOpenAI:
    """Cloud LLM for planner / search / writer — never Ollama."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def _llm_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    if content is None:
        return str(result or "").strip()
    return str(content).strip()


def _parse_plan(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None
    if not isinstance(payload, list):
        # Fallback: treat each non-empty line as an objective.
        lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if ln][:_MAX_PLAN_OBJECTIVES]
    out: list[str] = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            q = str(item.get("query") or item.get("objective") or "").strip()
            if q:
                out.append(q)
    return out[:_MAX_PLAN_OBJECTIVES]


def planner_agent(state: SwarmState) -> dict[str, Any]:
    """PlannerAgent — decompose topic into discrete Search Agent objectives."""
    query = (state.get("query") or "").strip() or "(empty query)"
    llm = _build_llm(temperature=0.1)
    try:
        raw = _llm_text(
            llm.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM),
                    HumanMessage(
                        content=f"Research topic:\n{query}\n\nEmit the JSON array now."
                    ),
                ]
            )
        )
        plan = _parse_plan(raw)
    except Exception as exc:  # noqa: BLE001
        plan = [query]
        return {
            "query": query,
            "plan": plan,
            "session_id": state.get("session_id") or "",
            "findings": [f"(PlannerAgent failed: {exc}; using topic as sole objective)"],
        }
    if not plan:
        plan = [query]
    session_id = state.get("session_id") or ""
    if not session_id:
        try:
            session_id = open_session(query)
        except Exception:  # noqa: BLE001
            session_id = ""
    return {
        "query": query,
        "plan": plan,
        "session_id": session_id,
    }


def _tool_query_from_call(tc: dict[str, Any] | Any) -> str:
    """Extract ``query`` from a LangChain tool_call dict; never invent one."""
    if isinstance(tc, dict):
        args = tc.get("args") or tc.get("arguments") or {}
    else:
        args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
    if not isinstance(args, dict):
        return ""
    return str(args.get("query") or "").strip()


def _run_search_objective(
    llm_with_tools: Any,
    tool: WebSearchTool,
    objective: str,
    *,
    session_id: str,
) -> str:
    """Tool-bound Search Agent loop for one planned objective.

    Blocks WebSearchTool invocation when ``query`` is empty — the LLM must
    populate arguments before the router/runtime executes the wrapper.
    """
    messages: list[Any] = [
        SystemMessage(content=SEARCH_SYSTEM),
        HumanMessage(
            content=(
                f"Search objective from PlannerAgent:\n{objective}\n\n"
                "Call web_search with a specific non-empty query derived from "
                "this objective."
            )
        ),
    ]
    last_findings = ""
    for _round in range(_MAX_SEARCH_TOOL_ROUNDS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            # Model answered without tools — if we already have findings, stop.
            if last_findings:
                break
            # Force one deterministic search_once using the objective itself.
            summary = search_once(objective)
            last_findings = (
                summary.findings_text
                if summary.ok
                else f"ERROR: {summary.error or 'search failed'}"
            )
            if session_id and summary.ok:
                write_finding(
                    session_id,
                    objective=objective,
                    query_used=summary.query,
                    findings_text=last_findings,
                )
            break

        for tc in tool_calls:
            name = str(
                (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
                or ""
            )
            call_id = str(
                (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", ""))
                or f"call-{_round}"
            )
            if name not in (tool.name, "WebSearchTool", "web.run"):
                messages.append(
                    ToolMessage(
                        content=f"ERROR: unknown tool `{name}` — only web_search is bound.",
                        tool_call_id=call_id,
                    )
                )
                continue
            query = _tool_query_from_call(tc)
            if not query:
                # Argument injection gate — do not pre-execute with empty params.
                messages.append(
                    ToolMessage(
                        content=(
                            "ERROR: web_search refused — empty query. "
                            "Repopulate a concrete non-empty query argument, then retry."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue
            observation = tool.invoke({"query": query})
            last_findings = str(observation or "")
            if session_id and not last_findings.startswith("ERROR:"):
                write_finding(
                    session_id,
                    objective=objective,
                    query_used=query,
                    findings_text=last_findings,
                )
            messages.append(ToolMessage(content=last_findings, tool_call_id=call_id))
        # One successful tool round is enough per objective.
        if last_findings and not last_findings.startswith("ERROR:"):
            break
    return last_findings or f"(no findings for objective: {objective})"


def search_agent(state: SwarmState) -> dict[str, Any]:
    """Search Agent — bind WebSearchTool; cache findings in Scratchpad."""
    query = (state.get("query") or "").strip()
    plan = list(state.get("plan") or [])
    if not plan:
        plan = [query or "(empty)"]
    session_id = state.get("session_id") or ""
    if not session_id:
        try:
            session_id = open_session(query)
        except Exception:  # noqa: BLE001
            session_id = ""

    tool = build_web_search_tool()
    llm = _build_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools([tool])

    finding_blobs: list[str] = list(state.get("findings") or [])
    for objective in plan:
        try:
            blob = _run_search_objective(
                llm_with_tools,
                tool,
                objective,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            blob = f"(Search Agent error for {objective!r}: {exc})"
        finding_blobs.append(blob)

    # Prefer scratchpad rows when available (canonical cache).
    if session_id:
        try:
            cached = read_findings(session_id)
            if cached:
                finding_blobs = [
                    format_findings_for_writer(cached),
                ]
        except Exception:  # noqa: BLE001
            pass

    return {
        "query": query,
        "plan": plan,
        "session_id": session_id,
        "findings": finding_blobs,
    }


def writer_agent(state: SwarmState) -> dict[str, Any]:
    """WriterAgent — synthesize from Scratchpad findings only (no tools)."""
    query = state.get("query") or ""
    session_id = state.get("session_id") or ""
    findings = list(state.get("findings") or [])

    scratch_text = ""
    if session_id:
        try:
            rows = read_findings(session_id)
            if rows:
                scratch_text = format_findings_for_writer(rows)
        except Exception:  # noqa: BLE001
            scratch_text = ""
    if not scratch_text:
        scratch_text = "\n\n".join(findings) if findings else "(scratchpad empty)"

    llm = _build_llm(temperature=0.2)
    # Writer must NOT bind tools — synthesis only.
    prompt = (
        f"{WRITER_SYSTEM}\n\n"
        f"USER QUERY:\n{query}\n\n"
        f"SCRATCHPAD FINDINGS:\n{scratch_text}\n"
    )
    try:
        report = _llm_text(llm.invoke(prompt)).strip()
    except Exception as exc:  # noqa: BLE001
        report = (
            f"# Swarm report (WriterAgent error)\n\n"
            f"**Query:** {query}\n\n"
            f"**Error:** {exc}\n\n"
            f"## Scratchpad\n\n{scratch_text}\n"
        )
    if not report:
        report = f"# Swarm report\n\n**Query:** {query}\n\n{scratch_text}\n"
    return {
        "query": query,
        "session_id": session_id,
        "findings": findings,
        "report": report,
    }


def write_node(state: SwarmState) -> dict[str, Any]:
    """Persist the final markdown to CAMGRASPER/docs/latest_swarm_report.txt."""
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    body = (state.get("report") or "").strip() or "(empty report)"
    header = (
        f"<!-- Donna swarm report generated "
        f"{datetime.now(timezone.utc).isoformat()} -->\n"
        f"<!-- query: {(state.get('query') or '').replace('--', '-')} -->\n"
        f"<!-- session: {state.get('session_id') or ''} -->\n\n"
    )
    _REPORT_PATH.write_text(header + body + "\n", encoding="utf-8")
    return state


# Backward-compatible aliases for older imports / tests.
research_node = search_agent
synthesis_node = writer_agent


def build_swarm_graph():
    """Compile Planner → Search → Writer → write LangGraph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(SwarmState)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("writer_agent", writer_agent)
    graph.add_node("write", write_node)
    graph.add_edge(START, "planner_agent")
    graph.add_edge("planner_agent", "search_agent")
    graph.add_edge("search_agent", "writer_agent")
    graph.add_edge("writer_agent", "write")
    graph.add_edge("write", END)
    return graph.compile()


def run_swarm(query: str) -> str:
    """Invoke the compiled graph; returns report path."""
    app = build_swarm_graph()
    session_id = ""
    try:
        session_id = open_session(query)
    except Exception:  # noqa: BLE001
        session_id = ""
    app.invoke(
        {
            "query": query,
            "plan": [],
            "session_id": session_id,
            "findings": [],
            "report": "",
        }
    )
    return str(_REPORT_PATH.resolve())


def _spoken_summary_from_report(topic: str, report: str, *, max_words: int = 45) -> str:
    """Condense Markdown report into a short TTS-friendly sentence."""
    text = (report or "").strip()
    text = re.sub(r"(?m)^<!--.*?-->\s*", "", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"[*`_>#]+", " ", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return (
            f"I finished researching {topic}, but the report was empty. "
            "You can open docs/latest_swarm_report.txt for details."
        )
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(".,;:") + "."
    return text


def run_research(topic: str) -> str:
    """Main entry for background threads: run the swarm and return a spoken summary."""
    q = (topic or "").strip() or "(empty topic)"
    run_swarm(q)
    try:
        report = _REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        report = ""
    return _spoken_summary_from_report(q, report)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            'Usage: python -m dana.swarm.swarm_main "<research query>"',
            file=sys.stderr,
        )
        return 2
    query = " ".join(args).strip()
    try:
        path = run_swarm(query)
    except Exception as exc:  # noqa: BLE001
        print(f"[dana.swarm] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[dana.swarm] OK wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
