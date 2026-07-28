"""WebSearchTool — LangChain-bound ``web.run`` wrapper for deep-research Search Agents.

Exposes ``search_once`` / ``SearchSummary`` semantics used by the Planner→Search→Writer
swarm. Empty queries are rejected before any network call.
"""

from __future__ import annotations

from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearchSummary(BaseModel):
    """Structured result of a single ``search_once`` call."""

    query: str
    ok: bool
    hit_count: int = 0
    findings_text: str = ""
    error: str | None = None


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Non-empty web search query. Must be a concrete search string — "
            "never empty, never a placeholder."
        ),
    )


def search_once(query: str, *, max_results: int = 5) -> SearchSummary:
    """Run one web search; refuse empty / whitespace-only queries."""
    q = (query or "").strip()
    if not q:
        return SearchSummary(
            query="",
            ok=False,
            hit_count=0,
            findings_text="",
            error="empty query — refuse to invoke web search",
        )
    from dana.web_search import format_search_observation, web_search

    payload = web_search(q, max_results=max_results)
    if not payload.get("ok"):
        err = str(payload.get("error") or "search failed")
        return SearchSummary(
            query=q,
            ok=False,
            hit_count=0,
            findings_text="",
            error=err,
        )
    hits = list(payload.get("results") or [])
    return SearchSummary(
        query=str(payload.get("query") or q),
        ok=True,
        hit_count=len(hits),
        findings_text=format_search_observation(payload),
        error=None,
    )


class WebSearchTool(BaseTool):
    """Donna's ``web.run`` wrapper — bind this tool to the Search Agent LLM."""

    name: str = "web_search"
    description: str = (
        "Search the public web for current facts, schedules, news, and definitions. "
        "You MUST supply a non-empty 'query' string before calling. "
        "Never call with an empty or default query."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str = "", **kwargs: Any) -> str:
        summary = search_once(str(query or ""))
        if not summary.ok:
            return f"ERROR: web_search refused: {summary.error or 'unknown'}"
        return summary.findings_text

    async def _arun(self, query: str = "", **kwargs: Any) -> str:
        return self._run(query, **kwargs)


def build_web_search_tool() -> WebSearchTool:
    """Factory used by LangGraph Search Agent ``llm.bind_tools([...])``."""
    return WebSearchTool()
