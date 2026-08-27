"""Semantic narrowing of an already-capability-gated tool set — Pillar 1 of
the token-compression architecture.

``dana.tools.registry.ToolRegistry`` already implements the embedding /
vector-index machinery this needs (``hash_embed`` + a FAISS-or-NumPy
``_VectorIndex``, both zero-cost/local) — it was built for the older
``dana.agentic``/``dana.tools.broker`` LangGraph stack and was never wired
into the live ``/ws/chat`` ReAct loop (``dana.core.react_dispatch`` +
``dana.api.server``). This module is that missing wire: it reuses the
EXISTING registry singleton rather than standing up a second embedding
pipeline.

``dana.core.react_dispatch._tool_ids_for_plugins`` (capability routing —
``load_capability``, the frontend's active-plugin set, ``_CORE_TOOL_IDS``)
still decides WHICH domains are even eligible this turn — that safety/gating
boundary is untouched. This module only decides, within whatever is already
eligible, which handful of tool schemas are worth their token cost for the
CURRENT user intent. That is the generalization of the old hand-curated
``freecad_essential``/``freecad_full`` split (dana.core.react_dispatch) to
ANY domain, present or future, with no per-plugin manual tiering ever
needed again as the plugin count grows into the dozens.
"""

from __future__ import annotations

import os

from dana.core.tool_retriever import FastToolRetriever
from dana.tools.registry import ToolRegistry, get_tool_registry

# Below this many allowed tools, narrowing adds latency/risk for no real
# token savings — the whole point is trimming a LARGE eligible set (e.g. 24
# FreeCAD tools) down, not second-guessing a session with 3 tools unlocked.
_MIN_TOOLS_TO_NARROW = 8

_DEFAULT_TOP_K = 6

# DANA_TOOL_RAG_BACKEND=inverted_index swaps the scoring engine below from
# ToolRegistry's default hash-embedding vector search to FastToolRetriever's
# pure-Python inverted-index + heapq.nlargest ranking (dana.core.
# tool_retriever) — same narrowing CONTRACT (still only narrows within
# `allowed_ids`, still unions `always_include`, still passes through
# unindexed ids), different scoring math, no FAISS/NumPy dependency for the
# query-time path. Defaults to "embedding" (today's already-live, already-
# tuned behavior) rather than switching the default — this is an opt-in
# alternative until it's been run against real traffic, not a replacement.
_DEFAULT_BACKEND = "embedding"

# One retriever per ToolRegistry instance (keyed by id() — the registry is
# normally a process-wide singleton, but tests construct fresh ones), lazily
# rebuilt whenever the registry's tool-id set changes (a new skill saved via
# save_new_skill, a forge/custom tool, etc.) rather than on every call.
_fast_retrievers: dict[int, tuple[frozenset[str], FastToolRetriever]] = {}


def _fast_retriever_for(registry: ToolRegistry) -> FastToolRetriever:
    current_ids = frozenset(registry.tools)
    cached = _fast_retrievers.get(id(registry))
    if cached is not None and cached[0] == current_ids:
        return cached[1]
    retriever = FastToolRetriever(registry)
    _fast_retrievers[id(registry)] = (current_ids, retriever)
    return retriever


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


def tool_rag_enabled() -> bool:
    return _env_flag("DANA_TOOL_RAG", True)


def tool_rag_backend() -> str:
    raw = (os.environ.get("DANA_TOOL_RAG_BACKEND") or "").strip().lower()
    return raw or _DEFAULT_BACKEND


def narrow_tool_ids_by_query(
    allowed_ids: frozenset[str],
    query: str,
    *,
    always_include: frozenset[str] = frozenset(),
    k: int | None = None,
) -> frozenset[str]:
    """Return a top-K subset of ``allowed_ids`` relevant to ``query``, union
    ``always_include`` — this NEVER adds a tool ``allowed_ids`` didn't
    already contain. Capability gating stays the single source of truth for
    WHAT a session may reach; this only narrows HOW MUCH of it gets
    serialized into this one turn's ``tools=`` payload.

    Falls back to returning ``allowed_ids`` unchanged whenever narrowing
    isn't worth it (too few candidates, no query text, the feature disabled)
    or whenever the retrieval call itself fails — a broken/slow embedding
    step is a pure optimization and must never cost the model a tool it was
    otherwise entitled to.

    Any tool id in ``allowed_ids`` that the registry singleton has no
    embedding for at all (e.g. a user-taught skill saved via
    ``save_new_skill`` after the registry last loaded ``tools.json``) passes
    through untouched rather than being silently dropped — retrieval can't
    judge the relevance of something it never indexed, so "unscored" must
    never mean "excluded".
    """
    query = (query or "").strip()
    if not tool_rag_enabled() or not query or len(allowed_ids) <= _MIN_TOOLS_TO_NARROW:
        return allowed_ids
    try:
        registry = get_tool_registry()
        top_k = k if k is not None else _env_int("DANA_TOOL_RAG_TOP_K", _DEFAULT_TOP_K)
        if tool_rag_backend() == "inverted_index":
            retriever = _fast_retriever_for(registry)
            matched_ids = frozenset(retriever.get_top_k_tools(query, k=top_k))
        else:
            specs = registry.retrieve_specs(query, k=top_k, always_include=always_include)
            matched_ids = frozenset(specs)
        known_ids = frozenset(registry.as_spec_dict())
        unindexed = allowed_ids - known_ids
        narrowed = (matched_ids & allowed_ids) | (always_include & allowed_ids) | unindexed
        return narrowed if narrowed else allowed_ids
    except Exception:  # noqa: BLE001 — retrieval is a pure optimization, never worth failing a turn over
        return allowed_ids


__all__ = ("narrow_tool_ids_by_query", "tool_rag_backend", "tool_rag_enabled")
