"""Pure-Python inverted-index tool retriever — no FAISS/NumPy/ML dependency.

An alternate scoring engine for the SAME narrowing contract
``dana.core.tool_retrieval.narrow_tool_ids_by_query`` already exposes:
``dana.tools.registry.ToolRegistry``'s existing hash-embedding vector index
(``hash_embed`` + ``_VectorIndex``) does catalog-wide top-K retrieval today,
live-wired into every real ``/ws/chat`` turn (see that module's own
docstring). This class is a drop-in alternative to the query side of that
pipeline — swap it in via ``DANA_TOOL_RAG_BACKEND=inverted_index`` (see
``tool_retrieval.py``) — NOT a second, competing narrowing pass. All of the
sticky-id/force-include/token-budget/capability-decay protections layered
around ``narrow_tool_ids_by_query`` in ``dana.core.react_dispatch`` are
untouched either way.

Classic TF(+IDF-lite)-weighted inverted index (``token -> [tool_id, ...]``)
plus ``heapq.nlargest`` for O(D log K) top-K extraction, where D is the
number of tools any query token actually touches — never the full catalog
size, unlike a linear score-and-sort over every tool.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from dana.tools.registry import is_bindable_tool

if TYPE_CHECKING:
    from dana.tools.registry import ToolRegistry

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class FastToolRetriever:
    """Inverted-index + min-heap top-K tool retriever.

    Built once from a ``ToolRegistry`` snapshot (``__init__``/``rebuild``);
    ``get_top_k_tools`` is then a pure read — no I/O, no ML calls, safe to
    run synchronously on every turn. This class never touches the registry
    after construction; ``tool_retrieval.py`` owns deciding when a rebuild
    is needed (the registry's tool set changes when a user teaches a new
    skill or a forge/custom tool gets saved).
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._index: dict[str, list[str]] = {}
        self._doc_freq: dict[str, int] = {}
        self._tool_ids: frozenset[str] = frozenset()
        self.rebuild(registry)

    def rebuild(self, registry: "ToolRegistry") -> None:
        """(Re)build the inverted index from every currently bindable tool
        in ``registry`` — phantom ``dynamic: true`` tools with no real
        source are skipped, same safety filter the embedding index's own
        ``retrieve()`` applies, so this backend can never suggest a tool
        the model couldn't actually call."""
        index: dict[str, list[str]] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)
        tool_ids: list[str] = []
        with registry._lock:
            entries = list(registry.tools.values())
        for entry in entries:
            if not is_bindable_tool(entry):
                continue
            tool_ids.append(entry.name)
            for tok in set(_tokenize(entry.schema_text)):
                index[tok].append(entry.name)
                doc_freq[tok] += 1
        self._index = dict(index)
        self._doc_freq = dict(doc_freq)
        self._tool_ids = frozenset(tool_ids)

    def get_top_k_tools(self, prompt: str, k: int = 5) -> list[str]:
        """Return up to ``k`` tool ids ranked by accumulated token-overlap
        score against ``prompt``, highest-scoring first.

        Falls back to an empty list — never raises — when the prompt has no
        indexable tokens or none of them match anything in the catalog, so
        a caller can safely union this straight into an existing tool set
        with no special-casing.
        """
        tokens = _tokenize(prompt)
        if not tokens or not self._tool_ids:
            return []

        total_docs = max(len(self._tool_ids), 1)
        scores: dict[str, float] = defaultdict(float)
        for tok in tokens:
            matches = self._index.get(tok)
            if not matches:
                continue
            doc_freq = self._doc_freq.get(tok, 1)
            # Rarer tokens ("fastener", "boolean") are more discriminating
            # than common ones ("create", "object") — a lightweight IDF
            # bonus on top of raw term frequency, not a full BM25/TF-IDF
            # normalization (this catalog is small enough that the extra
            # complexity wouldn't change ranking quality meaningfully).
            weight = 1.0 + math.log(total_docs / doc_freq)
            for tool_id in matches:
                scores[tool_id] += weight

        if not scores:
            return []
        # heapq.nlargest keeps a bounded size-k heap internally — O(D log K)
        # for D scored candidates, never a full sort of the whole catalog.
        top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [tool_id for tool_id, _score in top]


__all__ = ("FastToolRetriever",)
