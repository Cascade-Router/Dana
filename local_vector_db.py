"""JSON-backed local vector database using vector_math primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from vector_math import VectorDocument, cosine_similarity


class LocalVectorDB:
    """In-memory vector store with JSON persistence and cosine top-K search."""

    def __init__(self) -> None:
        self.documents: list[VectorDocument] = []

    def add_document(self, doc: VectorDocument | dict[str, Any]) -> None:
        if isinstance(doc, VectorDocument):
            self.documents.append(doc)
        elif isinstance(doc, dict):
            self.documents.append(VectorDocument.from_dict(doc))
        else:
            raise TypeError("doc must be a VectorDocument or dict")

    def save_to_disk(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [d.to_dict() for d in self.documents],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_from_disk(self, filepath: str | Path) -> None:
        path = Path(filepath)
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs = raw.get("documents") if isinstance(raw, dict) else raw
        self.documents = [
            VectorDocument.from_dict(d) if isinstance(d, dict) else d
            for d in (docs or [])
        ]

    def search_top_k(
        self, query_vector: Sequence[float], k: int
    ) -> list[VectorDocument]:
        if k <= 0 or not self.documents:
            return []
        scored: list[tuple[float, VectorDocument]] = []
        for doc in self.documents:
            score = cosine_similarity(query_vector, doc.get_vector())
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[: int(k)]]
