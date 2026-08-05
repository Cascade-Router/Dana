"""Vector math helpers and document envelope for the local vector DB."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class VectorDocument:
    """A text document paired with an embedding vector."""

    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None

    def get_vector(self) -> list[float]:
        return list(self.vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "vector": list(self.vector),
            "metadata": dict(self.metadata),
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorDocument:
        return cls(
            text=str(data.get("text") or ""),
            vector=[float(x) for x in (data.get("vector") or [])],
            metadata=dict(data.get("metadata") or {}),
            doc_id=data.get("doc_id"),
        )


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("vectors must have the same length")
    if not vec1:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(float(a) ** 2 for a in vec1))
    mag2 = math.sqrt(sum(float(b) ** 2 for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def euclidean_distance(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Euclidean (L2) distance between two equal-length float vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("vectors must have the same length")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vec1, vec2)))
