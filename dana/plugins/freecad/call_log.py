"""Standardized log of FreeCAD tool calls executed during a ReAct session.

One ``CadCallRecord`` per dispatched tool call, in execution order — the
single source of truth ``dana.plugins.freecad.py_export`` renders into a
standalone, human-editable FreeCAD macro (Frontier 4, "Show Your Work").

Deliberately separate from the OpenAI wire-format ``messages`` history the
ReAct loop already keeps (``dana.core.react_dispatch``): that history mixes
prose/JSON for LLM consumption and round-trips through assistant/tool role
framing, while this is a clean, ordered, non-LLM data structure meant purely
for code generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CadCallRecord:
    """One executed tool call, in the order it happened.

    ``arguments`` is what the call was asked to do (post-finalization, e.g.
    including any injected ``face_centroid``); ``result`` is the engine's
    own success payload (resolved default names, coerced dimensions,
    absolute placements) — keeping both means the exporter never has to
    re-derive geometry math the engine already computed once.
    """

    tool_id: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


class CadCallLog:
    """Append-only, ordered sequence of ``CadCallRecord`` for one session."""

    def __init__(self) -> None:
        self._records: list[CadCallRecord] = []

    def record(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> CadCallRecord:
        entry = CadCallRecord(
            tool_id=tool_id,
            arguments=dict(arguments),
            ok=ok,
            result=dict(result or {}),
            error=error,
        )
        self._records.append(entry)
        return entry

    @property
    def records(self) -> list[CadCallRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)


__all__ = ("CadCallLog", "CadCallRecord")
