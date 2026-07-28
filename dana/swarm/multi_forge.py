"""Split multi-tool forge utterances into individual Tool Forge goals."""

from __future__ import annotations

import re
from typing import Any

_ONE_THAT_RE = re.compile(
    r"(?:^|[,;:]|\band\b)\s*one\s+that\s+(.+?)(?=(?:,|\band\s+one\s+that\b|$))",
    re.IGNORECASE | re.DOTALL,
)
_NUMBERED_RE = re.compile(
    r"(?:^|[,;]|\band\b)\s*(?:\(\d+\)|\d+[.)])\s*(.+?)(?=(?:[,;]|\band\b\s*(?:\(\d+\)|\d+[.)])|$))",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_HINT_RE = re.compile(
    r"\b("
    r"\d+\s+(?:different\s+)?tools?|"
    r"tools?\s+back[- ]?to[- ]?back|"
    r"back[- ]?to[- ]?back|"
    r"one\s+that\b.+\bone\s+that\b"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


def looks_like_multi_forge(goal: str) -> bool:
    text = (goal or "").strip()
    if not text:
        return False
    return bool(_MULTI_HINT_RE.search(text)) or len(_ONE_THAT_RE.findall(text)) >= 2


def split_forge_goals(goal: str) -> list[str]:
    """Return 1..N concrete forge goals from a batch utterance.

    Example
    -------
    \"build three tools: one that tells the time, one that generates a random
    number, and one that lists files in the sandbox\"
    → three standalone \"build a tool that …\" goals.
    """
    text = (goal or "").strip()
    if not text:
        return []

    clauses: list[str] = []
    for match in _ONE_THAT_RE.finditer(text):
        clause = re.sub(r"\s+", " ", match.group(1)).strip(" .,;:")
        if clause:
            clauses.append(clause)
    if len(clauses) < 2:
        numbered: list[str] = []
        for match in _NUMBERED_RE.finditer(text):
            clause = re.sub(r"\s+", " ", match.group(1)).strip(" .,;:")
            if clause and "tool" not in clause.lower()[:12]:
                numbered.append(clause)
        if len(numbered) >= 2:
            clauses = numbered

    if len(clauses) < 2:
        return [text]

    goals: list[str] = []
    for clause in clauses:
        if not clause.lower().startswith(("build ", "create ", "make ", "forge ")):
            goals.append(f"build a tool that {clause}")
        else:
            goals.append(clause)
    return goals


def run_batch_tool_forge(goal: str, *, missing_tool: str = "") -> dict[str, Any]:
    """Forge one tool per split goal; return aggregate status + loaded names."""
    from dana.swarm.tool_forge_graph import route_tool_not_found

    goals = split_forge_goals(goal)
    if len(goals) == 1 and missing_tool:
        result = route_tool_not_found(goals[0], missing_tool=missing_tool)
        return {
            "status": result.get("status"),
            "loaded_tools": (
                [result["loaded_tool"]]
                if result.get("status") == "loaded" and result.get("loaded_tool")
                else []
            ),
            "feedback": result.get("feedback") or result.get("lint_errors") or "",
            "results": [result],
            "goals": goals,
        }

    results: list[dict[str, Any]] = []
    loaded: list[str] = []
    errors: list[str] = []
    for idx, sub in enumerate(goals):
        hint = missing_tool if idx == 0 and missing_tool else ""
        result = route_tool_not_found(sub, missing_tool=hint)
        results.append(result)
        if result.get("status") == "loaded" and result.get("loaded_tool"):
            loaded.append(str(result["loaded_tool"]))
        else:
            errors.append(
                f"{sub[:80]} → {result.get('status')}: "
                f"{result.get('feedback') or result.get('lint_errors') or 'failed'}"
            )

    if loaded and not errors:
        status = "loaded"
        feedback = (
            f"OK: forged and hot-loaded {len(loaded)} tool(s): "
            + ", ".join(f"`{n}`" for n in loaded)
        )
    elif loaded and errors:
        status = "partial"
        feedback = (
            f"PARTIAL: loaded {len(loaded)} tool(s) ({', '.join(loaded)}); "
            f"failed {len(errors)}: " + " | ".join(errors[:3])
        )
    else:
        status = "error"
        feedback = "ERROR: batch Tool Forge failed — " + " | ".join(errors[:3])

    return {
        "status": status,
        "loaded_tools": loaded,
        "loaded_tool": loaded[0] if loaded else "",
        "feedback": feedback,
        "results": results,
        "goals": goals,
    }
