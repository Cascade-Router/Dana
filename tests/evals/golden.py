"""Golden dataset loader for Dānā eval benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
_REQUIRED_KEYS = frozenset(
    {
        "id",
        "category",
        "user_input",
        "expected_initial_node",
        "requires_hitl",
        "ground_truth_output",
    }
)
CATEGORIES = frozenset(
    {
        "routing_intent",
        "vision_grounding",
        "hitl_safety",
        "memory_recall",
    }
)
ALLOWED_INITIAL_NODES = frozenset(
    {"chat", "planner", "agent", "tools", "ticket_validate"}
)


def load_golden_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate ``golden_dataset.json`` cases."""
    target = path or _GOLDEN_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        raise ValueError(f"golden dataset must contain a list under 'cases': {target}")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case[{i}] must be an object")
        missing = _REQUIRED_KEYS - set(case)
        if missing:
            raise ValueError(f"case[{i}] missing keys: {sorted(missing)}")
        cid = str(case["id"]).strip()
        if not cid:
            raise ValueError(f"case[{i}] has empty id")
        if cid in seen_ids:
            raise ValueError(f"duplicate case id: {cid}")
        seen_ids.add(cid)
        cat = str(case["category"]).strip()
        if cat not in CATEGORIES:
            raise ValueError(f"case {cid}: unknown category {cat!r}")
        node = str(case["expected_initial_node"]).strip()
        if node not in ALLOWED_INITIAL_NODES:
            raise ValueError(f"case {cid}: unexpected expected_initial_node {node!r}")
        if not isinstance(case["requires_hitl"], bool):
            raise ValueError(f"case {cid}: requires_hitl must be bool")
        validated.append(case)
    return validated
