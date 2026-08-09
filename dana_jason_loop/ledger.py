"""Roadmap ledger for Green-Flagged Dana/Jason discoveries."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from dana.paths import PROJECT_ROOT, TOOLS_JSON

_ROOT = str(PROJECT_ROOT)
DEFAULT_ROADMAP_PATH = str(TOOLS_JSON.parent / "roadmap.json")


def _utility_effort_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
    """Highest utility / lowest effort first.

    utility ≈ total score; low effort ≈ high feasibility (+ resource_cost).
    """
    scores = item.get("scores") or {}
    total = float(item.get("total_score") or item.get("total") or 0)
    feasibility = float(scores.get("feasibility") or 0)
    resource = float(scores.get("resource_cost") or 0)
    # Sort descending on these, then stable id.
    return (-total, -feasibility, -resource, str(item.get("proposed_id") or ""))


def load_roadmap(path: str | None = None) -> dict[str, Any]:
    roadmap_path = path or DEFAULT_ROADMAP_PATH
    if not os.path.isfile(roadmap_path):
        return {"version": 1, "items": [], "deployed": []}
    with open(roadmap_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {"version": 1, "items": [], "deployed": []}
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    deployed = data.get("deployed")
    if not isinstance(deployed, list):
        deployed = []
    return {
        "version": int(data.get("version") or 1),
        "items": items,
        "deployed": deployed,
    }


def save_roadmap(ledger: dict[str, Any], path: str | None = None) -> str:
    roadmap_path = path or DEFAULT_ROADMAP_PATH
    os.makedirs(os.path.dirname(roadmap_path), exist_ok=True)
    items = list(ledger.get("items") or [])
    items.sort(key=_utility_effort_key)
    payload: dict[str, Any] = {"version": int(ledger.get("version") or 1), "items": items}
    if "deployed" in ledger:
        payload["deployed"] = list(ledger.get("deployed") or [])
    with open(roadmap_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return roadmap_path


def append_green_flag_to_roadmap(
    proposal: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    path: str | None = None,
    round_label: str = "",
) -> dict[str, Any]:
    """Append a Green-Flagged task and re-sort the ledger (utility/effort)."""
    if str(evaluation.get("flag") or "").lower() != "green":
        raise ValueError("Only Green-Flagged evaluations may enter the roadmap ledger")

    ledger = load_roadmap(path)
    pid = str(proposal.get("proposed_id") or evaluation.get("proposed_id") or "")
    # Replace prior entry with same id so re-runs stay idempotent.
    remaining = [i for i in ledger["items"] if i.get("proposed_id") != pid]
    entry = {
        "proposed_id": pid,
        "problem_statement": proposal.get("problem_statement") or "",
        "proposed_solution_code_outline": proposal.get("proposed_solution_code_outline")
        or "",
        "dependencies": list(proposal.get("dependencies") or []),
        "scores": dict(evaluation.get("scores") or {}),
        "total_score": int(evaluation.get("total") or 0),
        "flag": "green",
        "rationale": evaluation.get("rationale") or "",
        "round": round_label,
        "appended_at": datetime.now(timezone.utc).isoformat(),
    }
    remaining.append(entry)
    ledger["items"] = remaining
    save_roadmap(ledger, path)
    return entry
