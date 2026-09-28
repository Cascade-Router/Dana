"""Sync dana/data/models_registry.json against public catalog sources.

Queries OpenRouter's public ``/api/v1/models`` endpoint (no API key needed)
for its own listed models — context length, pricing, and whether a model
still exists at all. Any OpenRouter-tracked model already in our catalog
whose id no longer appears in OpenRouter's listing is flagged deprecated
rather than removed (a model can vanish from OpenRouter's public catalog
while still being reachable through other providers we track it under).

Run manually (``python scripts/sync_model_catalog.py``) or via the monthly
``.github/workflows/update_models.yml`` job, which opens a PR with the diff
for a human to review rather than merging catalog drift unattended.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "dana" / "data" / "models_registry.json"

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_REQUEST_TIMEOUT_S = 20.0


def _fetch_json(url: str) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dana-model-catalog-sync"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)
        return None


def _openrouter_index() -> dict[str, dict[str, Any]]:
    """``{model_id: raw_openrouter_entry}`` — empty if the fetch failed, so
    callers degrade to a no-op sync rather than wiping the catalog."""
    payload = _fetch_json(OPENROUTER_MODELS_URL)
    if payload is None:
        return {}
    return {str(entry["id"]): entry for entry in payload.get("data") or [] if entry.get("id")}


def sync_openrouter_models(catalog: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[str]:
    """Mutates ``catalog["providers"]["openrouter"]["models"]`` in place.
    Returns a list of human-readable change descriptions for the PR body."""
    changes: list[str] = []
    openrouter = (catalog.get("providers") or {}).get("openrouter")
    if not openrouter:
        return changes

    for model in openrouter.get("models") or []:
        model_id = model.get("id")
        live = index.get(model_id)
        if live is None:
            if not model.get("is_deprecated"):
                model["is_deprecated"] = True
                changes.append(f"openrouter/{model_id}: marked deprecated (no longer in OpenRouter's catalog)")
            continue

        context_length = live.get("context_length")
        if isinstance(context_length, int) and context_length != model.get("context_window"):
            changes.append(
                f"openrouter/{model_id}: context_window {model.get('context_window')} -> {context_length}"
            )
            model["context_window"] = context_length

        pricing = live.get("pricing") or {}
        try:
            prompt_per_1m = float(pricing.get("prompt", 0)) * 1_000_000
            completion_per_1m = float(pricing.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            prompt_per_1m = completion_per_1m = None

        if prompt_per_1m is not None:
            current = model.get("pricing") or {}
            if (
                abs(current.get("prompt_per_1m", 0.0) - prompt_per_1m) > 1e-9
                or abs(current.get("completion_per_1m", 0.0) - completion_per_1m) > 1e-9
            ):
                changes.append(
                    f"openrouter/{model_id}: pricing "
                    f"({current.get('prompt_per_1m')}, {current.get('completion_per_1m')}) -> "
                    f"({prompt_per_1m}, {completion_per_1m})"
                )
                model["pricing"] = {
                    "prompt_per_1m": round(prompt_per_1m, 6),
                    "completion_per_1m": round(completion_per_1m, 6),
                }

    return changes


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    index = _openrouter_index()
    if not index:
        print("no live OpenRouter data fetched — leaving catalog unchanged")
        return 0

    changes = sync_openrouter_models(catalog, index)
    if not changes:
        print("catalog already up to date")
        return 0

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"updated {CATALOG_PATH} with {len(changes)} change(s):")
    for change in changes:
        print(f"  - {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
