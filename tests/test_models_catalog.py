"""dana/data/models_registry.json is well-formed: the checks the monthly
catalog sync (.github/workflows/update_models.yml) runs before offering its
change for review. Standard library + pytest only, so that job can run it
with ``pytest --noconftest`` and none of the app's dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

CATALOG_PATH = Path(__file__).resolve().parents[1] / "dana" / "data" / "models_registry.json"
_MODEL_FIELDS = {
    "id": str,
    "name": str,
    "context_window": int,
    "pricing": dict,
    "modalities": list,
    "supports_tool_calling": bool,
    "is_deprecated": bool,
}


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _models() -> list[tuple[str, dict]]:
    return [(provider, model) for provider, spec in _catalog()["providers"].items() for model in spec["models"]]


def test_catalog_shape() -> None:
    catalog = _catalog()
    assert catalog["version"] == 1
    assert isinstance(catalog["providers"], dict) and catalog["providers"]
    for provider, spec in catalog["providers"].items():
        assert isinstance(spec.get("models"), list) and spec["models"], provider
        assert isinstance(spec.get("requires_api_key"), bool), provider


@pytest.mark.parametrize(("provider", "model"), _models(), ids=[f"{p}/{m['id']}" for p, m in _models()])
def test_every_model_entry_is_well_formed(provider: str, model: dict) -> None:
    for field, kind in _MODEL_FIELDS.items():
        assert isinstance(model.get(field), kind), f"{provider}/{model.get('id')}: {field}={model.get(field)!r}"
    assert model["context_window"] > 0
    for side in ("prompt_per_1m", "completion_per_1m"):
        price = model["pricing"].get(side)
        assert isinstance(price, (int, float)) and math.isfinite(price) and price >= 0, (model["id"], side, price)
    if model["id"].endswith(":free"):
        assert not any(model["pricing"].values()), (model["id"], model["pricing"])
    max_output = model.get("max_output_tokens")
    assert max_output is None or (isinstance(max_output, int) and max_output > 0)


def test_model_ids_are_unique_per_provider() -> None:
    for provider, spec in _catalog()["providers"].items():
        ids = [m["id"] for m in spec["models"]]
        assert len(ids) == len(set(ids)), provider
