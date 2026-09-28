"""REST API for the frontend's Model Registry & Control Panel — see
``dana.core.model_registry`` for the actual catalog/detection logic. This
module is a thin FastAPI wrapper, same shape as ``dana.api.system``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from dana.core.model_registry import get_registry_service

router = APIRouter()

# Shared with dana.core.llm_router (reports fleet failures into the same
# instance) — see ModelRegistryService's module docstring / get_registry_service.
_registry = get_registry_service()


@router.get("/api/models/matrix")
def get_models_matrix() -> dict[str, Any]:
    return {"ok": True, **_registry.get_matrix()}


class SaveModelPreferencesRequest(BaseModel):
    order: list[str] = []
    disabled: list[str] = []


@router.post("/api/models/preferences")
def save_models_preferences(body: SaveModelPreferencesRequest) -> dict[str, Any]:
    prefs = _registry.save_preferences(body.order, body.disabled)
    return {"ok": True, "preferences": prefs, **_registry.get_matrix()}


@router.post("/api/models/{model_id:path}/clear-error")
def clear_model_runtime_error(model_id: str) -> dict[str, Any]:
    """Lets the HITL manually reset a model's circuit-breaker state (e.g.
    after funding an account or fixing a typo'd key) — ``:path`` so model ids
    containing "/" (``openai/gpt-oss-120b``, OpenRouter slugs, ...) still
    route correctly instead of 404ing on the first segment.
    """
    _registry.clear_runtime_error(model_id)
    return {"ok": True, **_registry.get_matrix()}
