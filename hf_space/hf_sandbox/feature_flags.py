"""Standalone feature/plugin toggle mapping for the HF sandbox.

Mirrors the shape of the real dana/features/feature_manager.py (same six
features, same "unregister the tools it gates" enforcement idea) but is
fully self-contained — no dana.* imports — since this Space is deployed
independently of the real dana/ package (see README.md's hybrid-architecture
note: this container has no Windows/FreeCAD binaries to actually gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Feature:
    id: str
    label: str
    tool_ids: tuple[str, ...] = ()
    implemented: bool = True


FEATURES: dict[str, Feature] = {
    f.id: f
    for f in (
        Feature(
            id="freecad",
            label="FreeCAD 3D Engine",
            tool_ids=(
                "create_freecad_box",
                "create_freecad_cylinder",
                "modify_existing_freecad_document",
            ),
        ),
        Feature(id="autocad_com", label="AutoCAD COM Engine", implemented=False),
        Feature(
            id="vision_vlm",
            label="Vision & VLM Analysis",
            tool_ids=("analyze_cad_blueprint",),
        ),
        Feature(id="piper_tts", label="Piper TTS Speech"),
        Feature(id="whisper_stt", label="Whisper STT Listening"),
        Feature(
            id="os_actuator",
            label="OS Actuator Hardware Control",
            tool_ids=("capture_cad_viewport", "get_active_windows", "move_window_no_activate"),
        ),
    )
}

DEFAULT_ENABLED: dict[str, bool] = {fid: (fid != "autocad_com") for fid in FEATURES}


def tool_id_to_feature(tool_id: str) -> str | None:
    for feature in FEATURES.values():
        if tool_id in feature.tool_ids:
            return feature.id
    return None


def filter_active_tools(
    enabled: dict[str, bool], tool_registry: dict[str, Callable]
) -> dict[str, Callable]:
    """Subset of tool_registry whose owning feature is enabled (ungated tools pass through)."""
    out: dict[str, Callable] = {}
    for tool_id, fn in tool_registry.items():
        owner = tool_id_to_feature(tool_id)
        if owner is None or enabled.get(owner, True):
            out[tool_id] = fn
    return out


def describe_feature_access(query: str, enabled: dict[str, bool]) -> str | None:
    """Deterministic 'do you have access to X?' answer, or None if no feature matched."""
    text = (query or "").strip().lower()
    if not text:
        return None
    match: Feature | None = None
    for feature in FEATURES.values():
        if feature.id.replace("_", " ") in text or feature.label.lower() in text:
            match = feature
            break
    if match is None:
        for feature in FEATURES.values():
            tokens = feature.label.lower().split()
            if any(tok in text for tok in tokens if len(tok) > 3):
                match = feature
                break
    if match is None:
        return None
    if not match.implemented:
        return f"No, the {match.label} is not implemented — it's a stub in this sandbox."
    is_on = enabled.get(match.id, DEFAULT_ENABLED.get(match.id, True))
    verb = "Yes" if is_on else "No"
    state = "enabled" if is_on else "disabled"
    return f"{verb}, the {match.label} is currently {state}."
