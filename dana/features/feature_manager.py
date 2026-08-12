"""Feature/plugin toggle engine — persisted flags + real tool-registry gating.

Mirrors dana/settings.py's persisted-JSON pattern, but each feature's default
is auto-detected from a real runtime signal (installed FreeCADCmd, an API key
in the environment, ...) rather than a static default, and toggling a feature
actually unbinds/rebinds its tools against the live ToolRegistry + IntentBroker
instance rather than just flipping a cosmetic flag.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dana.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from dana.tools.broker import IntentBroker

_FLAGS_PATH = PROJECT_ROOT / "feature_flags.json"
_CACHE: dict[str, Any] | None = None
_CACHE_LOCK = threading.RLock()


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
                "create_freecad_extrusion",
                "modify_existing_freecad_document",
                "execute_freecad_script",
            ),
        ),
        Feature(id="autocad_com", label="AutoCAD COM Engine", implemented=False),
        Feature(
            id="vision_vlm",
            label="Vision & VLM Analysis",
            tool_ids=("analyze_cad_blueprint", "verify_cad_rendering"),
        ),
        Feature(id="piper_tts", label="Piper TTS Speech"),
        Feature(id="whisper_stt", label="Whisper STT Listening"),
        Feature(
            id="os_actuator",
            label="OS Actuator Hardware Control",
            tool_ids=("capture_and_analyze_screen", "execute_os_keystrokes"),
        ),
    )
}


def list_features() -> list[Feature]:
    return list(FEATURES.values())


def tool_ids_for_feature(feature_id: str) -> tuple[str, ...]:
    feature = FEATURES.get(feature_id)
    return feature.tool_ids if feature is not None else ()


def _detect_default_state(feature_id: str) -> bool:
    if feature_id == "freecad":
        try:
            from dana.plugins.freecad.engine import detect_freecadcmd

            return detect_freecadcmd() is not None
        except Exception:  # noqa: BLE001
            return False
    if feature_id == "vision_vlm":
        return bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
    if feature_id == "autocad_com":
        return False
    return True  # piper_tts, whisper_stt, os_actuator — no real gating signal today


def _default_reason(feature_id: str, enabled: bool) -> str:
    if feature_id == "freecad":
        return "FreeCADCmd detected" if enabled else "FreeCADCmd not found on this machine"
    if feature_id == "vision_vlm":
        return (
            "ANTHROPIC_API_KEY or OPENAI_API_KEY is set"
            if enabled
            else "no vision API key set"
        )
    if feature_id == "autocad_com":
        return "not implemented — stub"
    return "user toggle"


def _read_flags_file() -> dict[str, Any]:
    if not _FLAGS_PATH.exists():
        return {}
    try:
        with open(_FLAGS_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_feature_flags(*, force_reload: bool = False) -> dict[str, Any]:
    """Loads ``{"enabled": {fid: bool}, "pinned_tools": [tool_id, ...]}``.

    Auto-detected defaults are merged under any explicit persisted choice.
    """
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None and not force_reload:
            return {
                "enabled": dict(_CACHE["enabled"]),
                "pinned_tools": list(_CACHE["pinned_tools"]),
            }
        raw = _read_flags_file()
        persisted_enabled = raw.get("enabled") if isinstance(raw.get("enabled"), dict) else {}
        enabled = {fid: _detect_default_state(fid) for fid in FEATURES}
        for fid, value in persisted_enabled.items():
            if fid in enabled:
                enabled[fid] = bool(value)
        pinned = [t for t in raw.get("pinned_tools", []) if isinstance(t, str)]
        _CACHE = {"enabled": enabled, "pinned_tools": pinned}
        return {"enabled": dict(enabled), "pinned_tools": list(pinned)}


def _write_flags_file(state: dict[str, Any]) -> None:
    try:
        with open(_FLAGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError:
        pass


def is_feature_enabled(feature_id: str) -> bool:
    return bool(load_feature_flags()["enabled"].get(feature_id, False))


def set_feature_enabled(feature_id: str, enabled: bool) -> None:
    """Persists the toggle, applies it to the live registry, and notifies listeners."""
    global _CACHE
    if feature_id not in FEATURES:
        return
    with _CACHE_LOCK:
        state = load_feature_flags(force_reload=True)
        state["enabled"][feature_id] = bool(enabled)
        _write_flags_file(state)
        _CACHE = {
            "enabled": dict(state["enabled"]),
            "pinned_tools": list(state["pinned_tools"]),
        }

    if feature_id == "os_actuator":
        os.environ["DANA_OS_DRY_RUN"] = "0" if enabled else "1"

    try:
        from dana.tools.broker import get_broker

        # A full reload_registry() (not just apply_feature_gating) so that
        # *re*-enabling a feature restores its tools too — apply_feature_gating
        # only ever removes; reload_registry rebuilds everything from scratch
        # and then re-applies gating (see dana/tools/broker.py's reload_registry),
        # correctly handling both directions of the toggle.
        get_broker().reload_registry()
    except Exception:  # noqa: BLE001
        pass

    try:
        from dana.core import shared_state

        shared_state.notify_feature_flags_changed(load_feature_flags())
    except Exception:  # noqa: BLE001
        pass


def get_pinned_tool_ids() -> set[str]:
    return set(load_feature_flags()["pinned_tools"])


def _set_pinned(tool_id: str, pinned: bool) -> None:
    global _CACHE
    with _CACHE_LOCK:
        state = load_feature_flags(force_reload=True)
        current = set(state["pinned_tools"])
        if pinned:
            current.add(tool_id)
        else:
            current.discard(tool_id)
        state["pinned_tools"] = sorted(current)
        _write_flags_file(state)
        _CACHE = {
            "enabled": dict(state["enabled"]),
            "pinned_tools": list(state["pinned_tools"]),
        }
    try:
        from dana.core import shared_state

        shared_state.notify_feature_flags_changed(load_feature_flags())
    except Exception:  # noqa: BLE001
        pass


def pin_tool(tool_id: str) -> None:
    _set_pinned(tool_id, True)


def unpin_tool(tool_id: str) -> None:
    _set_pinned(tool_id, False)


def disabled_tool_ids() -> set[str]:
    enabled = load_feature_flags()["enabled"]
    out: set[str] = set()
    for fid, feature in FEATURES.items():
        if not enabled.get(fid, False):
            out.update(feature.tool_ids)
    return out


def apply_feature_gating(broker: "IntentBroker") -> None:
    """Unbinds every disabled feature's tools from a live IntentBroker + ToolRegistry.

    Filters three surfaces so a disabled tool truly can't be reached: the
    broker's alias-matching registry, the process-wide ToolRegistry singleton
    (bind_tools/retrieve_specs exposure), and the broker's cached
    ``_initialized_tools`` dispatch-fallback snapshot.
    """
    from dana.tools.registry import get_tool_registry

    disabled = disabled_tool_ids()
    if not disabled:
        return

    for tid in disabled:
        broker.registry.pop(tid, None)

    reg = get_tool_registry()
    for tid in disabled:
        reg.unregister(tid)

    initialized = getattr(broker, "_initialized_tools", None)
    if initialized is not None:
        broker._initialized_tools = [
            entry for entry in initialized if entry[0] not in disabled
        ]


def describe_feature_access(query: str) -> str:
    """Deterministic answer to 'do you have access to <feature>?'-style questions."""
    text = (query or "").strip().lower()
    match: Feature | None = None
    if text:
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
        return (
            "I couldn't match that to a known feature. Ask about: "
            + ", ".join(f.label for f in FEATURES.values())
        )

    enabled = is_feature_enabled(match.id)
    if not match.implemented:
        return f"No, the {match.label} is not implemented — it's a stub in this build."
    reason = _default_reason(match.id, enabled)
    verb = "Yes" if enabled else "No"
    state = "enabled" if enabled else "disabled"
    return f"{verb}, the {match.label} is currently {state} ({reason})."


def active_feature_manifest_text() -> str:
    """Human-readable block for injection into the LLM system prompt."""
    enabled_map = load_feature_flags()["enabled"]
    lines: list[str] = []
    for feature in FEATURES.values():
        enabled = enabled_map.get(feature.id, False)
        if not feature.implemented:
            lines.append(f"- {feature.label}: NOT IMPLEMENTED (stub)")
            continue
        state = "ENABLED" if enabled else "DISABLED"
        reason = _default_reason(feature.id, enabled)
        lines.append(f"- {feature.label}: {state} ({reason})")
    return "\n".join(lines)
