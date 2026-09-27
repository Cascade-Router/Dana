"""Dynamic LLM Router fleet config — YAML load/validate/reload.

Isolates the PyYAML dependency to this one module. ``routing_config.yaml``
is entirely optional: its absence (the out-of-the-box state — see
``dana.paths.ROUTING_CONFIG_PATH``'s own docstring) means
``dana.core.llm_router`` has nothing to select from, and
``dana.core.model_provider`` falls back to its existing ``.env``-toggle
resolution unchanged. A malformed file degrades the same way (logged,
never raised) rather than crashing turn dispatch over an optional config
mistake.
"""

from __future__ import annotations

import os
from typing import Any

from dana.paths import ROUTING_CONFIG_PATH

_CACHE: dict[str, Any] | None = None
_CACHE_MTIME: float | None = None

_REQUIRED_ENTRY_KEYS = ("id", "provider", "model", "context_window", "priority")


def _validate(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    fleet = raw.get("fleet")
    if not isinstance(fleet, list) or not fleet:
        return None
    seen_ids: set[str] = set()
    for entry in fleet:
        if not isinstance(entry, dict):
            return None
        if not all(k in entry for k in _REQUIRED_ENTRY_KEYS):
            return None
        entry_id = str(entry.get("id") or "")
        if not entry_id or entry_id in seen_ids:
            return None
        seen_ids.add(entry_id)
    return raw


def load_routing_config(*, force_reload: bool = False) -> dict[str, Any] | None:
    """Return the parsed+validated fleet config, or ``None`` when the file
    is absent, unreadable, unparsable, or fails minimal shape validation.

    Reloads whenever the file's mtime changes (or ``force_reload=True``) —
    same "no restart needed" promise ``dana.core.model_provider.
    ensure_dotenv_loaded`` already makes for ``.env``, applied here via an
    mtime check instead of ``.env``'s per-call ``override=True`` reload
    (this file is read far less often — once per routing decision, not
    once per env-var lookup).
    """
    global _CACHE, _CACHE_MTIME
    path = str(ROUTING_CONFIG_PATH)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _CACHE = None
        _CACHE_MTIME = None
        return None

    if not force_reload and _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE

    try:
        import yaml
    except ImportError:
        _CACHE = None
        _CACHE_MTIME = None
        return None

    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except Exception:  # noqa: BLE001 — a bad config file must never crash dispatch
        _CACHE = None
        _CACHE_MTIME = None
        return None

    validated = _validate(raw)
    _CACHE = validated
    _CACHE_MTIME = mtime
    return validated


def cloud_allowed() -> bool:
    """``defaults.allow_cloud`` from ``routing_config.yaml`` — ``True``
    (cloud fleet entries usable) when the config is absent/invalid, or the
    key itself is unset, so this is purely additive like everything else in
    this module. Deliberately a standalone function (not a ``resolve_chain``
    parameter) so callers with no other reason to touch the router at all —
    ``dana.core.react_dispatch``'s Planning-Phase Cloud Lock, which forces a
    turn straight to ``"openrouter"`` by literal string BEFORE
    ``resolve_chain`` is ever reached — can still honor a full-local policy
    without importing ``dana.core.llm_router`` just for this one flag.
    """
    raw = load_routing_config()
    if raw is None:
        return True
    value = (raw.get("defaults") or {}).get("allow_cloud", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


__all__ = ("load_routing_config", "cloud_allowed")
