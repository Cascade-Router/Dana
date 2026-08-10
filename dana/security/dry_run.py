"""Single source of truth for ``DANA_OS_DRY_RUN`` parsing.

Every actuator previously re-implemented this check inline; most accepted
``{"1", "true", "yes", "on"}`` but ``dana.tools.os_control`` and
``dana.os_automation`` silently dropped ``"on"``, so the same env var meant
slightly different things depending which module read it.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def is_dry_run_enabled(env_var: str = "DANA_OS_DRY_RUN") -> bool:
    """True when ``env_var`` is set to a recognized truthy value (case-insensitive)."""
    return os.environ.get(env_var, "").strip().lower() in _TRUE_VALUES


__all__ = ("is_dry_run_enabled",)
