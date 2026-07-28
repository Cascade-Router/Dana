"""System-tray startup toggle helpers (pystray checked-menu binding)."""

from __future__ import annotations

from typing import Any


def check_startup_registry_status(_item: Any = None) -> bool:
    """Return True when Dānā is registered for login/startup (quiet)."""
    try:
        from donna.tools.setup_startup import is_startup_enabled

        return bool(is_startup_enabled())
    except Exception:  # noqa: BLE001
        return False


def toggle_run_on_startup(icon: Any = None, _item: Any = None) -> None:
    """Toggle OS login/startup registration and refresh the tray checkmark."""
    try:
        from donna.tools.setup_startup import (
            disable_startup,
            enable_startup,
            is_startup_enabled,
        )
    except Exception:  # noqa: BLE001
        return

    try:
        if is_startup_enabled():
            disable_startup()
        else:
            enable_startup()
    except Exception:  # noqa: BLE001
        return

    # Force pystray to re-evaluate ``checked=`` on next menu open.
    if icon is not None:
        try:
            update = getattr(icon, "update_menu", None)
            if callable(update):
                update()
        except Exception:  # noqa: BLE001
            pass
