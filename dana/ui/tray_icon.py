"""Dana's Windows system tray icon.

Extracted verbatim from ``dana.core_agent`` (Phase 5 of the core_agent.py
decomposition; see the approved refactor plan). ``_install_signal_handlers``
and ``_shutdown_agent_threads`` stay in ``dana.core_agent`` — they orchestrate
agent-thread/signal lifecycle rather than tray widgets, and are Phase 6
(``dana/core/app_runtime.py``) territory.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any, Optional

import pystray
from PIL import Image, ImageDraw

import dana.core.shared_state as state
from dana.audio.tts_worker import reset_tts_audio_state
from dana.core.shared_state import (
    register_ui_state_listener,
    speech_queue,
    stop_event,
)
from dana.logging import log, log_debug

if TYPE_CHECKING:
    from dana.ui.app_gui import DanaGUI

_TRAY_FILL_IDLE = (37, 99, 235, 255)  # blue
_TRAY_FILL_LISTENING = (22, 163, 74, 255)  # green
_TRAY_GLYPH = (226, 232, 240, 255)
_TRAY_LISTENING_STATES = frozenset({"listening", "followup"})


def create_tray_image(mode: str = "idle") -> Image.Image:
    """Branded tray icon; prefers keyed RGBA logo, procedural fallback."""
    size = 64
    try:
        from dana.ui.startup_tray import build_tray_image

        logo = build_tray_image(mode=mode, size=size)
    except Exception:  # noqa: BLE001
        logo = None
    if logo is not None:
        return logo.convert("RGBA") if hasattr(logo, "convert") else logo
    fill = _TRAY_FILL_LISTENING if mode == "listening" else _TRAY_FILL_IDLE
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (4, 4, size - 5, size - 5),
        radius=14,
        fill=fill,
    )
    draw.ellipse((18, 16, 46, 44), fill=_TRAY_GLYPH)
    draw.rectangle((28, 40, 36, 52), fill=_TRAY_GLYPH)
    # Extra bright status pip when listening (glances faster in the tray).
    if mode == "listening":
        draw.ellipse((42, 8, 56, 22), fill=(250, 250, 250, 255))
        draw.ellipse((45, 11, 53, 19), fill=(34, 197, 94, 255))
    return img


def update_tray_icon_for_state(ui_state: str) -> None:
    """Swap tray icon / tooltip when entering or leaving the listening states."""
    icon = state.get_tray_icon()
    if icon is None:
        return
    listening = ui_state in _TRAY_LISTENING_STATES
    mode = "listening" if listening else "idle"
    title = "Dānā — Listening" if listening else "Dānā · Cybernetic Control Plane"
    try:
        icon.icon = create_tray_image(mode)
        icon.title = title
    except Exception as exc:  # noqa: BLE001
        log_debug("UI", f"Tray icon update skipped ({exc})")


# Safe to register unconditionally — update_tray_icon_for_state no-ops until
# _tray_icon exists (see the None-guard above).
register_ui_state_listener(update_tray_icon_for_state)


def _device_menu_label(index: int, name: str) -> str:
    return f"[{index}] {name}"


def _parse_device_menu_label(label: str) -> Optional[int]:
    from dana.audio.devices import SYSTEM_DEFAULT_LABEL

    if not label or label == SYSTEM_DEFAULT_LABEL:
        return None
    if label.strip().lower() in {"system default (auto)", "system default", "(none)"}:
        return None
    if not label.startswith("[") or "]" not in label:
        return None
    try:
        return int(label[1 : label.index("]")])
    except ValueError:
        return None


def request_dana_quit(icon: Optional[pystray.Icon] = None, _item: Any = None) -> None:
    """Tray Quit / cleanup — stop agent threads and close the GUI."""
    log("Main", "Quit requested (system tray).")
    try:
        from dana.telemetry import set_system_status, stop_dashboard_thread

        set_system_status("Restarting")
        stop_dashboard_thread()
    except Exception:
        pass
    try:
        from dana.tools.registry import cleanup_ephemeral_tools

        cleaned = cleanup_ephemeral_tools(archive=True)
        if cleaned:
            log("Main", f"Ephemeral tool GC archived {len(cleaned)} tool(s): {cleaned}")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: ephemeral tool GC failed: {exc}")
    stop_event.set()
    reset_tts_audio_state("application quit", flush_queue=False)
    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass
    state.set_tray_icon(None)
    gui = state.get_gui_instance()
    if gui is not None:
        try:
            gui.after(0, gui.destroy)
        except Exception:
            try:
                gui.destroy()
            except Exception:
                pass


def run_system_tray(gui: "DanaGUI") -> None:
    """Blocking pystray loop — must only run in a daemon thread (never on CTk)."""
    try:

        def open_settings(icon: pystray.Icon, _item: Any = None) -> None:
            gui.after(0, gui.show_window)

        from dana.ui.startup_tray import (
            check_startup_registry_status,
            toggle_run_on_startup,
        )
        from dana.ui.watchdog import (
            check_shell_watchdog_status,
            get_shared_watchdog,
            toggle_shell_watchdog,
        )

        # Ensure shared watchdog is constructed (wires toast/planner when enabled).
        try:
            get_shared_watchdog()
        except Exception:  # noqa: BLE001
            pass

        menu = pystray.Menu(
            pystray.MenuItem("Open Settings", open_settings, default=True),
            pystray.MenuItem(
                "Run on Startup",
                toggle_run_on_startup,
                checked=lambda item: check_startup_registry_status(item),
            ),
            pystray.MenuItem(
                "Enable Shell Watchdog",
                toggle_shell_watchdog,
                checked=lambda item: check_shell_watchdog_status(item),
            ),
            pystray.MenuItem("Quit", request_dana_quit),
        )
        icon = pystray.Icon(
            "Dana",
            create_tray_image("idle"),
            "Dānā · Cybernetic Control Plane",
            menu,
        )
        state.set_tray_icon(icon)
        log("Main", "System tray icon ready (bottom-right notification area).")
        icon.run()
    except Exception as exc:  # noqa: BLE001
        log("Main", f"WARNING: system tray exited ({type(exc).__name__}: {exc})")
        state.set_tray_icon(None)
