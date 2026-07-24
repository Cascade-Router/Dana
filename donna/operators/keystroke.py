"""Arrow / navigation keystroke operator (SendInput virtual keys).

Lightweight Actuator action for Feather slide navigation (Left/Right/Tab/Enter)
with stochastic human pauses around key down/up.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

# Common navigation VKs (also defined in os_control for overlapping keys).
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_UP = 0x26
VK_DOWN = 0x28
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_BACK = 0x08

_KEY_ALIASES: dict[str, int] = {
    "left": VK_LEFT,
    "right": VK_RIGHT,
    "up": VK_UP,
    "down": VK_DOWN,
    "arrow_left": VK_LEFT,
    "arrow_right": VK_RIGHT,
    "arrowleft": VK_LEFT,
    "arrowright": VK_RIGHT,
    "vk_left": VK_LEFT,
    "vk_right": VK_RIGHT,
    "vk_up": VK_UP,
    "vk_down": VK_DOWN,
    "tab": VK_TAB,
    "vk_tab": VK_TAB,
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "vk_return": VK_RETURN,
    "esc": VK_ESCAPE,
    "escape": VK_ESCAPE,
    "space": VK_SPACE,
    "backspace": VK_BACK,
    "back": VK_BACK,
}


def _dry_run() -> bool:
    return os.environ.get("DONNA_OS_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_vk(key_name: str) -> int | None:
    """Map a friendly key name to a virtual-key code."""
    raw = (key_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    if raw in _KEY_ALIASES:
        return int(_KEY_ALIASES[raw])
    # Allow numeric VK codes: "0x25" or "37".
    try:
        if raw.startswith("0x"):
            return int(raw, 16) & 0xFF
        if raw.isdigit():
            return int(raw) & 0xFF
    except ValueError:
        return None
    return None


def _human_pause() -> None:
    """Stochastic pause around key events (50–150 ms)."""
    time.sleep(random.uniform(0.050, 0.150))


def press_key(key_name: str) -> str:
    """Send one key down/up via SendInput with human pauses; return observation."""
    vk = resolve_vk(key_name)
    if vk is None:
        return f"ERROR: press_key unknown key_name={key_name!r}"

    # Stage 7.2 / 7.4 — honor halt + yield before injecting.
    try:
        from donna.middleware.kill_switch import halt_if_requested

        if halt_if_requested():
            return "HALTED: press_key — halted by GLOBAL_HALT_EVENT"
    except Exception:  # noqa: BLE001
        pass
    try:
        from donna.middleware.human_yield import yield_check

        yield_check(operator="keystroke")
    except Exception:  # noqa: BLE001
        pass

    if _dry_run():
        _human_pause()
        _human_pause()
        return f"OK: press_key dry_run key={key_name!r} vk=0x{vk:02X}"

    try:
        from donna.tools.os_control import _send_scan

        _human_pause()
        _send_scan(int(vk), key_up=False)
        _human_pause()
        _send_scan(int(vk), key_up=True)
        _human_pause()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: press_key failed: {exc}"

    return f"OK: press_key key={key_name!r} vk=0x{vk:02X}"


class KeystrokeOperator:
    """Thin wrapper for multi-key sequences (slide nav)."""

    def press(self, key_name: str) -> dict[str, Any]:
        obs = press_key(key_name)
        ok = str(obs).startswith("OK:")
        return {"ok": ok, "observation": obs, "key": key_name, "engine": "keystroke"}
