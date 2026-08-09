"""Stage 6.1 — Ghost Typist Operator (closed-loop Sense-Evaluate-Act).

Refactors the stealth SendInput typing path from ``dana.tools.os_control``
into a deterministic control agent that:

* types with stochastic human cadence (40–120 ms between keystrokes)
* processes text in small chunks (15–20 chars)
* senses typed ``perception.ocr`` from the Blackboard after each chunk
* pauses immediately on focus-loss / popup / drastic visual change

Hotkey arming defaults to F9 (Win32 GetAsyncKeyState). Set
``DANA_GHOST_SKIP_HOTKEY=1`` for headless/tests. ``DANA_OS_DRY_RUN=1``
skips real OS injection while still exercising the SEA loop.
"""

from __future__ import annotations

import difflib
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Reuse the production stealth keystroke engine (SendInput scan codes —
# hardware-level, stronger than pyautogui/pynput for anti-cheat surfaces).
from dana.tools import os_control as _osc

CHUNK_SIZE_MIN = 15
CHUNK_SIZE_MAX = 20
DELAY_MIN_S = 0.040
DELAY_MAX_S = 0.120
DEFAULT_HOTKEY = "f9"
VISUAL_SIMILARITY_FLOOR = 0.40

_UNSAFE_VISUAL_RE = re.compile(
    r"(?is)\b("
    r"pop[\s-]?up|modal|dialog|alert\s+box|focus\s+lost|lost\s+focus|"
    r"alt[\s-]?tab|lock\s+screen|uac|permission\s+denied|screen\s+lock|"
    r"different\s+window|window\s+changed|notepad\s+closed"
    r")\b"
)


def _dry_run() -> bool:
    return os.environ.get("DANA_OS_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _skip_hotkey() -> bool:
    return os.environ.get("DANA_GHOST_SKIP_HOTKEY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chunk_text(text: str, *, size: int | None = None) -> list[str]:
    """Split ``text`` into 15–20 character chunks (last chunk may be shorter)."""
    blob = text if text is not None else ""
    if not blob:
        return []
    n = int(size) if size is not None else random.randint(CHUNK_SIZE_MIN, CHUNK_SIZE_MAX)
    n = max(CHUNK_SIZE_MIN, min(CHUNK_SIZE_MAX, n))
    return [blob[i : i + n] for i in range(0, len(blob), n)]


def _stochastic_delay() -> None:
    time.sleep(random.uniform(DELAY_MIN_S, DELAY_MAX_S))


def evaluate_visual_guard(
    *,
    baseline: str,
    current: str,
) -> tuple[bool, str]:
    """Return ``(safe, reason)``. ``safe=False`` means the operator must pause."""
    cur = (current or "").strip()
    base = (baseline or "").strip()
    if not cur and base:
        return False, "visual_context_empty_after_baseline"
    if _UNSAFE_VISUAL_RE.search(cur):
        return False, "unsafe_visual_keyword"
    if base and cur and len(cur) >= 24:
        ratio = difflib.SequenceMatcher(None, base.lower(), cur.lower()).ratio()
        if ratio < VISUAL_SIMILARITY_FLOOR:
            return False, f"drastic_visual_change ratio={ratio:.2f}"
    return True, "ok"


@dataclass
class GhostTypistOperator:
    """Closed-loop stealth typist control agent."""

    hotkey: str = DEFAULT_HOTKEY
    chunk_size: int | None = None
    read_visual: Callable[[], str] | None = None
    type_char: Callable[[str], bool] | None = None
    wait_hotkey_fn: Callable[[str, float | None], bool] | None = None
    chars_typed: int = 0
    chunks_done: int = 0
    paused: bool = False
    pause_reason: str = ""
    baseline_visual: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)

    def _sense(self) -> str:
        if self.read_visual is not None:
            try:
                return str(self.read_visual() or "")
            except Exception as exc:  # noqa: BLE001
                return f"(sense_error: {exc})"
        try:
            from dana.memory import read_perception_ocr_text

            # Prefer OCR for focus/popup safety; empty OCR = unknown (do not
            # fall back to YOLO object labels — they are not UI text).
            return read_perception_ocr_text() or ""
        except Exception as exc:  # noqa: BLE001
            return f"(sense_error: {exc})"

    def _type_one(self, ch: str) -> bool:
        if self.type_char is not None:
            return bool(self.type_char(ch))
        if _dry_run():
            _stochastic_delay()
            return True
        # Prefer OS-control stealth SendInput (existing typist backend).
        try:
            # Temporarily widen cadence to Stage 6.1 40–120 ms window.
            old_min, old_max = _osc._HUMAN_DELAY_MIN, _osc._HUMAN_DELAY_MAX
            _osc._HUMAN_DELAY_MIN = DELAY_MIN_S
            _osc._HUMAN_DELAY_MAX = DELAY_MAX_S
            try:
                return bool(_osc._type_char_stealth(ch))
            finally:
                _osc._HUMAN_DELAY_MIN = old_min
                _osc._HUMAN_DELAY_MAX = old_max
        except Exception:  # noqa: BLE001
            return False

    def wait_for_hotkey(self, *, timeout_s: float | None = None) -> bool:
        """Block until the arming hotkey (default F9) is pressed."""
        if _skip_hotkey() or _dry_run():
            self.events.append({"event": "hotkey_skipped", "key": self.hotkey})
            return True
        if self.wait_hotkey_fn is not None:
            ok = bool(self.wait_hotkey_fn(self.hotkey, timeout_s))
            self.events.append({"event": "hotkey", "key": self.hotkey, "ok": ok})
            return ok
        ok = _wait_hotkey_win32(self.hotkey, timeout_s=timeout_s)
        self.events.append({"event": "hotkey", "key": self.hotkey, "ok": ok})
        return ok

    def run(self, text: str, *, wait_hotkey: bool = True) -> dict[str, Any]:
        """Sense-Evaluate-Act loop over ``text``; return a status dict."""
        self.chars_typed = 0
        self.chunks_done = 0
        self.paused = False
        self.pause_reason = ""
        self.events.clear()

        body = text if text is not None else ""
        if not str(body):
            return {
                "ok": False,
                "error": "empty text",
                "chars_typed": 0,
                "paused": False,
                "engine": "ghost_typist",
            }

        if wait_hotkey:
            armed = self.wait_for_hotkey()
            if not armed:
                return {
                    "ok": False,
                    "error": f"hotkey {self.hotkey!r} not received",
                    "chars_typed": 0,
                    "paused": False,
                    "engine": "ghost_typist",
                }

        self.baseline_visual = self._sense()
        self.events.append(
            {
                "event": "baseline_sense",
                "visual_chars": len(self.baseline_visual),
            }
        )

        chunks = _chunk_text(body, size=self.chunk_size)
        # Stage 7.3 — mute mic/VAD while keystrokes are injected.
        try:
            from dana.memory.blackboard import set_is_typing

            set_is_typing(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            for idx, chunk in enumerate(chunks):
                # Stage 7.2 — hardware kill switch aborts mid-SEA.
                try:
                    from dana.middleware.kill_switch import halt_if_requested

                    if halt_if_requested():
                        self.paused = True
                        self.pause_reason = "halted by GLOBAL_HALT_EVENT"
                        self.events.append({"event": "halt", "index": idx})
                        return {
                            "ok": False,
                            "halted": True,
                            "error": "halted by GLOBAL_HALT_EVENT",
                            "chars_typed": self.chars_typed,
                            "chunks_done": self.chunks_done,
                            "chunks_total": len(chunks),
                            "paused": True,
                            "pause_reason": self.pause_reason,
                            "engine": "ghost_typist",
                            "events": list(self.events),
                        }
                except Exception:  # noqa: BLE001
                    pass

                # Stage 7.4 — yield to physical human input (soft pause).
                try:
                    from dana.middleware.human_yield import yield_check

                    yield_check(operator="ghost_typist")
                except Exception:  # noqa: BLE001
                    pass

                # Act — type this chunk with stochastic inter-key delays.
                for ch in chunk:
                    try:
                        from dana.middleware.kill_switch import halt_if_requested

                        if halt_if_requested():
                            self.paused = True
                            self.pause_reason = "halted by GLOBAL_HALT_EVENT"
                            return {
                                "ok": False,
                                "halted": True,
                                "error": "halted by GLOBAL_HALT_EVENT",
                                "chars_typed": self.chars_typed,
                                "chunks_done": self.chunks_done,
                                "chunks_total": len(chunks),
                                "paused": True,
                                "pause_reason": self.pause_reason,
                                "engine": "ghost_typist",
                                "events": list(self.events),
                            }
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        from dana.middleware.human_yield import yield_check

                        yield_check(operator="ghost_typist")
                    except Exception:  # noqa: BLE001
                        pass
                    if self._type_one(ch):
                        self.chars_typed += 1
                    # Extra stochastic gap even when backend already slept
                    # (dry-run / custom type_char paths).
                    if self.type_char is not None or _dry_run():
                        _stochastic_delay()

                self.chunks_done += 1
                # Sense
                visual = self._sense()
                # Evaluate
                safe, reason = evaluate_visual_guard(
                    baseline=self.baseline_visual,
                    current=visual,
                )
                self.events.append(
                    {
                        "event": "chunk",
                        "index": idx,
                        "chunk_len": len(chunk),
                        "safe": safe,
                        "reason": reason,
                        "visual_preview": (visual or "")[:120],
                    }
                )
                if not safe:
                    self.paused = True
                    self.pause_reason = reason
                    return {
                        "ok": False,
                        "paused": True,
                        "pause_reason": reason,
                        "chars_typed": self.chars_typed,
                        "chunks_done": self.chunks_done,
                        "chunks_total": len(chunks),
                        "engine": "ghost_typist",
                        "dry_run": _dry_run(),
                    }

            return {
                "ok": True,
                "paused": False,
                "chars_typed": self.chars_typed,
                "chunks_done": self.chunks_done,
                "chunks_total": len(chunks),
                "engine": "ghost_typist",
                "dry_run": _dry_run(),
            }
        finally:
            try:
                from dana.memory.blackboard import set_is_typing

                set_is_typing(False)
            except Exception:  # noqa: BLE001
                pass


def type_stealth_text(
    text: str,
    *,
    wait_hotkey: bool = True,
    hotkey: str = DEFAULT_HOTKEY,
) -> str:
    """Tool / actuator entry — run GhostTypistOperator and return observation."""
    op = GhostTypistOperator(hotkey=hotkey or DEFAULT_HOTKEY)
    result = op.run(text or "", wait_hotkey=wait_hotkey)
    if result.get("halted"):
        return f"HALTED: type_stealth_text — {result.get('error')}"
    if result.get("paused"):
        return (
            f"PAUSED: ghost_typist pause_reason={result.get('pause_reason')} "
            f"chars_typed={result.get('chars_typed')} "
            f"chunks={result.get('chunks_done')}/{result.get('chunks_total')}"
        )
    if not result.get("ok"):
        return f"ERROR: ghost_typist failed: {result.get('error')}"
    return (
        f"OK: type_stealth_text chars={result.get('chars_typed')} "
        f"chunks={result.get('chunks_done')} "
        f"engine={result.get('engine')} "
        f"dry_run={result.get('dry_run')}"
    )


def _wait_hotkey_win32(hotkey: str, *, timeout_s: float | None = None) -> bool:
    """Wait for a function key via GetAsyncKeyState (no pynput dependency)."""
    if os.name != "nt":
        # Non-Windows: optional pynput fallback.
        return _wait_hotkey_pynput(hotkey, timeout_s=timeout_s)

    vk_map = {
        "f8": 0x77,
        "f9": 0x78,
        "f10": 0x79,
        "f11": 0x7A,
        "f12": 0x7B,
    }
    vk = vk_map.get((hotkey or "f9").strip().lower(), 0x78)
    import ctypes

    user32 = ctypes.windll.user32
    deadline = None if timeout_s is None else (time.monotonic() + float(timeout_s))
    # Require a rising edge so a stuck key does not instantly arm.
    was_down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
    while True:
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        if down and not was_down:
            # Debounce release wait (short).
            time.sleep(0.05)
            return True
        was_down = down
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.04)


def _wait_hotkey_pynput(hotkey: str, *, timeout_s: float | None = None) -> bool:
    try:
        from pynput import keyboard
    except Exception:  # noqa: BLE001
        return False

    target = (hotkey or "f9").strip().lower()
    key_map = {
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11,
        "f12": keyboard.Key.f12,
    }
    want = key_map.get(target, keyboard.Key.f9)
    hit = {"ok": False}

    def _on_press(key: Any) -> bool | None:
        if key == want:
            hit["ok"] = True
            return False
        return None

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()
    listener.join(timeout=timeout_s)
    try:
        listener.stop()
    except Exception:  # noqa: BLE001
        pass
    return bool(hit["ok"])
