"""Foundational Clipboard Actuator — safe, size-bounded OS clipboard I/O.

Completes Milestone 2 (Workspace Orchestration): once a window is focused
and a select-all + copy shortcut has fired (``dana.tools.keyboard_actuator``),
the fastest and most faithful way to extract its text is the OS clipboard
rather than lossy vision OCR. This wraps the raw Win32 clipboard primitives
in ``dana.tools.os_control`` (``read_clipboard_text``, ``write_clipboard_text``)
with size limits and, for the mutating write path, the same
dry-run/rate-limit/kill-switch safety pipeline as the other actuators.

Safety:
  - Reads are side-effect-free (they never change OS state), so they are
    NOT gated by ``DANA_OS_DRY_RUN`` or the module-wide rate limit —
    matching the precedent set by the read-only ``list_active_windows``
    tool. They ARE size-bounded: content beyond ``_MAX_READ_CHARS`` is
    truncated (reported via ``truncated``) rather than returned in full, to
    avoid flooding the caller/LLM context with an unbounded clipboard blob.
  - Writes mutate shared OS state, so they go through the same pipeline as
    every other actuator: validate size (reject outright rather than
    silently truncate — a caller should never end up writing less than it
    asked for) -> rate limit -> ``DANA_OS_DRY_RUN`` short-circuit -> best
    effort kill-switch check -> real ``SetClipboardData`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dana.tools.rate_limiter import get_limiter

_MAX_READ_CHARS = 200_000
_MAX_WRITE_CHARS = 200_000

_limiter = get_limiter("clipboard")


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second write actuation inside the cooldown window.

    Shared implementation: see ``dana.tools.rate_limiter``.
    """
    return _limiter.check_and_mark()


@dataclass
class ClipboardActuator:
    """Size-bounded clipboard read + rate-limited, dry-run-safe clipboard write.

    ``read_fn``/``write_fn`` default to the real ``dana.tools.os_control``
    Win32 clipboard backend; tests inject stubs so the pipeline runs with
    no real clipboard touched.
    """

    read_fn: Callable[[], str | None] | None = None
    write_fn: Callable[[str], None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _read(self) -> str | None:
        if self.read_fn is not None:
            return self.read_fn()
        from dana.tools.os_control import read_clipboard_text

        return read_clipboard_text()

    def _write(self, text: str) -> None:
        if self.write_fn is not None:
            self.write_fn(text)
            return
        from dana.tools.os_control import write_clipboard_text

        write_clipboard_text(text)

    def read_text(self, *, max_chars: int = _MAX_READ_CHARS) -> dict[str, Any]:
        """Read plaintext off the system clipboard, capped to ``max_chars``.

        Never gated by dry-run or the rate limit — reading has no OS side
        effects, so it always executes for real (tests inject ``read_fn``
        instead of relying on dry-run to avoid touching real hardware).

        Returns a result dict (``ok``, and on success ``text``/``empty``/
        ``truncated``/``chars``; on failure ``error``). Never raises.
        """
        self.events.clear()
        try:
            raw = self._read()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"clipboard read failed: {exc}"}

        text = raw or ""
        cap = max(1, int(max_chars))
        truncated = len(text) > cap
        if truncated:
            text = text[:cap]

        self.events.append({"event": "read", "chars": len(text), "truncated": truncated})
        return {
            "ok": True,
            "text": text,
            "empty": not bool(text.strip()),
            "truncated": truncated,
            "chars": len(text),
        }

    def write_text(self, text: str, *, max_chars: int = _MAX_WRITE_CHARS) -> dict[str, Any]:
        """Replace the system clipboard's contents with ``text``.

        Rejects outright (no partial write) if ``text`` exceeds
        ``max_chars`` — better to fail closed than silently write less than
        requested.

        Returns a result dict (``ok``, and on success ``chars``/
        ``dry_run``; on failure ``error`` and optionally ``halted``). Never
        raises for expected failure modes (oversized text, rate limit,
        kill switch, backend failure).
        """
        self.events.clear()
        body = text if isinstance(text, str) else str(text or "")
        cap = max(1, int(max_chars))
        if len(body) > cap:
            return {
                "ok": False,
                "error": (
                    f"text too large ({len(body)} > {cap} chars); "
                    "refusing to write a partial clipboard"
                ),
            }

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason}

        if _dry_run():
            self.events.append({"event": "dry_run_write", "chars": len(body)})
            return {"ok": True, "chars": len(body), "dry_run": True}

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt"})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                }
        except Exception:  # noqa: BLE001
            pass

        try:
            self._write(body)
        except Exception as exc:  # noqa: BLE001
            self.events.append({"event": "write_failed", "error": str(exc)})
            return {"ok": False, "error": f"clipboard write failed: {exc}"}

        self.events.append({"event": "write", "chars": len(body)})
        return {"ok": True, "chars": len(body), "dry_run": False}


def read_clipboard_text() -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``ClipboardActuator``."""
    return ClipboardActuator().read_text()


def write_clipboard_text(text: str) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``ClipboardActuator``."""
    return ClipboardActuator().write_text(text)


__all__ = ("ClipboardActuator", "read_clipboard_text", "write_clipboard_text")
