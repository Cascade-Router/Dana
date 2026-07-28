"""Ambient shell / terminal error watchdog for Dānā.

Pure parsing and detection — no GUI dependency. Notifications and planner
handoff are injected via callbacks (see ``donna.ui.notifications``).

Default: **off**. Ambient monitoring is opt-in via the tray menu so users are
not surprised by toasts from unrelated terminal noise.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# (trace_window, summary) — unit tests mock this; production wires toast+planner.
ErrorCallback = Callable[[str, str], None]

# Default off: ambient shell monitoring is an explicit opt-in.
DEFAULT_ENABLED = False

_PREF_FILENAME = "shell_watchdog.json"

# Common failure signatures (compiled once).
_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError\b"),
    re.compile(r"\b(?:ERROR|Error|Exception)\b"),
    re.compile(r"^={5,}\s*FAILURES\s*={5,}", re.MULTILINE),
    re.compile(r"^={5,}\s*ERRORS\s*={5,}", re.MULTILINE),
    re.compile(r"^FAILED\s+\S+", re.MULTILINE),
    re.compile(r"^=+\s+\d+\s+failed", re.MULTILINE | re.IGNORECASE),
    re.compile(r"short test summary info", re.IGNORECASE),
    re.compile(r"Exit status\s+[1-9]\d*", re.IGNORECASE),
    re.compile(r"exited with (?:code|status)\s+[1-9]\d*", re.IGNORECASE),
    re.compile(r"nonzero exit", re.IGNORECASE),
    re.compile(r"Process finished with exit code [1-9]\d*"),
)

_SOURCE_FILE_RE = re.compile(
    r"""(?:File\s+"([^"]+\.py)"|([\w./\\-]+\.py)(?::\d+)?)""",
)

_TRACE_WINDOW = 15


def _pref_path() -> Path:
    """User-local preference file (does not touch ``donna.paths``)."""
    appdata = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if appdata:
        base = Path(appdata) / "Donna"
    else:
        base = Path.home() / ".config" / "donna"
    return base / _PREF_FILENAME


def is_shell_watchdog_enabled() -> bool:
    """Return persisted enable flag; default off when unset/unreadable."""
    path = _pref_path()
    try:
        if not path.is_file():
            return DEFAULT_ENABLED
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "enabled" in data:
            return bool(data["enabled"])
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_ENABLED


def set_shell_watchdog_enabled(enabled: bool) -> None:
    """Persist the tray toggle (best-effort; never raises)."""
    path = _pref_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"enabled": bool(enabled)}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def check_shell_watchdog_status(_item: Any = None) -> bool:
    """pystray ``checked=`` binder for ``Enable Shell Watchdog``."""
    try:
        return bool(is_shell_watchdog_enabled())
    except Exception:  # noqa: BLE001
        return False


def toggle_shell_watchdog(icon: Any = None, _item: Any = None) -> None:
    """Flip enable flag, refresh tray checkmark, and sync the shared instance."""
    try:
        new_state = not is_shell_watchdog_enabled()
        set_shell_watchdog_enabled(new_state)
        wd = get_shared_watchdog()
        wd.set_enabled(new_state)
    except Exception:  # noqa: BLE001
        return

    if icon is not None:
        try:
            update = getattr(icon, "update_menu", None)
            if callable(update):
                update()
        except Exception:  # noqa: BLE001
            pass


def extract_trace_window(lines: Sequence[str], match_index: int, *, window: int = _TRACE_WINDOW) -> str:
    """Return up to ``window`` lines centered on the matched error line."""
    if not lines:
        return ""
    n = len(lines)
    half = max(1, window // 2)
    start = max(0, match_index - half)
    end = min(n, start + window)
    start = max(0, end - window)
    return "\n".join(lines[start:end])


def summarize_error(trace: str, matched_line: str = "") -> str:
    """Build a short toast-friendly summary from a trace window."""
    source = None
    for m in _SOURCE_FILE_RE.finditer(trace or ""):
        source = m.group(1) or m.group(2)
    if source:
        source = Path(source.replace("\\", "/")).name

    kind = "Shell error"
    blob = f"{matched_line}\n{trace}"
    for label, pat in (
        ("ModuleNotFoundError", r"ModuleNotFoundError"),
        ("ImportError", r"ImportError"),
        ("Traceback", r"Traceback \(most recent call last\)"),
        ("pytest failure", r"(?:FAILURES|FAILED\s|\d+\s+failed)"),
        ("Nonzero exit", r"(?:Exit status|exited with|nonzero exit|exit code [1-9])"),
    ):
        if re.search(pat, blob, re.IGNORECASE):
            kind = label
            break

    if source:
        return f"{kind} detected in {source}"
    return f"{kind} detected"


def find_error_matches(text: str) -> list[tuple[int, str]]:
    """Return ``(line_index, matched_line)`` for each error signature hit."""
    if not text:
        return []
    lines = text.splitlines()
    hits: list[tuple[int, str]] = []
    # Prefer scanning the whole buffer for multiline patterns, then map to lines.
    for pat in _ERROR_PATTERNS:
        for m in pat.finditer(text):
            # Map character offset → line index.
            line_idx = text.count("\n", 0, m.start())
            if 0 <= line_idx < len(lines):
                hits.append((line_idx, lines[line_idx]))
    # Deduplicate by line index, keep first.
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for idx, line in sorted(hits, key=lambda t: t[0]):
        if idx in seen:
            continue
        seen.add(idx)
        unique.append((idx, line))
    return unique


class ShellWatchdog:
    """Listen to shell/log text and emit error events via an injectable sink."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        on_error: ErrorCallback | None = None,
        dedupe: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._enabled = DEFAULT_ENABLED if enabled is None else bool(enabled)
        self._on_error = on_error
        self._dedupe = bool(dedupe)
        self._seen: set[str] = set()
        self._line_buf: list[str] = []

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def set_on_error(self, callback: ErrorCallback | None) -> None:
        with self._lock:
            self._on_error = callback

    def clear_seen(self) -> None:
        with self._lock:
            self._seen.clear()

    def feed_line(self, line: str) -> list[tuple[str, str]]:
        """Ingest one output line; return list of ``(trace, summary)`` emitted."""
        return self.feed_text((line or "") + ("\n" if line and not line.endswith("\n") else ""))

    def feed_text(self, text: str) -> list[tuple[str, str]]:
        """Ingest a chunk / buffer of terminal or log text."""
        with self._lock:
            if not self._enabled:
                return []
            if not text:
                return []
            # Keep a rolling context so windows can look backward.
            new_lines = text.splitlines()
            self._line_buf.extend(new_lines)
            # Cap buffer to avoid unbounded growth (keep last ~200 lines).
            if len(self._line_buf) > 200:
                self._line_buf = self._line_buf[-200:]
            return self._emit_from_lines(list(self._line_buf), scan_from=max(0, len(self._line_buf) - len(new_lines)))

    def process_buffer(self, text: str) -> list[tuple[str, str]]:
        """Scan a complete buffer without mutating the rolling line buffer."""
        with self._lock:
            if not self._enabled:
                return []
            lines = (text or "").splitlines()
            return self._emit_from_lines(lines, scan_from=0)

    def feed_lines(self, lines: Iterable[str]) -> list[tuple[str, str]]:
        return self.feed_text("\n".join(lines) + "\n")

    def _emit_from_lines(self, lines: list[str], *, scan_from: int) -> list[tuple[str, str]]:
        if not lines:
            return []
        blob = "\n".join(lines)
        emitted: list[tuple[str, str]] = []
        for idx, matched_line in find_error_matches(blob):
            if idx < scan_from:
                continue
            trace = extract_trace_window(lines, idx)
            summary = summarize_error(trace, matched_line)
            fingerprint = f"{matched_line}\n{trace}"
            if self._dedupe:
                if fingerprint in self._seen:
                    continue
                self._seen.add(fingerprint)
                if len(self._seen) > 64:
                    # Drop oldest-ish entries (set order is insertion-ordered on 3.7+).
                    self._seen = set(list(self._seen)[-32:])
            emitted.append((trace, summary))
            cb = self._on_error
            if cb is not None:
                try:
                    cb(trace, summary)
                except Exception:  # noqa: BLE001
                    pass
            # One toast/event per feed to avoid ambient spam from multi-pattern hits.
            break
        return emitted


_shared: ShellWatchdog | None = None
_shared_lock = threading.Lock()


def get_shared_watchdog() -> ShellWatchdog:
    """Process-wide watchdog instance (tray + producers share this)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ShellWatchdog(enabled=is_shell_watchdog_enabled())
            # Wire production notification/planner path when available.
            try:
                from donna.ui.notifications import make_watchdog_error_handler

                _shared.set_on_error(make_watchdog_error_handler())
            except Exception:  # noqa: BLE001
                pass
        return _shared


def reset_shared_watchdog_for_tests() -> None:
    """Drop the singleton so unit tests start clean."""
    global _shared
    with _shared_lock:
        _shared = None
