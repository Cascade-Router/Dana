"""Safe stdio under ``pythonw.exe`` (Windows windowed launches).

``pythonw`` sets ``sys.stdout`` / ``sys.stderr`` to ``None``. Libraries that
call ``.write()`` (tqdm, transformers, sounddevice, print flush paths) then
crash with ``'NoneType' object has no attribute 'write'``.

Call ``ensure_stdio()`` as early as possible in every process entry point.
Importing ``donna`` also installs this once via ``donna/__init__.py``.
"""

from __future__ import annotations

import sys
from typing import Any


class NullStdio:
    """Minimal file-like object that absorbs writes (no console attached)."""

    encoding = "utf-8"
    errors = "replace"
    closed = False
    name = "<donna-null-stdio>"

    def write(self, data: Any) -> int:
        if data is None:
            return 0
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        return len(str(data))

    def writelines(self, lines: Any) -> None:
        for line in lines or ():
            self.write(line)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("pythonw has no console fileno")

    def reconfigure(self, **_kwargs: Any) -> None:
        return None

    @property
    def buffer(self) -> NullStdio:
        return self


_installed = False


def ensure_stdio() -> bool:
    """Replace ``None`` stdout/stderr. Idempotent. Returns True if patched."""
    global _installed
    patched = False
    if sys.stdout is None:
        sys.stdout = NullStdio()  # type: ignore[assignment]
        patched = True
    if sys.stderr is None:
        sys.stderr = NullStdio()  # type: ignore[assignment]
        patched = True
    _installed = _installed or patched
    return patched


# Alias used by older call sites / logging helpers.
ensure_stdio_for_pythonw = ensure_stdio
