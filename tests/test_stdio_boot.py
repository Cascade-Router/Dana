"""pythonw-safe stdio boot — None stdout/stderr must never crash writers."""

from __future__ import annotations

import sys

from dana.stdio_boot import NullStdio, ensure_stdio


def test_ensure_stdio_replaces_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert ensure_stdio() is True
    assert sys.stdout is not None
    assert sys.stderr is not None
    assert sys.stdout.write("hello") == 5
    assert sys.stderr.write(b"abc") == 3
    sys.stdout.flush()
    assert sys.stdout.isatty() is False
    print("[PASS] ensure_stdio replaces None streams")


def test_ensure_stdio_idempotent_when_present(monkeypatch) -> None:
    real_out = sys.__stdout__
    real_err = sys.__stderr__
    monkeypatch.setattr(sys, "stdout", real_out)
    monkeypatch.setattr(sys, "stderr", real_err)
    # Should not replace a live console stream with NullStdio.
    before_out = sys.stdout
    assert ensure_stdio() is False
    assert sys.stdout is before_out
    print("[PASS] ensure_stdio leaves real streams alone")


def test_null_stdio_buffer_chain() -> None:
    n = NullStdio()
    assert n.buffer is n
    assert n.writable() is True
    n.writelines(["a", "b"])
    print("[PASS] NullStdio buffer/writelines")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
