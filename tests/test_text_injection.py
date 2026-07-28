"""Unit tests for legacy input.txt reader (deprecated; prefer task_queue.json)."""

from __future__ import annotations

from pathlib import Path

from dana.core_agent import pop_text_injection


def test_pop_text_injection_reads_and_clears(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text(
        "I need you to build a tool that checks my current CPU and RAM usage",
        encoding="utf-8",
    )
    text = pop_text_injection(path=path)
    assert text is not None
    assert "CPU and RAM" in text
    assert path.read_text(encoding="utf-8") == ""
    assert pop_text_injection(path=path) is None
    print("[PASS] pop_text_injection reads + clears")


def test_pop_text_injection_empty_and_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert pop_text_injection(path=missing) is None
    empty = tmp_path / "input.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert pop_text_injection(path=empty) is None
    print("[PASS] pop_text_injection empty/missing")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_pop_text_injection_reads_and_clears(Path(td))
        test_pop_text_injection_empty_and_missing(Path(td))
    print("OK")
