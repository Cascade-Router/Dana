"""AST Sanitizer — missing stdlib import injection / unused import cleanup."""

from __future__ import annotations

from pathlib import Path

from dana.graph.nodes.ast_sanitizer import (
    sanitize_python_file,
    sanitize_python_source,
)


def test_injects_missing_math_and_deque_imports() -> None:
    src = """
def hypotenuse(a, b):
    q = deque([a, b])
    return math.sqrt(a * a + b * b)
"""
    fixed, report = sanitize_python_source(src)
    assert report["changed"] is True
    assert any("math" in x for x in report["added_imports"])
    assert any("deque" in x for x in report["added_imports"])
    # Must parse and reference the injected names.
    assert "import math" in fixed or "math" in fixed
    assert "deque" in fixed
    compile(fixed, "<sanitized>", "exec")


def test_removes_unused_dangling_imports() -> None:
    src = """
import os
import sys

def version():
    return sys.version
"""
    fixed, report = sanitize_python_source(src)
    assert report["changed"] is True
    assert any("os" in x for x in report["removed_imports"])
    assert "import os" not in fixed
    assert "sys" in fixed
    compile(fixed, "<sanitized>", "exec")


def test_sanitize_python_file_writes_disk(tmp_path: Path) -> None:
    path = tmp_path / "widget.py"
    path.write_text(
        "def f(x):\n    return math.ceil(x)\n",
        encoding="utf-8",
    )
    report = sanitize_python_file(path)
    assert report["changed"] is True
    body = path.read_text(encoding="utf-8")
    assert "import math" in body
    compile(body, str(path), "exec")


def test_noop_when_imports_already_present() -> None:
    src = """
import math
from collections import deque

def f(a):
    deque([a])
    return math.sqrt(a)
"""
    fixed, report = sanitize_python_source(src)
    assert report["changed"] is False
    assert fixed == src
