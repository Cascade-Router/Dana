"""AST outline / symbol extraction for Python and C++."""

from __future__ import annotations

from pathlib import Path

from dana.paths import PROJECT_ROOT
from dana.tools.ast_tools import get_file_outline, get_symbol_definition

_DIR = Path(PROJECT_ROOT) / "logs" / "ast_tools_fixtures"


def _write(name: str, body: str) -> str:
    _DIR.mkdir(parents=True, exist_ok=True)
    path = _DIR / name
    path.write_text(body, encoding="utf-8")
    return path.relative_to(Path(PROJECT_ROOT)).as_posix()


def test_python_outline_and_symbol() -> None:
    rel = _write(
        "sample.py",
        "import os\n\n"
        "class Foo:\n"
        "    def bar(self, x):\n"
        "        return x + 1\n\n"
        "def top(a, b):\n"
        "    return a + b\n",
    )
    outline = get_file_outline(rel)
    assert outline.startswith("OK: outline")
    assert "import os" in outline
    assert "class Foo" in outline
    assert "def bar" in outline
    assert "def top" in outline
    assert "return x + 1" not in outline  # bodies omitted from outline

    sym = get_symbol_definition(rel, "bar")
    assert sym.startswith("OK: symbol")
    assert "def bar" in sym
    assert "return x + 1" in sym

    qual = get_symbol_definition(rel, "Foo.bar")
    assert "def bar" in qual
    print("[PASS] python_outline_and_symbol")


def test_cpp_outline_and_symbol() -> None:
    rel = _write(
        "sample.cpp",
        '#include <vector>\n'
        "using namespace std;\n"
        "class Widget {\n"
        "public:\n"
        "  int size() { return 1; }\n"
        "};\n"
        "int add(int a, int b) {\n"
        "  return a + b;\n"
        "}\n",
    )
    outline = get_file_outline(rel)
    assert outline.startswith("OK: outline")
    assert "#include" in outline
    assert "class Widget" in outline
    assert "fn add()" in outline or "add" in outline

    sym = get_symbol_definition(rel, "add")
    assert sym.startswith("OK: symbol")
    assert "int add" in sym
    assert "return a + b" in sym

    widget = get_symbol_definition(rel, "Widget")
    assert "class Widget" in widget
    print("[PASS] cpp_outline_and_symbol")


def test_worker_registry_prefers_outline() -> None:
    from dana.graph.nodes.worker import WORKER_TOOL_REGISTRY, run_worker
    from dana.graph.state import empty_worker_state

    assert WORKER_TOOL_REGISTRY[0] == "get_file_outline"
    assert WORKER_TOOL_REGISTRY.index("get_file_outline") < WORKER_TOOL_REGISTRY.index(
        "read_local_file"
    )

    rel = _write("nav.py", "def alpha():\n    return 42\n")
    worker = empty_worker_state(1, f"Read {rel} and map its structure")
    reads: list[str] = []

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        reads.append(action)
        return f"OK: {action}"

    finished = run_worker(worker, tool_fn=tool_fn)
    tools_used = [o.get("tool") for o in finished.get("tool_outputs") or []]
    assert tools_used[0] == "get_file_outline"
    assert "read_local_file" not in tools_used
    assert reads == []
    assert finished["status"] == "completed"
    print("[PASS] worker_registry_prefers_outline")


if __name__ == "__main__":
    test_python_outline_and_symbol()
    test_cpp_outline_and_symbol()
    test_worker_registry_prefers_outline()
    print("\nAll ast_tools tests passed.")
