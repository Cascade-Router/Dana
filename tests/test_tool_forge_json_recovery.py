"""Smoke tests for Tool Forge coder JSON recovery."""

from __future__ import annotations

from dana.swarm.tool_forge_template import (
    assemble_forged_tool,
    extract_coder_json,
    normalize_coder_payload,
)


def test_soft_extract_unescaped_newlines() -> None:
    raw = (
        '{\n'
        '  "tool_name": "check_cpu_ram",\n'
        '  "description": "Report CPU and RAM",\n'
        '  "docstring": "Report usage",\n'
        '  "python_code": "import psutil\n'
        "return f'cpu={psutil.cpu_percent()}'\"\n"
        "}"
    )
    data = extract_coder_json(raw)
    assert data is not None
    assert data.get("tool_name") == "check_cpu_ram"
    assert "psutil" in str(data.get("python_code") or "")
    print("[PASS] soft extract unescaped newlines")


def test_python_body_fallback() -> None:
    raw = (
        "def check_cpu_ram(text: str = '', filepath: str = '') -> str:\n"
        "    return 'ok'\n"
    )
    data = extract_coder_json(raw)
    assert data is not None
    payload = normalize_coder_payload(data, fallback_name="check_cpu_ram")
    code = assemble_forged_tool(
        tool_name=payload["tool_name"],
        docstring=payload["docstring"],
        python_code=payload["python_code"],
        description=payload["description"],
    )
    assert "@tool" in code
    assert "def check_cpu_ram" in code
    print("[PASS] python body fallback")


if __name__ == "__main__":
    test_soft_extract_unescaped_newlines()
    test_python_body_fallback()
    print("OK")
