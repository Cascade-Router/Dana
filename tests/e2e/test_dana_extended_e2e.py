"""Extended live e2e: vault ingest/search, chained OS write.

Mirrors ``core_agent`` tool-handler binding (same underlying actuators) without
importing the full agent monolith / LLM router.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dana.memory.vault import FakeEmbeddings, ingest_local_directory, search_vault
from dana.tools.actuators import execute_command, write_to_file
from dana.tools.powershell import execute_powershell
from dana.tools.schema import ToolCall

pytestmark = pytest.mark.e2e

_MOCK_CPP = """\
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class CascadeRouterMockNode : public rclcpp::Node {
public:
  CascadeRouterMockNode() : Node("cascade_router_mock_node") {
    pub_ = this->create_publisher<std_msgs::msg::String>("cascade_status", 10);
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};
"""


def _invoke_live_tool(tool_id: str, **arguments: object) -> str:
    """Dispatch like ``core_agent`` handlers → real actuators (no LLM/router)."""
    call = ToolCall(tool_id=tool_id, arguments=dict(arguments))
    if tool_id == "write_to_file":
        filepath = call.arguments.get("filepath")
        if filepath is None or not str(filepath).strip():
            return "ERROR: missing filepath"
        content = call.arguments.get("content")
        return write_to_file(str(filepath), "" if content is None else str(content))
    if tool_id == "execute_powershell":
        command = call.arguments.get("command")
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        return execute_powershell(str(command))
    if tool_id == "execute_command":
        command = call.arguments.get("command")
        if command is None or not str(command).strip():
            return "ERROR: missing command"
        timeout_raw = call.arguments.get("timeout", 15)
        try:
            timeout_sec = int(timeout_raw) if timeout_raw is not None else 15
        except (TypeError, ValueError):
            timeout_sec = 15
        return execute_command(str(command), timeout=timeout_sec)
    if tool_id == "ingest_local_directory":
        path = call.arguments.get("path")
        if path is None or not str(path).strip():
            return "ERROR: missing path for ingest_local_directory"
        persist = call.arguments.get("persist_directory")
        embeddings = call.arguments.get("embeddings")
        kwargs: dict[str, object] = {}
        if persist is not None:
            kwargs["persist_directory"] = persist
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        return ingest_local_directory(str(path).strip(), **kwargs)  # type: ignore[arg-type]
    if tool_id == "search_vault":
        query = call.arguments.get("query")
        if query is None or not str(query).strip():
            return "ERROR: missing query for search_vault"
        n_raw = call.arguments.get("n_results", 5)
        try:
            n_results = int(n_raw) if n_raw is not None else 5
        except (TypeError, ValueError):
            n_results = 5
        persist = call.arguments.get("persist_directory")
        embeddings = call.arguments.get("embeddings")
        kwargs = {}
        if persist is not None:
            kwargs["persist_directory"] = persist
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        return search_vault(
            str(query).strip(), n_results=n_results, **kwargs  # type: ignore[arg-type]
        )
    return f"ERROR: unknown live tool {tool_id!r}"


def _powershell_available() -> bool:
    return shutil.which("powershell") is not None


def _desktop_or_temp() -> Path:
    desktop = Path.home() / "Desktop"
    if not desktop.is_dir():
        desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    if desktop.is_dir():
        return desktop
    return Path(os.environ.get("TEMP", str(Path.home())))


def test_vault_and_cpp_parsing(tmp_path: Path) -> None:
    """Ingest ROS2-ish mock_node.cpp then search vault for CascadeRouterMockNode."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "mock_node.cpp").write_text(_MOCK_CPP, encoding="utf-8")
    vault_dir = tmp_path / "vault"
    embeddings = FakeEmbeddings(dim=32)

    ingest_out = _invoke_live_tool(
        "ingest_local_directory",
        path=str(src),
        persist_directory=vault_dir,
        embeddings=embeddings,
    )
    assert ingest_out.startswith("OK: ingested "), ingest_out
    assert "0 chunks" not in ingest_out

    search_out = _invoke_live_tool(
        "search_vault",
        query="CascadeRouterMockNode",
        n_results=5,
        persist_directory=vault_dir,
        embeddings=embeddings,
    )
    assert "CascadeRouterMockNode" in search_out, search_out


@pytest.mark.skipif(
    os.name != "nt" or not _powershell_available(),
    reason="Chained OS manipulation requires Windows + PowerShell",
)
def test_chained_os_manipulation(tmp_path: Path) -> None:
    """PS list → write_to_file dana_diagnostic.txt → read first 3 lines; cleanup."""
    out_dir = _desktop_or_temp()
    diag = out_dir / "dana_diagnostic.txt"
    try:
        list_out = _invoke_live_tool(
            "execute_powershell",
            command="Get-Process | Select-Object -First 15 Name,Id | Format-Table -AutoSize | Out-String",
        )
        if list_out.startswith("ERROR:") or not (list_out or "").strip():
            list_out = _invoke_live_tool(
                "execute_command",
                command="Get-ChildItem | Select-Object -First 15 Name,Length | Format-Table -AutoSize | Out-String",
                timeout=30,
            )
        assert not list_out.startswith("ERROR:"), list_out
        # Strip actuator observation wrapper if present; keep processable text.
        content = list_out
        if "stdout:" in content:
            after = content.split("stdout:", 1)[1]
            if "stderr:" in after:
                after = after.split("stderr:", 1)[0]
            content = after.strip()
        assert content, list_out

        write_out = _invoke_live_tool(
            "write_to_file",
            filepath=str(diag),
            content=content + "\n",
        )
        assert write_out.startswith("OK:"), write_out
        assert diag.is_file()

        text = diag.read_text(encoding="utf-8", errors="replace")
        assert text.strip(), "dana_diagnostic.txt is empty"
        first_three = "\n".join(text.splitlines()[:3])
        assert first_three.strip(), first_three
    finally:
        if diag.exists():
            try:
                diag.unlink()
            except OSError:
                pass
