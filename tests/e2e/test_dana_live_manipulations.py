"""Live e2e suite: OS write/exec, screen OCR, and Playwright fetch via real actuators.

Mirrors ``core_agent`` tool-handler binding (same underlying actuators) without
importing the full agent monolith.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from dana.tools.actuators import execute_command, write_to_file
from dana.tools.browser import fetch_webpage
from dana.tools.powershell import execute_powershell
from dana.tools.schema import ToolCall
from dana.tools.vision import analyze_visual_context

pytestmark = pytest.mark.e2e

_HELLO = "Hello from the Sandbox"
_VISION_MARKER = "VISION_GROUNDING_TEST_899"


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
    if tool_id == "analyze_visual_context":
        return analyze_visual_context()
    if tool_id == "fetch_webpage":
        url = call.arguments.get("url")
        if url is None or not str(url).strip():
            return "ERROR: missing url"
        return fetch_webpage(str(url))
    return f"ERROR: unknown live tool {tool_id!r}"


def _desktop_os_test_dir() -> Path:
    desktop = Path.home() / "Desktop"
    if not desktop.is_dir():
        desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    return desktop / "dana_os_test"


def _powershell_available() -> bool:
    return shutil.which("powershell") is not None


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    os.name != "nt" or not _powershell_available(),
    reason="Live OS manipulation requires Windows + PowerShell",
)
def test_os_manipulation() -> None:
    """Create Desktop/dana_os_test, write+run a script, assert sandbox hello stdout."""
    folder = _desktop_os_test_dir()
    script = folder / "hello_sandbox.py"
    try:
        write_out = _invoke_live_tool(
            "write_to_file",
            filepath=str(script),
            content=f'print("{_HELLO}")\n',
        )
        assert write_out.startswith("OK:"), write_out
        assert script.is_file()

        # Prefer execute_powershell (same handler path as core_agent).
        run_out = _invoke_live_tool(
            "execute_powershell",
            command=f'& python "{script}"',
        )
        if _HELLO not in run_out:
            # Fallback mirrors execute_command handler binding.
            run_out = _invoke_live_tool(
                "execute_command",
                command=f'python "{script}"',
                timeout=30,
            )
        assert _HELLO in run_out, run_out
        assert "returncode=0" in run_out or "stdout:" in run_out
    finally:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.skipif(
    os.name != "nt" or not _tesseract_available(),
    reason="Live vision OCR requires Windows + Tesseract",
)
def test_vision_manipulation() -> None:
    """Show a temporary CTk window on the main thread, OCR, assert marker text."""
    ctk = pytest.importorskip("customtkinter")

    root = ctk.CTk()
    try:
        root.title("Dānā Vision Grounding")
        root.attributes("-topmost", True)
        root.geometry("1100x320+80+80")
        label = ctk.CTkLabel(
            root,
            text=_VISION_MARKER,
            font=ctk.CTkFont(family="Consolas", size=36, weight="bold"),
            text_color="#FFFFFF",
            fg_color="#111111",
        )
        label.pack(expand=True, fill="both", padx=24, pady=24)
        root.update_idletasks()
        root.lift()
        root.focus_force()
        # Keep the window mapped/visible ~2s before OCR (pump Tk so it paints).
        deadline = time.time() + 2.0
        while time.time() < deadline:
            root.update()
            time.sleep(0.05)
        root.update()

        ocr = _invoke_live_tool("analyze_visual_context")
        assert not ocr.startswith("SYSTEM_ERROR:"), ocr
        # OCR may insert spaces/newlines; normalize for the unique marker.
        compact = "".join(ch for ch in ocr.upper() if ch.isalnum())
        marker_compact = "".join(ch for ch in _VISION_MARKER.upper() if ch.isalnum())
        assert marker_compact in compact or _VISION_MARKER in ocr, ocr
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.skipif(
    not _chromium_available(),
    reason="Live Playwright fetch requires Chromium installed",
)
def test_playwright_manipulation() -> None:
    """Fetch example.com via real Playwright actuator; assert Example Domain text."""
    out = _invoke_live_tool("fetch_webpage", url="https://example.com")
    assert not out.startswith("ERROR:"), out
    assert "Example Domain" in out
