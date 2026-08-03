"""Theme tokens + chat bubble headless smoke tests."""

from __future__ import annotations

import pytest

from dana.ui import theme as T


def test_theme_tokens_slate_palette() -> None:
    T.set_theme("Obsidian Mint")
    assert T.BG == "#0a0e17"
    assert T.CARD == "#131b2e"
    assert T.BORDER == "#1e293b"
    assert T.TEXT == "#F8FAFC"
    assert T.MUTED == "#94A3B8"
    assert T.ACCENT == "#10b981"
    assert T.EMERALD == "#10b981"
    assert T.ROSE == "#F43F5E"


def test_three_theme_catalog() -> None:
    names = set(T.THEME_NAMES)
    assert names == {"Obsidian Mint", "Cyber Amber", "Ghost Light"}
    mint = T.get_theme("Obsidian Mint")
    amber = T.get_theme("Cyber Amber")
    ghost = T.get_theme("Ghost Light")
    assert mint["bg"] == "#0a0e17" and mint["accent"] == "#10b981"
    assert amber["bg"] == "#070b14" and amber["accent"] == "#f59e0b"
    assert ghost["bg"] == "#f8fafc" and ghost["accent"] == "#4f46e5"
    assert ghost["text"] == "#0f172a"
    T.set_theme("Cyber Amber")
    assert T.active_theme_name() == "Cyber Amber"
    assert T.ACCENT == "#f59e0b"
    T.set_theme("Ghost Light")
    assert T.BG == "#f8fafc" and T.TEXT == "#0f172a"
    T.set_theme("Obsidian Mint")
    assert T.ACCENT == "#10b981"


def test_dana_theme_json_and_apply() -> None:
    """Global CTk theme file exists and loads mint primary accents."""
    import json

    T.ensure_theme_json_files()
    path = T.dana_theme_path("Obsidian Mint")
    assert path.is_file(), f"missing theme: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["CTkButton"]["fg_color"][1] == "#10b981"
    assert data["CTkButton"]["hover_color"][1] == "#059669"
    assert data["CTkSegmentedButton"]["selected_color"][1] == "#10b981"
    assert data["CTk"]["fg_color"][1] == "#0a0e17"
    amber_path = T.dana_theme_path("Cyber Amber")
    ghost_path = T.dana_theme_path("Ghost Light")
    assert amber_path.is_file()
    assert ghost_path.is_file()
    assert T.apply_dana_ctk_theme("Obsidian Mint") is True


def test_ui_sources_drop_legacy_cyan() -> None:
    """Live dashboard surfaces must not hardcode legacy cyan accents."""
    from pathlib import Path

    banned = ("#00ADB5", "#00adb5", "#008E95", "#00a8e8", "#0284c7")
    roots = (
        Path(__file__).resolve().parents[2] / "dana" / "ui",
        Path(__file__).resolve().parents[2] / "dana" / "core_agent.py",
    )
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in banned:
                assert token not in text, f"{path.name} still contains {token}"


def test_chat_bubble_view_headless() -> None:
    ctk = pytest.importorskip("customtkinter")
    from dana.ui.chat_view import ChatBubbleView, _classify_role

    assert _classify_role("User (Text)") == "user"
    assert _classify_role("Dana", "broker") == "dana"
    assert _classify_role("Vision Output", "vision") == "system"

    try:
        root = ctk.CTk()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        view = ChatBubbleView(root, wraplength=320)
        view.pack(fill="both", expand=True)
        box = view.transcript_box
        box.configure(state="normal")
        box.insert("1.0", "[Dana] hello\n\n")
        box.configure(state="disabled")
        assert "[Dana]" in str(box.get("1.0", "end"))
        view.append_bubble("User (Text)", "hi there", role="user")
        view.append_bubble("Dana", "hello **world**", agent_id="broker")
        view.append_bubble("Vision Output", "button @ (10,20)", role="system")
        assert len(view._bubbles) >= 3
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_gui_uses_theme_and_chat_view() -> None:
    try:
        from dana.core_agent import DonnaGUI, _UI_CANVAS, _UI_ACCENT
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI unavailable: {exc}")

    assert _UI_CANVAS == T.BG
    assert _UI_ACCENT == T.ACCENT

    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert app.transcript_box is not None
        assert getattr(app, "_chat_view", None) is not None
        raw = str(app.transcript_box.get("1.0", "end"))
        assert "[Dana]" in raw
        assert "Type below or say Dana" in raw
        assert app._engage_btn is not None
        assert app._standby_btn is not None
        # Theme-owned primary accents (no per-widget cyan/mint hardcodes).
        send_fg = str(app._chat_send_btn.cget("fg_color")).lower()
        assert "#10b981" in send_fg
        engage_fg = str(app._engage_btn.cget("fg_color")).lower()
        assert "#10b981" in engage_fg
        stop_fg = str(app.stop_donna_btn.cget("fg_color")).lower()
        assert "#f43f5e" in stop_fg
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
