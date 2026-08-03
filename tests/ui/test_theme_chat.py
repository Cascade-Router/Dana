"""Theme tokens + chat bubble headless smoke tests."""

from __future__ import annotations

import pytest

from dana.ui import theme as T


def test_theme_tokens_slate_palette() -> None:
    assert T.BG == "#0a0e17"
    assert T.CARD == "#131b2e"
    assert T.BORDER == "#1e293b"
    assert T.TEXT == "#F8FAFC"
    assert T.MUTED == "#94A3B8"
    assert T.ACCENT == "#10b981"
    assert T.EMERALD == "#10b981"
    assert T.ROSE == "#F43F5E"


def test_ui_sources_drop_legacy_cyan() -> None:
    """Live dashboard surfaces must not hardcode legacy cyan accents."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "dana" / "ui"
    banned = ("#00ADB5", "#00adb5", "#008E95", "#00a8e8", "#0284c7")
    for path in root.rglob("*.py"):
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
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
