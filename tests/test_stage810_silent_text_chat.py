"""Stage 8.10 — Dashboard silent text chat bar (STT bypass → LangGraph)."""

from __future__ import annotations


def test_chat_bar_widgets_and_standby_guard() -> None:
    from donna.core_agent import (
        DonnaGUI,
        clear_injected_question,
        pop_injected_question_ex,
        set_engine_engaged,
    )

    set_engine_engaged(False)
    clear_injected_question()
    app = DonnaGUI()
    try:
        assert app.chat_entry is not None
        assert app._chat_send_btn is not None
        assert "Type below or say Donna" in str(app.chat_entry.cget("placeholder_text"))
        accent = str(app._chat_send_btn.cget("fg_color")).lower()
        assert "#00adb5" in accent

        # Welcome banner once at build; Start Chat must not re-append it.
        box = app.transcript_box
        assert box is not None
        welcome = "Type below or say Donna, then speak."
        initial = str(box.get("1.0", "end"))
        assert welcome in initial
        assert initial.count(welcome) == 1
        app.engine_active = True
        set_engine_engaged(True)
        app._dashboard_start_chat()
        app._dashboard_start_chat()
        after = str(box.get("1.0", "end"))
        assert after.count(welcome) == 1
        set_engine_engaged(False)
        app.engine_active = False

        # Standby: Send aborts and does not inject.
        app.chat_entry.insert(0, "hello from text")
        app.submit_text_command()
        text, _src, _logged = pop_injected_question_ex()
        assert text is None
        assert app.engine_active is False
        warn = str(app._engine_warn_lbl.cget("text"))
        assert "Engage Engine" in warn
        # Abort is not a successful send — keep the draft for Engage → resend.
        assert "hello from text" in str(app.chat_entry.get())
    finally:
        set_engine_engaged(False)
        clear_injected_question()
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_submit_text_command_injects_when_engaged() -> None:
    from donna.core_agent import (
        DonnaGUI,
        clear_injected_question,
        is_recording,
        pop_injected_question_ex,
        set_engine_engaged,
    )

    set_engine_engaged(False)
    clear_injected_question()
    is_recording.clear()
    app = DonnaGUI()
    try:
        app.engage_engine()
        assert app.engine_active is True
        app.chat_entry.delete(0, "end")
        app.chat_entry.insert(0, "silent command for dana")
        app.submit_text_command()
        assert str(app.chat_entry.get()).strip() == ""
        text, source, already_logged = pop_injected_question_ex()
        assert text == "silent command for dana"
        assert source == "text"
        assert already_logged is True
        assert is_recording.is_set()
        # Transcript echo uses User (Text) label.
        tk_text = app._transcript_tk()
        assert tk_text is not None
        app.update_idletasks()
        # Flush after(0) transcript append.
        app.update()
        body = tk_text.get("1.0", "end")
        assert "User (Text)" in body
        assert "silent command for dana" in body
        assert "user_text" in set(tk_text.tag_names())
    finally:
        set_engine_engaged(False)
        clear_injected_question()
        is_recording.clear()
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
