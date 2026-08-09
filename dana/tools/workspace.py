"""list_active_windows / focus_window / press_keyboard_shortcut / read_clipboard /
write_clipboard — Milestone 2 workspace-orchestration tools.

Extends the physical OS actuator suite (mouse, keyboard, scroll, drag) with
basic window management (enumerate visible top-level windows and bring one
to the foreground before interacting with it, so click/type/drag tools land
on the intended application rather than whatever last had focus) and with
keyboard-shortcut + clipboard I/O (fire a combo like "ctrl+a"/"ctrl+c" and
read the result back verbatim — a large log or code file extracted via the
clipboard is exact, where vision OCR would be lossy).
"""

from __future__ import annotations


def list_active_windows() -> str:
    """Return a clean, readable list of currently visible application windows.

    Pipeline: ``dana.tools.os_control.get_active_windows`` (Win32
    ``EnumWindows`` + ``IsWindowVisible`` + ``GetWindowTextW``), formatted as
    one line per window (title + process id) in Z-order (topmost first).

    Returns ``"SUCCESS: ..."`` with the window list (or a "no windows"
    notice) on success, or ``"ERROR: ..."`` if the OS window list couldn't
    be read.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="list_active_windows")
    except Exception:  # noqa: BLE001
        pass

    try:
        from dana.tools.os_control import get_active_windows

        windows = get_active_windows()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: list_active_windows failed: {exc}"

    if not windows:
        return "SUCCESS: No visible application windows found."

    lines = [f"- {w.get('title')} (pid={w.get('pid')})" for w in windows]
    return "SUCCESS: Visible windows:\n" + "\n".join(lines)


def focus_window(target_description: str) -> str:
    """Bring the window whose title best matches ``target_description`` to the front.

    ``target_description`` is treated as a case-insensitive regular
    expression matched against every visible window's title (e.g.
    ``"Cursor"``, ``"chrome"``, ``"Untitled - Notepad"``); a window whose
    full title matches exactly wins over a partial substring match.
    Fails closed: nothing is focused if the pattern is invalid, no window
    matches, or Windows denies the focus-steal.

    Returns ``"SUCCESS: Focused '<title>'"`` on a confirmed focus change,
    ``"ERROR: ..."`` when matching or focusing fails, or ``"HALTED: ..."``
    if the global kill switch fired mid-action.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="focus_window")
    except Exception:  # noqa: BLE001
        pass

    pattern = str(target_description or "").strip()
    if not pattern:
        return "ERROR: focus_window requires a non-empty target_description"

    from dana.tools.window_actuator import WindowActuator

    result = WindowActuator().focus_by_title(pattern)
    if result.get("halted"):
        return f"HALTED: focus_window — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: focus_window failed to focus {pattern!r}: {result.get('error')}"

    title = result.get("window", {}).get("title")
    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return f"SUCCESS: Focused {title!r}{dry_note}"


def press_keyboard_shortcut(shortcut: str) -> str:
    """Press a keyboard shortcut/key combo, e.g. ``"ctrl+c"``, ``"alt+tab"``.

    Pipeline: ``dana.tools.keyboard_actuator.KeyboardActuator.execute_shortcut``
    parses ``shortcut`` on ``"+"``, resolves each key name to a Win32 VK, and
    presses them via ``dana.tools.os_control.press_key_combo`` (down in
    order, up in reverse order). Fails closed if any key name is
    unrecognized — nothing is pressed on a bad combo.

    Returns ``"SUCCESS: Pressed '<shortcut>'"`` on success, ``"ERROR: ..."``
    when parsing/pressing fails, or ``"HALTED: ..."`` if the global kill
    switch fired mid-action.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="press_keyboard_shortcut")
    except Exception:  # noqa: BLE001
        pass

    combo = str(shortcut or "").strip()
    if not combo:
        return "ERROR: press_keyboard_shortcut requires a non-empty shortcut"

    from dana.tools.keyboard_actuator import KeyboardActuator

    result = KeyboardActuator().execute_shortcut(combo)
    if result.get("halted"):
        return f"HALTED: press_keyboard_shortcut — {result.get('error')}"
    if not result.get("ok"):
        return (
            f"ERROR: press_keyboard_shortcut failed to press {combo!r}: "
            f"{result.get('error')}"
        )

    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return f"SUCCESS: Pressed {combo!r}{dry_note}"


def read_clipboard() -> str:
    """Read the current plaintext system clipboard, verbatim.

    Pipeline: ``dana.tools.clipboard_actuator.ClipboardActuator.read_text``
    (raw Win32 ``GetClipboardData``), capped to a safety size limit —
    content beyond the cap is truncated rather than returned in full.
    Always executes for real; reading has no OS side effects so it is not
    gated by ``DANA_OS_DRY_RUN``.

    Returns ``"SUCCESS: ..."`` with the clipboard text (or an "empty"
    notice) on success, or ``"ERROR: ..."`` if the clipboard couldn't be
    read.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="read_clipboard")
    except Exception:  # noqa: BLE001
        pass

    from dana.tools.clipboard_actuator import ClipboardActuator

    result = ClipboardActuator().read_text()
    if not result.get("ok"):
        return f"ERROR: read_clipboard failed: {result.get('error')}"
    if result.get("empty"):
        return "SUCCESS: Clipboard is empty."

    trunc_note = " (truncated)" if result.get("truncated") else ""
    text = result.get("text") or ""
    return f"SUCCESS: Clipboard text{trunc_note}:\n{text}"


def write_clipboard(text: str) -> str:
    """Write ``text`` to the system clipboard, replacing its contents.

    Pipeline: ``dana.tools.clipboard_actuator.ClipboardActuator.write_text``
    (raw Win32 ``SetClipboardData``) — rejects outright rather than
    truncating if ``text`` exceeds the size limit, and applies the standard
    dry-run/rate-limit/kill-switch actuator pipeline since writing mutates
    shared OS state.

    Returns ``"SUCCESS: Wrote N chars to clipboard"`` on success,
    ``"ERROR: ..."`` when the write fails, or ``"HALTED: ..."`` if the
    global kill switch fired mid-action.
    """
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="write_clipboard")
    except Exception:  # noqa: BLE001
        pass

    body = text if isinstance(text, str) else str(text or "")
    if not body.strip():
        return "ERROR: write_clipboard requires non-empty text"

    from dana.tools.clipboard_actuator import ClipboardActuator

    result = ClipboardActuator().write_text(body)
    if result.get("halted"):
        return f"HALTED: write_clipboard — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: write_clipboard failed: {result.get('error')}"

    dry_note = " (dry_run)" if result.get("dry_run") else ""
    return f"SUCCESS: Wrote {result.get('chars')} chars to clipboard{dry_note}"
