"""Plugins & Env Keys panel for DanaGUI's Memory & Settings tab.

Mounted by ``dana.ui.app_gui.DanaGUI._build_ui`` via ``build_plugin_manager_card``.
Kept as a standalone module (rather than inline in app_gui.py) so this
feature's addition to that already-large file stays a two-line
import+call. All ``dana.ui.app_gui`` / ``customtkinter`` imports are lazy
(inside the functions, not at module scope) to avoid a circular import with
app_gui.py, which imports this module.
"""

from __future__ import annotations

from typing import Any

_PINNABLE_TOOLS: tuple[tuple[str, str], ...] = (
    ("delegate_to_cursor", "Cursor Handoff"),
    ("dispatch_research_swarm", "Research Swarm"),
    ("dispatch_titan_repair", "Titan Repair Swarm"),
    ("capture_and_analyze_screen", "OS Screen Capture + VLM"),
    ("execute_os_keystrokes", "OS Keystroke Actuator"),
)


def build_plugin_manager_card(gui: Any, parent: Any) -> None:
    """Adds the "Plugins & Env Keys" card: per-feature toggles + Add Tool dialog."""
    import customtkinter as ctk

    from dana.core import shared_state
    from dana.features import feature_manager
    from dana.ui.app_gui import _UI_CARD_BORDER, _UI_EMERALD, _UI_MUTED, _UI_TEXT

    card = gui._make_card(parent, title="Plugins & Env Keys")
    gui._feature_rows = {}

    def _dot_color(feature) -> str:
        if not feature.implemented:
            return _UI_MUTED
        return _UI_EMERALD if feature_manager.is_feature_enabled(feature.id) else _UI_MUTED

    def _row_caption(feature) -> str:
        if not feature.implemented:
            return "Not implemented in this build (stub)."
        return "Enabled" if feature_manager.is_feature_enabled(feature.id) else "Disabled"

    def _refresh_row(feature_id: str) -> None:
        row = gui._feature_rows.get(feature_id)
        if row is None:
            return
        feature = feature_manager.FEATURES[feature_id]
        try:
            row["dot"].configure(fg_color=_dot_color(feature))
            row["var"].set(feature_manager.is_feature_enabled(feature_id))
            row["caption"].configure(text=_row_caption(feature))
        except Exception:  # noqa: BLE001
            pass

    def _make_toggle_handler(feature_id: str, var):
        def _on_toggle() -> None:
            feature_manager.set_feature_enabled(feature_id, bool(var.get()))
            _refresh_row(feature_id)

        return _on_toggle

    for feature in feature_manager.list_features():
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", pady=(0, 6))

        dot = ctk.CTkFrame(
            row, width=10, height=10, corner_radius=5, fg_color=_dot_color(feature)
        )
        dot.pack(side="left", padx=(2, 8), pady=4)

        var = ctk.BooleanVar(value=feature_manager.is_feature_enabled(feature.id))
        chk = ctk.CTkCheckBox(
            row,
            text=feature.label,
            variable=var,
            command=_make_toggle_handler(feature.id, var),
            text_color=_UI_TEXT,
        )
        if not feature.implemented:
            chk.configure(state="disabled")
        chk.pack(side="left")

        caption = ctk.CTkLabel(
            row, text=_row_caption(feature), text_color=_UI_MUTED, anchor="w"
        )
        caption.pack(side="left", padx=(10, 0))

        gui._feature_rows[feature.id] = {"dot": dot, "var": var, "caption": caption}

    def _on_flags_changed(_flags: dict) -> None:
        gui.after(0, lambda: [_refresh_row(fid) for fid in list(gui._feature_rows)])

    shared_state.register_feature_flags_listener(_on_flags_changed)

    ctk.CTkButton(
        card,
        text="+ Add Tool",
        command=lambda: _open_add_tool_dialog(gui),
        fg_color=_UI_CARD_BORDER,
        hover_color=_UI_EMERALD,
        text_color=_UI_TEXT,
    ).pack(anchor="w", pady=(8, 0))


def _open_add_tool_dialog(gui: Any) -> None:
    """Small popup letting the user pin already-registered tools into this session."""
    import customtkinter as ctk

    from dana.features import feature_manager
    from dana.ui.app_gui import _UI_CARD, _UI_MUTED, _UI_TEXT

    dialog = ctk.CTkToplevel(gui)
    dialog.title("Add Existing Tool")
    dialog.geometry("380x300")
    dialog.configure(fg_color=_UI_CARD)
    dialog.transient(gui)
    dialog.grab_set()

    ctk.CTkLabel(
        dialog,
        text=(
            "Pin an already-registered repo tool into this session's active "
            "toolset (forces it into the LLM's bound tools, bypassing semantic "
            "top-K retrieval)."
        ),
        text_color=_UI_MUTED,
        wraplength=340,
        justify="left",
    ).pack(fill="x", padx=14, pady=(14, 10))

    pinned = feature_manager.get_pinned_tool_ids()

    for tool_id, label in _PINNABLE_TOOLS:
        var = ctk.BooleanVar(value=tool_id in pinned)

        def _on_toggle(tool_id=tool_id, var=var) -> None:
            if var.get():
                feature_manager.pin_tool(tool_id)
            else:
                feature_manager.unpin_tool(tool_id)

        ctk.CTkCheckBox(
            dialog,
            text=label,
            variable=var,
            command=_on_toggle,
            text_color=_UI_TEXT,
        ).pack(anchor="w", padx=14, pady=4)

    ctk.CTkButton(dialog, text="Done", command=dialog.destroy).pack(pady=(14, 14))
