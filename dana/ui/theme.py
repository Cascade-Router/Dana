"""Shared design tokens + runtime 3-theme engine for Dana CustomTkinter surfaces."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Theme catalogs (Obsidian Mint / Cyber Amber / Ghost Light)
# ---------------------------------------------------------------------------

THEME_NAMES = (
    "Obsidian Mint",
    "Cyber Amber",
    "Ghost Light",
)

DEFAULT_THEME = "Obsidian Mint"

_THEME_DEFS: dict[str, dict[str, str]] = {
    "Obsidian Mint": {
        "name": "Obsidian Mint",
        "appearance": "dark",
        "bg": "#0a0e17",
        "card": "#131b2e",
        "border": "#1e293b",
        "ghost": "#1e293b",
        "text": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_on_accent": "#0a0e17",
        "accent": "#10b981",
        "accent_hover": "#059669",
        "emerald": "#10b981",
        "emerald_hover": "#059669",
        "rose": "#F43F5E",
        "rose_hover": "#E11D48",
        "amber": "#F59E0B",
        "bubble_user": "#10b981",
        "bubble_user_text": "#F8FAFC",
        "bubble_dana": "#131b2e",
        "bubble_dana_border": "#1e293b",
        "bubble_system": "#1e293b",
    },
    "Cyber Amber": {
        "name": "Cyber Amber",
        "appearance": "dark",
        "bg": "#070b14",
        "card": "#0f172a",
        "border": "#1e293b",
        "ghost": "#1e293b",
        "text": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_on_accent": "#070b14",
        "accent": "#f59e0b",
        "accent_hover": "#d97706",
        "emerald": "#f59e0b",
        "emerald_hover": "#d97706",
        "rose": "#F43F5E",
        "rose_hover": "#E11D48",
        "amber": "#f59e0b",
        "bubble_user": "#f59e0b",
        "bubble_user_text": "#070b14",
        "bubble_dana": "#0f172a",
        "bubble_dana_border": "#1e293b",
        "bubble_system": "#1e293b",
    },
    "Ghost Light": {
        "name": "Ghost Light",
        "appearance": "light",
        "bg": "#f8fafc",
        "card": "#ffffff",
        "border": "#e2e8f0",
        "ghost": "#e2e8f0",
        "text": "#0f172a",
        "text_secondary": "#64748b",
        "text_on_accent": "#ffffff",
        "accent": "#4f46e5",
        "accent_hover": "#4338ca",
        "emerald": "#4f46e5",
        "emerald_hover": "#4338ca",
        "rose": "#E11D48",
        "rose_hover": "#BE123C",
        "amber": "#D97706",
        "bubble_user": "#4f46e5",
        "bubble_user_text": "#ffffff",
        "bubble_dana": "#ffffff",
        "bubble_dana_border": "#e2e8f0",
        "bubble_system": "#f1f5f9",
    },
}

# Active module-level tokens (default = Obsidian Mint). Mutated by ``set_theme``.
BG = "#0a0e17"
CARD = "#131b2e"
BORDER = "#1e293b"
GHOST = "#1e293b"
TEXT = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_ON_ACCENT = "#0a0e17"
MUTED = TEXT_SECONDARY
ACCENT = "#10b981"
ACCENT_HOVER = "#059669"
EMERALD = "#10b981"
EMERALD_HOVER = "#059669"
ROSE = "#F43F5E"
ROSE_HOVER = "#E11D48"
AMBER = "#F59E0B"

STATUS_IDLE = MUTED
STATUS_UPDATE_READY = EMERALD
STATUS_UPDATE_AVAILABLE = ACCENT
STATUS_STAGING = AMBER
STATUS_HEALTHY = EMERALD
STATUS_FAILED = ROSE
STATUS_SLOT_ACTIVE = EMERALD

BUBBLE_USER = "#10b981"
BUBBLE_USER_TEXT = "#F8FAFC"
BUBBLE_DANA = "#131b2e"
BUBBLE_DANA_BORDER = "#1e293b"
BUBBLE_SYSTEM = "#1e293b"

CANVAS = BG
CARD_BORDER = BORDER

_ACTIVE_THEME = DEFAULT_THEME
_THEME_APPLIED = False
_THEME_REL = "dana/ui/dana_theme.json"
_THEME_LISTENERS: list[Callable[[str], None]] = []


def theme_defs() -> dict[str, dict[str, str]]:
    """Return a shallow copy of the built-in theme catalog."""
    return {k: dict(v) for k, v in _THEME_DEFS.items()}


def active_theme_name() -> str:
    return _ACTIVE_THEME


def get_theme(name: str | None = None) -> dict[str, str]:
    key = (name or _ACTIVE_THEME).strip()
    if key not in _THEME_DEFS:
        key = DEFAULT_THEME
    return dict(_THEME_DEFS[key])


def _apply_tokens(tokens: dict[str, str]) -> None:
    """Push a token dict onto module-level aliases."""
    global BG, CARD, BORDER, GHOST, TEXT, TEXT_SECONDARY, TEXT_ON_ACCENT
    global MUTED, ACCENT, ACCENT_HOVER, EMERALD, EMERALD_HOVER
    global ROSE, ROSE_HOVER, AMBER
    global STATUS_IDLE, STATUS_UPDATE_READY, STATUS_UPDATE_AVAILABLE
    global STATUS_STAGING, STATUS_HEALTHY, STATUS_FAILED, STATUS_SLOT_ACTIVE
    global BUBBLE_USER, BUBBLE_USER_TEXT, BUBBLE_DANA, BUBBLE_DANA_BORDER, BUBBLE_SYSTEM
    global CANVAS, CARD_BORDER

    BG = tokens["bg"]
    CARD = tokens["card"]
    BORDER = tokens["border"]
    GHOST = tokens["ghost"]
    TEXT = tokens["text"]
    TEXT_SECONDARY = tokens["text_secondary"]
    TEXT_ON_ACCENT = tokens["text_on_accent"]
    MUTED = TEXT_SECONDARY
    ACCENT = tokens["accent"]
    ACCENT_HOVER = tokens["accent_hover"]
    EMERALD = tokens["emerald"]
    EMERALD_HOVER = tokens["emerald_hover"]
    ROSE = tokens["rose"]
    ROSE_HOVER = tokens["rose_hover"]
    AMBER = tokens["amber"]
    STATUS_IDLE = MUTED
    STATUS_UPDATE_READY = EMERALD
    STATUS_UPDATE_AVAILABLE = ACCENT
    STATUS_STAGING = AMBER
    STATUS_HEALTHY = EMERALD
    STATUS_FAILED = ROSE
    STATUS_SLOT_ACTIVE = EMERALD
    BUBBLE_USER = tokens["bubble_user"]
    BUBBLE_USER_TEXT = tokens["bubble_user_text"]
    BUBBLE_DANA = tokens["bubble_dana"]
    BUBBLE_DANA_BORDER = tokens["bubble_dana_border"]
    BUBBLE_SYSTEM = tokens["bubble_system"]
    CANVAS = BG
    CARD_BORDER = BORDER


def dana_theme_path(name: str | None = None) -> Path:
    """Absolute path to a CTk theme JSON (dev + MEIPASS).

    Default / Obsidian Mint → ``dana_theme.json``.
    Other themes → ``dana_theme_<slug>.json`` beside it.
    """
    key = (name or _ACTIVE_THEME).strip()
    if key in ("", DEFAULT_THEME, "Obsidian Mint"):
        rel = _THEME_REL
        fname = "dana_theme.json"
    else:
        slug = key.lower().replace(" ", "_")
        fname = f"dana_theme_{slug}.json"
        rel = f"dana/ui/{fname}"
    try:
        from dana.resources import get_resource_path

        return Path(os.path.abspath(str(get_resource_path(rel))))
    except Exception:  # noqa: BLE001
        return Path(os.path.abspath(str(Path(__file__).resolve().parent / fname)))


def _build_ctk_theme_dict(tokens: dict[str, str]) -> dict[str, Any]:
    """Generate a CustomTkinter theme JSON payload from token colors."""
    bg = tokens["bg"]
    card = tokens["card"]
    border = tokens["border"]
    ghost = tokens["ghost"]
    text = tokens["text"]
    muted = tokens["text_secondary"]
    accent = tokens["accent"]
    accent_h = tokens["accent_hover"]
    on_acc = tokens.get("text_on_accent") or text
    return {
        "CTk": {"fg_color": [bg, bg]},
        "CTkToplevel": {"fg_color": [bg, bg]},
        "CTkFrame": {
            "corner_radius": 6,
            "border_width": 0,
            "fg_color": [card, card],
            "top_fg_color": [ghost, ghost],
            "border_color": [border, border],
        },
        "CTkButton": {
            "corner_radius": 6,
            "border_width": 0,
            "fg_color": [accent, accent],
            "hover_color": [accent_h, accent_h],
            "border_color": [border, border],
            "text_color": [on_acc if tokens.get("appearance") == "light" else text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkLabel": {
            "corner_radius": 0,
            "border_width": 0,
            "fg_color": "transparent",
            "border_color": [border, border],
            "text_color": [text, text],
        },
        "CTkEntry": {
            "corner_radius": 6,
            "border_width": 1,
            "fg_color": [ghost, ghost],
            "border_color": [border, border],
            "text_color": [text, text],
            "placeholder_text_color": [muted, muted],
        },
        "CTkCheckBox": {
            "corner_radius": 6,
            "border_width": 3,
            "fg_color": [accent, accent],
            "border_color": [border, border],
            "hover_color": [accent_h, accent_h],
            "checkmark_color": [on_acc, on_acc],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkSwitch": {
            "corner_radius": 1000,
            "border_width": 3,
            "button_length": 0,
            "fg_color": [ghost, ghost],
            "progress_color": [accent, accent],
            "button_color": [text, text],
            "button_hover_color": [muted, muted],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkRadioButton": {
            "corner_radius": 1000,
            "border_width_checked": 6,
            "border_width_unchecked": 3,
            "fg_color": [accent, accent],
            "border_color": [border, border],
            "hover_color": [accent_h, accent_h],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkProgressBar": {
            "corner_radius": 1000,
            "border_width": 0,
            "fg_color": [ghost, ghost],
            "progress_color": [accent, accent],
            "border_color": [border, border],
        },
        "CTkSlider": {
            "corner_radius": 1000,
            "button_corner_radius": 1000,
            "border_width": 6,
            "button_length": 0,
            "fg_color": [ghost, ghost],
            "progress_color": [accent, accent],
            "button_color": [text, text],
            "button_hover_color": [muted, muted],
        },
        "CTkOptionMenu": {
            "corner_radius": 6,
            "fg_color": [ghost, ghost],
            "button_color": [accent, accent],
            "button_hover_color": [accent_h, accent_h],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkComboBox": {
            "corner_radius": 6,
            "border_width": 2,
            "fg_color": [ghost, ghost],
            "border_color": [border, border],
            "button_color": [accent, accent],
            "button_hover_color": [accent_h, accent_h],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkScrollbar": {
            "corner_radius": 1000,
            "border_spacing": 4,
            "fg_color": "transparent",
            "button_color": [muted, muted],
            "button_hover_color": [text, text],
        },
        "CTkSegmentedButton": {
            "corner_radius": 6,
            "border_width": 2,
            "fg_color": [card, card],
            "selected_color": [accent, accent],
            "selected_hover_color": [accent_h, accent_h],
            "unselected_color": [ghost, ghost],
            "unselected_hover_color": [border, border],
            "text_color": [text, text],
            "text_color_disabled": [muted, muted],
        },
        "CTkTextbox": {
            "corner_radius": 6,
            "border_width": 0,
            "fg_color": [bg, bg],
            "border_color": [border, border],
            "text_color": [text, text],
            "scrollbar_button_color": [muted, muted],
            "scrollbar_button_hover_color": [text, text],
        },
        "CTkScrollableFrame": {"label_fg_color": [card, card]},
        "DropdownMenu": {
            "fg_color": [card, card],
            "hover_color": [ghost, ghost],
            "text_color": [text, text],
        },
        "CTkFont": {
            "macOS": {"family": "SF Display", "size": 13, "weight": "normal"},
            "Windows": {"family": "Segoe UI", "size": 13, "weight": "normal"},
            "Linux": {"family": "DejaVu Sans", "size": 13, "weight": "normal"},
        },
    }


def ensure_theme_json_files() -> None:
    """Write / refresh theme JSON files next to ``dana_theme.json`` (best-effort)."""
    for name, tokens in _THEME_DEFS.items():
        path = dana_theme_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _build_ctk_theme_dict(tokens)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue


def apply_dana_ctk_theme(name: str | None = None) -> bool:
    """Load Dana CTk theme before any widgets/root. Idempotent for first call."""
    global _THEME_APPLIED, _ACTIVE_THEME
    try:
        import customtkinter as ctk
    except Exception:  # noqa: BLE001
        return False
    key = (name or _ACTIVE_THEME).strip()
    if key not in _THEME_DEFS:
        key = DEFAULT_THEME
    tokens = _THEME_DEFS[key]
    _apply_tokens(tokens)
    _ACTIVE_THEME = key
    path = dana_theme_path(key)
    try:
        if not path.is_file():
            ensure_theme_json_files()
        ctk.set_appearance_mode(tokens.get("appearance", "dark"))
        if path.is_file():
            ctk.set_default_color_theme(str(path))
        else:
            ctk.set_default_color_theme("green")
        _THEME_APPLIED = True
        return True
    except Exception:  # noqa: BLE001
        return False


def register_theme_listener(callback: Callable[[str], None]) -> None:
    if callback not in _THEME_LISTENERS:
        _THEME_LISTENERS.append(callback)


def unregister_theme_listener(callback: Callable[[str], None]) -> None:
    try:
        _THEME_LISTENERS.remove(callback)
    except ValueError:
        pass


def set_theme(name: str, *, root: Any | None = None) -> str:
    """Switch active theme at runtime; optionally recolor ``root`` widget tree."""
    global _ACTIVE_THEME, _THEME_APPLIED
    key = (name or "").strip()
    if key not in _THEME_DEFS:
        key = DEFAULT_THEME
    tokens = _THEME_DEFS[key]
    _apply_tokens(tokens)
    _ACTIVE_THEME = key
    try:
        import customtkinter as ctk

        ctk.set_appearance_mode(tokens.get("appearance", "dark"))
        path = dana_theme_path(key)
        if not path.is_file():
            ensure_theme_json_files()
        # CTk only fully applies default theme pre-widget; still set for new widgets.
        if path.is_file():
            try:
                ctk.set_default_color_theme(str(path))
            except Exception:  # noqa: BLE001
                pass
        _THEME_APPLIED = True
    except Exception:  # noqa: BLE001
        pass
    if root is not None:
        recolor_widget_tree(root, tokens)
    for cb in list(_THEME_LISTENERS):
        try:
            cb(key)
        except Exception:  # noqa: BLE001
            pass
    return key


def recolor_widget_tree(widget: Any, tokens: Optional[dict[str, str]] = None) -> None:
    """Best-effort walk: update fg/text/border colors on known CTk widgets."""
    tok = tokens or get_theme()
    bg = tok["bg"]
    card = tok["card"]
    border = tok["border"]
    ghost = tok["ghost"]
    text = tok["text"]
    muted = tok["text_secondary"]
    accent = tok["accent"]
    accent_h = tok["accent_hover"]
    rose = tok["rose"]
    rose_h = tok["rose_hover"]
    amber = tok["amber"]

    def _safe_cfg(w: Any, **kwargs: Any) -> None:
        try:
            w.configure(**kwargs)
        except Exception:  # noqa: BLE001
            pass

    cls = type(widget).__name__
    try:
        if cls in {"CTk", "CTkToplevel"}:
            _safe_cfg(widget, fg_color=bg)
        elif cls in {"CTkFrame", "CTkScrollableFrame"}:
            cur = str(widget.cget("fg_color") or "").lower()
            if cur in {"transparent", ""}:
                pass
            elif cur in {bg.lower(), card.lower(), ghost.lower(), "#0a0e17", "#070b14",
                         "#f8fafc", "#131b2e", "#0f172a", "#ffffff", "#1e293b"}:
                # Recolor known surface roles; leave custom accent frames alone.
                if "card" in cur or cur in {"#131b2e", "#0f172a", "#ffffff"}:
                    _safe_cfg(widget, fg_color=card, border_color=border)
                elif cur in {ghost.lower(), "#1e293b", "#e2e8f0"}:
                    _safe_cfg(widget, fg_color=ghost, border_color=border)
                else:
                    _safe_cfg(widget, fg_color=bg)
        elif cls == "CTkLabel":
            tc = str(widget.cget("text_color") or "").lower()
            if tc in {muted.lower(), "#94a3b8", "#64748b", "#888888"}:
                _safe_cfg(widget, text_color=muted)
            elif tc not in {rose.lower(), "#f43f5e", "#e11d48", amber.lower(),
                            "#f59e0b", "#c084fc"}:
                _safe_cfg(widget, text_color=text)
        elif cls == "CTkButton":
            fg = str(widget.cget("fg_color") or "").lower()
            if rose.lower() in fg or fg in {"#f43f5e", "#e11d48"}:
                _safe_cfg(widget, fg_color=rose, hover_color=rose_h, text_color="#FFFFFF")
            elif amber.lower() in fg or fg in {"#f59e0b", "#d97706"}:
                _safe_cfg(widget, fg_color=amber, hover_color=accent_h, text_color="#FFFFFF")
            elif fg in {ghost.lower(), "#1e293b", "#475569", "#e2e8f0"}:
                _safe_cfg(
                    widget,
                    fg_color=ghost,
                    hover_color=border,
                    text_color=text,
                    border_color=border,
                )
            else:
                _safe_cfg(widget, fg_color=accent, hover_color=accent_h)
        elif cls == "CTkEntry":
            _safe_cfg(
                widget,
                fg_color=ghost,
                border_color=border,
                text_color=text,
                placeholder_text_color=muted,
            )
        elif cls == "CTkTextbox":
            _safe_cfg(widget, fg_color=bg, text_color=text, border_color=border)
        elif cls == "CTkOptionMenu":
            _safe_cfg(
                widget,
                fg_color=ghost,
                button_color=accent,
                button_hover_color=accent_h,
                text_color=text,
            )
        elif cls == "CTkCheckBox":
            _safe_cfg(
                widget,
                fg_color=accent,
                hover_color=accent_h,
                border_color=border,
                text_color=muted,
            )
        elif cls == "CTkSlider":
            _safe_cfg(
                widget,
                fg_color=ghost,
                progress_color=accent,
                button_color=text,
            )
        elif cls == "CTkTabview":
            _safe_cfg(widget, fg_color=bg, text_color=text)
    except Exception:  # noqa: BLE001
        pass

    try:
        children = widget.winfo_children()
    except Exception:  # noqa: BLE001
        children = []
    for child in children:
        recolor_widget_tree(child, tok)


# Initialize tokens from default theme definition.
_apply_tokens(_THEME_DEFS[DEFAULT_THEME])

__all__ = (
    "THEME_NAMES",
    "DEFAULT_THEME",
    "BG",
    "CARD",
    "BORDER",
    "GHOST",
    "TEXT",
    "TEXT_SECONDARY",
    "TEXT_ON_ACCENT",
    "MUTED",
    "ACCENT",
    "ACCENT_HOVER",
    "EMERALD",
    "EMERALD_HOVER",
    "ROSE",
    "ROSE_HOVER",
    "AMBER",
    "STATUS_IDLE",
    "STATUS_UPDATE_READY",
    "STATUS_UPDATE_AVAILABLE",
    "STATUS_STAGING",
    "STATUS_HEALTHY",
    "STATUS_FAILED",
    "STATUS_SLOT_ACTIVE",
    "BUBBLE_USER",
    "BUBBLE_USER_TEXT",
    "BUBBLE_DANA",
    "BUBBLE_DANA_BORDER",
    "BUBBLE_SYSTEM",
    "CANVAS",
    "CARD_BORDER",
    "theme_defs",
    "active_theme_name",
    "get_theme",
    "dana_theme_path",
    "ensure_theme_json_files",
    "apply_dana_ctk_theme",
    "set_theme",
    "recolor_widget_tree",
    "register_theme_listener",
    "unregister_theme_listener",
)
