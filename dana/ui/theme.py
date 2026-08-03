"""Shared dark-slate design tokens for Dana CustomTkinter surfaces."""

from __future__ import annotations

import os
from pathlib import Path

# Canvas / surfaces
BG = "#0a0e17"
CARD = "#131b2e"
BORDER = "#1e293b"
GHOST = "#1e293b"

# Text
TEXT = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_ON_ACCENT = "#0a0e17"
MUTED = TEXT_SECONDARY

# Accents
ACCENT = "#10b981"  # hyper mint — primary
ACCENT_HOVER = "#059669"
EMERALD = "#10b981"  # active / success (mint family)
EMERALD_HOVER = "#059669"
ROSE = "#F43F5E"  # stop / alert
ROSE_HOVER = "#E11D48"
AMBER = "#F59E0B"

_THEME_APPLIED = False
_THEME_REL = "dana/ui/dana_theme.json"


def dana_theme_path() -> Path:
    """Absolute path to ``dana_theme.json`` (dev + MEIPASS)."""
    try:
        from dana.resources import get_resource_path

        return Path(os.path.abspath(str(get_resource_path(_THEME_REL))))
    except Exception:  # noqa: BLE001
        return Path(os.path.abspath(str(Path(__file__).resolve().parent / "dana_theme.json")))


def apply_dana_ctk_theme() -> bool:
    """Load Dana CTk theme before any widgets/root. Idempotent."""
    global _THEME_APPLIED
    if _THEME_APPLIED:
        return True
    try:
        import customtkinter as ctk
    except Exception:  # noqa: BLE001
        return False
    path = dana_theme_path()
    try:
        ctk.set_appearance_mode("dark")
        if path.is_file():
            ctk.set_default_color_theme(str(path))
        else:
            # Source/MEIPASS miss — keep mint-ish green built-in rather than blue.
            ctk.set_default_color_theme("green")
        _THEME_APPLIED = True
        return True
    except Exception:  # noqa: BLE001
        return False

# OTA / blue-green status pills
STATUS_IDLE = MUTED
STATUS_UPDATE_READY = EMERALD
STATUS_UPDATE_AVAILABLE = ACCENT
STATUS_STAGING = AMBER
STATUS_HEALTHY = EMERALD
STATUS_FAILED = ROSE
STATUS_SLOT_ACTIVE = EMERALD

# Chat bubbles
BUBBLE_USER = "#10b981"
BUBBLE_USER_TEXT = "#F8FAFC"
BUBBLE_DANA = "#131b2e"
BUBBLE_DANA_BORDER = "#1e293b"
BUBBLE_SYSTEM = "#1e293b"

# Compat aliases used by core_agent / views
CANVAS = BG
CARD_BORDER = BORDER

__all__ = (
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
    "dana_theme_path",
    "apply_dana_ctk_theme",
)
