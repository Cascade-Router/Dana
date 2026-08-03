"""Shared dark-slate design tokens for Dana CustomTkinter surfaces."""

from __future__ import annotations

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
)
