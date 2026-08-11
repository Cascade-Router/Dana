"""DanaGUI — the CustomTkinter Control Dashboard desktop app.

Extracted verbatim from ``dana.core_agent`` (Phase 5 of the core_agent.py
decomposition; see the approved refactor plan). This module owns the
``TraceCell``/``DanaGUI`` widget tree; it never calls audio/vision workers,
``agent_loop``, or vault-unlock logic directly — it only talks to the rest
of the app through ``dana.core.shared_state`` listener registration, the
``gui_telemetry_queue`` + ``emit_trace()`` bridge, and ``dana.ui.status_bus``
/ ``dana.graph.monitor_bus`` polling, all funneled onto the Tk main thread
via ``self.after(...)``.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import customtkinter as ctk
import tkinter as tk

import dana.core.shared_state as state
from dana.agentic import get_dana_mode
from dana.audio.mic_input import (
    _device_rate,
    ensure_mic_ingest_thread,
    request_mic_ingest_restart,
    save_audio_settings,
)
from dana.core.agent_loop import execute_tool_call
from dana.core.constants import WAKEWORD_MODELS
from dana.ingestion.text_injection import set_injected_question
from dana.core.shared_state import (
    _TRACE_STATUS_ICONS,
    gui_telemetry_queue,
    get_ui_state,
    is_recording,
    set_subtitle,
    stop_event,
    vad_capture_active,
    register_dictation_sessions_listener,
    register_spec_approval_listener,
    register_transcript_listener,
    register_vault_prompt_listener,
    supply_vault_unlock_response,
    emit_live_transcript,
)
from dana.core.telemetry import AsyncRingBuffer, NeuralStreamEmitter
from dana.audio.tts_worker import reset_tts_audio_state
from dana.logging import log, log_conversation
from dana.tools import ToolCall
from dana.utils.adaptive_poller import AdaptivePoller

# Shared dark-slate tokens (dana.ui.theme) — keep local aliases for call sites.
try:
    from dana.ui import theme as _UI_THEME

    _UI_CANVAS = _UI_THEME.BG
    _UI_CARD = _UI_THEME.CARD
    _UI_CARD_BORDER = _UI_THEME.BORDER
    _UI_GHOST = _UI_THEME.GHOST
    _UI_MUTED = _UI_THEME.MUTED
    _UI_TEXT = _UI_THEME.TEXT
    _UI_ACCENT = _UI_THEME.ACCENT
    _UI_ACCENT_HOVER = _UI_THEME.ACCENT_HOVER
    _UI_EMERALD = _UI_THEME.EMERALD
    _UI_EMERALD_HOVER = _UI_THEME.EMERALD_HOVER
    _UI_ROSE = _UI_THEME.ROSE
    _UI_ROSE_HOVER = _UI_THEME.ROSE_HOVER
    _UI_AMBER = _UI_THEME.AMBER
except Exception:  # noqa: BLE001
    _UI_CANVAS = "#0a0e17"
    _UI_CARD = "#131b2e"
    _UI_CARD_BORDER = "#1e293b"
    _UI_GHOST = "#1e293b"
    _UI_MUTED = "#94A3B8"
    _UI_TEXT = "#F8FAFC"
    _UI_ACCENT = "#10b981"
    _UI_ACCENT_HOVER = "#059669"
    _UI_EMERALD = "#10B981"
    _UI_EMERALD_HOVER = "#059669"
    _UI_ROSE = "#F43F5E"
    _UI_ROSE_HOVER = "#E11D48"
    _UI_AMBER = "#F59E0B"

_UI_STATE_LABELS = {
    "idle": "Idle",
    "listening": "Listening",
    "speaking": "Speaking",
    "followup": "Listening (follow-up)",
    "transcribing": "Processing",
    "thinking": "Processing",
}

# Master telemetry dispatcher cadences (seconds) — see _master_telemetry_tick.
# One AdaptivePoller-driven heartbeat replaces five independent self.after()
# loops; each consumer below still only fires at its own interval, tracked
# via monotonic elapsed time so a backed-off (idle) heartbeat never drifts.
_TELEMETRY_CADENCES_S: dict[str, float] = {
    "live_trace": 0.08,
    "process_telemetry": 0.10,
    "state_changes": 0.10,
    "dag_monitor": 0.25,
    "task_tracker": 0.40,
}

_TRACE_MODE_COLORS: dict[str, str] = {
    "chat": _UI_EMERALD,
    "developer": _UI_AMBER,
    "vision": _UI_ACCENT,
    "research": _UI_AMBER,
    "dictation": "#A855F7",
}
_TRACE_IDLE_COLOR = _UI_MUTED


def _warm_heavy_runtime_assets() -> None:
    """Stage 8.9.7 — background warm after ENGAGE (GUI already interactive).

    Imports LangGraph wiring only. Florence-2 / YOLO stay JIT on first vision
    call so Standby boot never blocks on VRAM-heavy models.
    """
    try:
        import dana.agentic_react_graph  # noqa: F401

        log("Main", "Heavy warm: agentic_react_graph imported (Florence remains JIT)")
    except Exception as exc:  # noqa: BLE001
        log("Main", f"Heavy warm skipped: {exc}")


class TraceCell(ctk.CTkFrame):
    """One pipeline stage row in the Live Trace scroll area."""

    def __init__(self, master: Any, stage: str, message: str, status: str = "active") -> None:
        super().__init__(
            master,
            corner_radius=8,
            border_width=2,
            border_color=_TRACE_IDLE_COLOR,
            fg_color=("gray92", "gray17"),
        )
        self.stage = stage
        self.current_status = "active"
        self.icon_label = ctk.CTkLabel(
            self,
            text=_TRACE_STATUS_ICONS["active"],
            width=28,
            font=ctk.CTkFont(size=16),
        )
        self.icon_label.pack(side="left", padx=(10, 6), pady=8)
        self.msg_label = ctk.CTkLabel(
            self,
            text=message or stage,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=13),
        )
        self.msg_label.pack(side="left", fill="x", expand=True, padx=(0, 12), pady=8)
        self.update_status(status, message=message)

    def update_status(
        self,
        status: str,
        message: str | None = None,
        *,
        accent: str | None = None,
    ) -> None:
        normalized = (status or "active").strip().lower()
        if normalized not in _TRACE_STATUS_ICONS:
            normalized = "active"
        self.current_status = normalized
        self.icon_label.configure(text=_TRACE_STATUS_ICONS[normalized])
        if message is not None:
            self.msg_label.configure(text=message or self.stage)
        color = accent or _TRACE_IDLE_COLOR
        if normalized == "active":
            border = color if color != _TRACE_IDLE_COLOR else _UI_ACCENT
            text_color = border
        elif normalized == "completed":
            border = color if color != _TRACE_IDLE_COLOR else "#10B981"
            text_color = ("gray20", "gray85")
        else:  # bypassed
            border = _TRACE_IDLE_COLOR
            text_color = _TRACE_IDLE_COLOR
        try:
            self.configure(border_color=border)
            self.msg_label.configure(text_color=text_color)
        except Exception:  # noqa: BLE001
            pass


class DanaGUI(ctk.CTk):
    """Live Trace window with settings tabs; retreats to the tray on close."""

    def __init__(self) -> None:
        # Theme must load before CTk root/widgets (not after super()).
        try:
            from dana.ui.theme import apply_dana_ctk_theme

            apply_dana_ctk_theme()
        except Exception:  # noqa: BLE001
            pass
        super().__init__()
        state.set_gui_instance(self)
        register_transcript_listener(self._on_transcript_event)
        register_vault_prompt_listener(self._on_vault_unlock_request)
        register_spec_approval_listener(self._on_spec_approval_requested)
        register_dictation_sessions_listener(self._on_dictation_sessions_changed)
        self.title("Dana — Control Dashboard")
        self.geometry("1440x900")
        self.minsize(1280, 800)
        self._dictation_active = False
        self._behavior_locked = False
        # Stage 8.9.7 — soft LangGraph engine ignition (False = STANDBY).
        self.engine_active = False
        self._engine_stopped = False
        self._behavior_sliders: dict[str, ctk.CTkSlider] = {}
        self._behavior_labels: dict[str, ctk.CTkLabel] = {}
        self._behavior_last_write: dict[str, float] = {}
        # Stage 8.5.1 — static settings that must not change while system is hot.
        self._static_behavior_widgets: list[Any] = []
        self._behavior_lock_hint: ctk.CTkLabel | None = None
        self._behavior_lock_overlay: ctk.CTkFrame | None = None
        self._behavior_mixer_host: ctk.CTkFrame | None = None
        self._behavior_reload_btn: ctk.CTkButton | None = None
        self.dictation_status: ctk.CTkLabel | None = None
        self._engine_status_lbl: ctk.CTkLabel | None = None
        self._engine_warn_lbl: ctk.CTkLabel | None = None
        self._vault_warn_lbl: ctk.CTkLabel | None = None
        self._vault_unlock_win: ctk.CTkToplevel | None = None
        self._engage_btn: ctk.CTkButton | None = None
        self._standby_btn: ctk.CTkButton | None = None  # legacy; merged into toggle
        self._engage_toggle_btn: ctk.CTkButton | None = None
        self._tasks_toggle_btn: ctk.CTkButton | None = None
        self._dag_toggle_btn: ctk.CTkButton | None = None
        self._diag_header_btn: ctk.CTkButton | None = None
        self._header_status_lbl: ctk.CTkLabel | None = None
        self._header_seg: Any | None = None
        self._assistant_main: Any | None = None
        self._assistant_side: Any | None = None
        self.task_tracker_frame: Any | None = None
        self._tasks_drawer_visible = False
        self.dag_monitor_frame: Any | None = None
        self.dag_monitor_view: Any | None = None
        self._dag_drawer_visible = False
        self._dag_status_lbl: Any | None = None
        self._dag_stream_thread: Any | None = None
        self._engine_warn_job: str | None = None
        self._diag_overlay: Any | None = None
        self._spec_approval_host: Any | None = None
        self._spec_approval_card: Any | None = None
        self._spec_approval_visible = False
        self._pending_spec_payload: dict[str, Any] | None = None
        # Stage 9.3 — Settings auto-updater chrome.
        self._update_status_lbl: ctk.CTkLabel | None = None
        self._update_check_btn: ctk.CTkButton | None = None
        self._update_apply_btn: ctk.CTkButton | None = None
        self._update_busy = False
        # Phase 1 OTA — Auto-Update Mode + Hot Apply pill.
        self._ota_mode_var: Any = None
        self._ota_mode_menu: ctk.CTkOptionMenu | None = None
        self._ota_pill_lbl: ctk.CTkLabel | None = None
        self._ota_slot_lbl: ctk.CTkLabel | None = None
        self._ota_staging_lbl: ctk.CTkLabel | None = None
        self._ota_hot_apply_btn: ctk.CTkButton | None = None
        self._ota_manager: Any = None
        # Stage 8.10 — Dashboard silent text chat.
        self.chat_entry: ctk.CTkEntry | None = None
        self._chat_send_btn: ctk.CTkButton | None = None
        self.bottom_input_frame: Any | None = None
        self._tasks_empty_lbl: Any | None = None
        self.transcript_box = None
        self._chat_view = None
        # VAD / supervisor STATE_CHANGE indicators (above chat input).
        self._vad_mic_lbl: ctk.CTkLabel | None = None
        self._system_status_lbl: ctk.CTkLabel | None = None
        self._vad_listening = False
        self._vad_pulse_on = False

        try:
            self.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        self._mic_labels: list[str] = []
        self._speaker_labels: list[str] = []
        self._mic_by_label: dict[str, int] = {}
        self._speaker_by_label: dict[str, int] = {}
        self.mic_menu = None
        self.speaker_menu = None
        self.save_btn = None
        self.apply_note = None
        self._theme_menu = None
        self._theme_var = None
        self._trace_cells: dict[str, TraceCell] = {}
        self._pulse_on = False
        self._header_mode = "chat"
        self.assistive_orb: Any | None = None
        self._perception_feed_job: str | None = None
        self._perception_feed_img = None
        self._perception_feed_lbl = None
        self._perception_feed_busy = False

        # Unified Agent Canvas — 60/40 split (chat | workspace inspector).
        self._canvas_frame: Any | None = None
        self._workspace_inspector: Any | None = None
        self._neural_stream_text: Any | None = None
        self.artifact_viewer: Any | None = None
        self._neural_rendered = 0
        self._telemetry_buffer = AsyncRingBuffer(capacity=500)
        self._telemetry_emitter = NeuralStreamEmitter(self._telemetry_buffer)
        # Master telemetry dispatcher — see _master_telemetry_tick. Tracks the
        # last-fired monotonic timestamp per consumer so one adaptive
        # heartbeat can gate all five original polling cadences.
        self._telemetry_last: dict[str, float] = {}
        self._adaptive_poller = AdaptivePoller(
            self._telemetry_had_activity, t_min=0.05, t_max=0.5, gamma=1.5
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_to_tray)
        self.withdraw()
        # Post-init icon lifecycle: uniquely named runtime PNG/ICO + PhotoImage keepalive.
        self._icon_keepalive = None
        try:
            from dana.ui.logo import apply_window_icon, schedule_window_icon

            apply_window_icon(self)
            schedule_window_icon(self, delay_ms=100)
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.7 — floating AssistiveTouch orb (DISABLED for now).
        # try:
        #     self.after(200, self._start_assistive_orb)
        # except Exception:  # noqa: BLE001
        #     pass
        self.after(400, self._refresh_stats)
        self.after(500, self._pulse_active_cells)
        # process_telemetry / _poll_state_changes no longer self-schedule —
        # _master_telemetry_tick dispatches both (plus LiveTracePanel /
        # DagMonitorView / TaskTrackerView) on one shared heartbeat whose
        # delay adapts via self._adaptive_poller.note_activity() each tick.
        # AdaptivePoller.start() is intentionally NOT used — see its
        # docstring: a background thread cannot safely touch Tk, so this
        # chain stays a normal main-thread self.after() loop throughout.
        self.after(int(self._adaptive_poller.t_min * 1000), self._master_telemetry_tick)
        # Phase 2A — optional IPC attach (no-op / degrade when daemon down).
        try:
            self.after(250, self._init_daemon_client)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(700, self._schedule_perception_feed)
        except Exception:  # noqa: BLE001
            pass

    def _init_daemon_client(self) -> None:
        """Attach Control Dashboard to Agent Engine sidecar (graceful if absent)."""
        try:
            from dana.ui.daemon_client import DaemonClient, daemon_ipc_enabled
        except Exception:  # noqa: BLE001
            return
        if not daemon_ipc_enabled():
            return
        if getattr(self, "_daemon_client", None) is not None:
            return

        def _on_state(state: str, badge: str) -> None:
            try:
                self.after(
                    0,
                    lambda b=badge, s=state: self._set_daemon_badge(
                        b if s == "reconnecting" else ""
                    ),
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            client = DaemonClient(on_state_change=_on_state)
            self._daemon_client = client
            client.connect(retries=1)
            client.start_auto_reconnect()
        except Exception:  # noqa: BLE001
            self._daemon_client = None

    def _set_daemon_badge(self, text: str) -> None:
        lbl = getattr(self, "daemon_badge", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=text or "")
        except Exception:  # noqa: BLE001
            pass

    def _mode_accent(self, mode: str | None = None) -> str:
        key = (mode or self._header_mode or "chat").strip().lower()
        if self._dictation_active and key in {"chat", "developer", "vision", "research"}:
            # Dictation latch overrides the glowing status badge.
            return _TRACE_MODE_COLORS.get("dictation", "#9C27B0")
        return _TRACE_MODE_COLORS.get(key, _TRACE_IDLE_COLOR)

    def _set_mode_indicator(self, mode: str | None) -> None:
        key = (mode or "chat").strip().lower()
        if key not in _TRACE_MODE_COLORS:
            key = "chat"
        self._header_mode = key
        display_key = "dictation" if self._dictation_active else key
        color = _TRACE_MODE_COLORS.get(display_key, self._mode_accent(key))
        label = "Dictation" if self._dictation_active else key.title()
        try:
            # Stage 8.9.8 — header live status (CHAT badge removed).
            if hasattr(self, "mode_badge") and self.mode_badge is not None:
                self.mode_badge.configure(
                    text=f"  ●  {label.upper()}  ",
                    text_color=color,
                    fg_color=_UI_GHOST,
                )
            hdr = getattr(self, "_header_status_lbl", None)
            if hdr is not None and not bool(getattr(self, "engine_active", False)):
                # When engaged, _refresh_engine_ui owns the ACTIVE/STANDBY text.
                hdr.configure(text=f"• {label.upper()}", text_color=color)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _make_card(
        self,
        parent: Any,
        *,
        title: str,
        padx: int = 12,
        pady: tuple[int, int] = (10, 10),
        expand: bool = True,
    ) -> ctk.CTkFrame:
        """Floating dark card container with padded header."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=16,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.pack(fill="both" if expand else "x", expand=expand, padx=padx, pady=pady)
        ctk.CTkLabel(
            card,
            text=title,
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
        ).pack(fill="x", padx=14, pady=(12, 6))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return body

    def _build_ui(self) -> None:
        # Single HUD row — grid keeps brand | tabs | controls from overlapping.
        header = ctk.CTkFrame(
            self,
            fg_color=_UI_CARD,
            corner_radius=0,
            border_width=0,
            height=54,
        )
        header.pack(fill="x", padx=0, pady=0)
        try:
            header.grid_columnconfigure(0, weight=0, minsize=200)
            header.grid_columnconfigure(1, weight=1, minsize=420)
            header.grid_columnconfigure(2, weight=0, minsize=360)
            header.grid_propagate(False)
            header.configure(height=54)
        except Exception:  # noqa: BLE001
            pass
        self.mode_dot = None
        self.mode_label = None
        self._header_logo_img = None
        self._header_logo_lbl = None
        self.mode_badge = None  # removed redundant CHAT badge

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=(10, 4), pady=6)
        try:
            from dana.ui.logo import invalidate_logo_cache, load_premium_logo

            invalidate_logo_cache()
            self._header_logo_img = load_premium_logo((24, 24))
            if self._header_logo_img is not None:
                self._header_logo_lbl = ctk.CTkLabel(
                    left,
                    text="",
                    image=self._header_logo_img,
                    width=24,
                    height=24,
                )
                self._header_logo_lbl.pack(side="left", padx=(0, 6))
        except Exception:  # noqa: BLE001
            self._header_logo_img = None
            self._header_logo_lbl = None
        ctk.CTkLabel(
            left,
            text="Dānā",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).pack(side="left", padx=(0, 8))
        self._header_status_lbl = ctk.CTkLabel(
            left,
            text="• STANDBY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._header_status_lbl.pack(side="left")
        # Mic / VAD pipeline indicator (Idle | Listening | Processing).
        self._vad_mic_lbl = ctk.CTkLabel(
            left,
            text="● Idle",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._vad_mic_lbl.pack(side="left", padx=(8, 0))
        self._system_status_lbl = ctk.CTkLabel(
            left,
            text="Idle",
            font=ctk.CTkFont(size=10),
            text_color=_UI_MUTED,
            anchor="w",
        )
        self._system_status_lbl.pack(side="left", padx=(6, 0))
        self._vad_listening = False
        self._vad_processing = False
        # Phase 2A — engine sidecar reconnect badge (hidden until IPC drop).
        self.daemon_badge = ctk.CTkLabel(
            left,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#FBBF24",
            fg_color="transparent",
        )
        self.daemon_badge.pack(side="left", padx=(6, 0))
        self._daemon_client = None

        # Center tab switcher (mirrors CTkTabview; built-in segment hidden below).
        center = ctk.CTkFrame(header, fg_color="transparent")
        center.grid(row=0, column=1, sticky="ew", padx=4, pady=6)
        self._header_seg = ctk.CTkSegmentedButton(
            center,
            values=["Assistant & Tasks", "Perception", "Memory & Settings"],
            command=self._on_header_tab,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self._header_seg.pack(fill="x", expand=True, padx=2)
        try:
            self._header_seg.set("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e", padx=(4, 10), pady=6)

        self.stop_dana_btn = ctk.CTkButton(
            right,
            text="STOP DANA",
            width=96,
            height=28,
            corner_radius=8,
            fg_color=_UI_ROSE,
            hover_color=_UI_ROSE_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._on_stop_dana_clicked,
        )
        self.stop_dana_btn.pack(side="right", padx=(4, 0))

        self._diag_header_btn = ctk.CTkButton(
            right,
            text="Diagnostics",
            width=88,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10),
            command=self._dashboard_open_trace,
        )
        self._diag_header_btn.pack(side="right", padx=(4, 0))

        try:
            _engage_font = ctk.CTkFont(
                family="Segoe UI Historic", size=10, weight="bold"
            )
        except Exception:  # noqa: BLE001
            _engage_font = ctk.CTkFont(size=10, weight="bold")
        self._engage_toggle_btn = ctk.CTkButton(
            right,
            text="Engaged",
            width=88,
            height=28,
            corner_radius=8,
            font=_engage_font,
            fg_color=_UI_EMERALD,
            hover_color="#059669",
            text_color="#ECFDF5",
            command=self.toggle_engine_engage,
        )
        self._engage_toggle_btn.pack(side="right", padx=(4, 0))
        self._engage_btn = self._engage_toggle_btn
        self._standby_btn = None

        self._tasks_toggle_btn = ctk.CTkButton(
            right,
            text="Tasks ▸",
            width=64,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._toggle_tasks_drawer,
        )
        self._tasks_toggle_btn.pack(side="right", padx=(4, 0))

        self._dag_toggle_btn = ctk.CTkButton(
            right,
            text="DAG ▸",
            width=64,
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._toggle_dag_drawer,
        )
        self._dag_toggle_btn.pack(side="right", padx=(4, 0))

        try:
            from dana.ui.tooltips import attach_tooltip

            attach_tooltip(
                self._engage_toggle_btn,
                "When Engaged, Dānā listens for the wake word and responds to "
                "voice/text. Click to put in Standby.",
            )
            attach_tooltip(
                self._diag_header_btn,
                "Open system logs, audio RMS meters, and component health status.",
            )
            attach_tooltip(
                self._tasks_toggle_btn,
                "Shows real-time status of background Python sandbox scripts "
                "and research swarms.",
            )
            attach_tooltip(
                self._dag_toggle_btn,
                "Live LangGraph DAG execution tree and tool micro-log. "
                "Hotkey: Ctrl+Shift+D.",
            )
            attach_tooltip(
                self.stop_dana_btn,
                "Emergency kill-switch. Instantly halts all background processes "
                "and shuts down the engine.",
            )
        except Exception:  # noqa: BLE001
            pass

        # Three polished surfaces: Assistant, Perception, Memory & Settings.
        tabs = ctk.CTkTabview(
            self,
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=14,
            border_width=0,
        )
        tabs.pack(fill="both", expand=True, padx=14, pady=(8, 14))
        self._tabs = tabs

        tab_assistant = tabs.add("Assistant & Tasks")
        tab_perception = tabs.add("Perception")
        tab_memory = tabs.add("Memory & Settings")

        def _hide_builtin_tab_strip() -> None:
            """CTkTabview re-grids ``_segmented_button`` on every ``add`` — hide it."""
            try:
                seg = tabs._segmented_button  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return
            for forget in (
                getattr(seg, "grid_forget", None),
                getattr(seg, "pack_forget", None),
                getattr(seg, "place_forget", None),
            ):
                if forget is None:
                    continue
                try:
                    forget()
                except Exception:  # noqa: BLE001
                    pass
            try:
                seg.configure(height=0, width=0)
            except Exception:  # noqa: BLE001
                pass

        _hide_builtin_tab_strip()
        try:
            # Stop future layout passes from restoring the duplicate tab row.
            tabs._set_grid_segmented_button = (  # type: ignore[attr-defined]
                lambda *a, **k: None
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self.after(50, _hide_builtin_tab_strip)
            self.after(250, _hide_builtin_tab_strip)
        except Exception:  # noqa: BLE001
            pass

        try:
            # Unified Agent Canvas owns tab_assistant's whole surface (pack-based);
            # Perception / Memory & Settings keep their own grid.
            tab_perception.grid_columnconfigure(0, weight=1)
            tab_perception.grid_rowconfigure(0, weight=1)
            tab_memory.grid_columnconfigure(0, weight=1)
            tab_memory.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        self._build_unified_canvas(tab_assistant)
        self._build_spec_approval_host(tab_assistant)
        self._build_perception_tab(tab_perception)
        try:
            from dana.ui.tooltips import attach_tooltip

            # Perception tab tip via header segment (best-effort).
            attach_tooltip(
                self._header_seg,
                "Perception tab: view live screen captures, camera feeds, and "
                "spatial tracking buffers. Other segments open Assistant or Settings.",
            )
        except Exception:  # noqa: BLE001
            pass

        # Memory & Settings: configuration only (telemetry lives in Diagnostics overlay).
        mem_scroll = ctk.CTkScrollableFrame(tab_memory, fg_color=_UI_CANVAS)
        mem_scroll.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        try:
            mem_scroll.grid_columnconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass
        # Row 1 — Engine Runtime · Row 2 — Behavior Mixer ·
        # Row 3 — Memory + Appearance · Row 4 — Updates + Dictation.
        self._build_settings_tab(mem_scroll)
        self._build_behavior_tab(mem_scroll)
        self._build_memory_appearance_row(mem_scroll)
        self._build_updates_dictation_row(mem_scroll)
        self._build_developer_diagnostics(mem_scroll)

        try:
            tabs.set("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass
        _hide_builtin_tab_strip()

        try:
            self.after(300, self.refresh_dictation_sessions)
            self.after(350, self._reload_behavior_sliders)
        except Exception:  # noqa: BLE001
            pass

    def _on_header_tab(self, name: str) -> None:
        self._select_tab(str(name))

    def _assistant_tab(self) -> Any | None:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return None
        try:
            return tabs.tab("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            return None

    def _toggle_tasks_drawer(self) -> None:
        """Expand / collapse the Task Tracker overlay over the Workspace Inspector."""
        if bool(getattr(self, "_tasks_drawer_visible", False)):
            self._collapse_tasks_drawer()
        else:
            self._expand_tasks_drawer()

    def _collapse_tasks_drawer(self) -> None:
        frame = getattr(self, "task_tracker_frame", None) or getattr(
            self, "_assistant_side", None
        )
        if frame is None:
            self._tasks_drawer_visible = False
            return
        self._tasks_drawer_visible = False
        try:
            frame.place_forget()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._tasks_toggle_btn is not None:
                self._tasks_toggle_btn.configure(text="Tasks ▸")
        except Exception:  # noqa: BLE001
            pass

    def _expand_tasks_drawer(self) -> None:
        frame = getattr(self, "task_tracker_frame", None) or getattr(
            self, "_assistant_side", None
        )
        if frame is None:
            self._tasks_drawer_visible = True
            return
        self._tasks_drawer_visible = True
        try:
            frame.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
            frame.lift()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._tasks_toggle_btn is not None:
                self._tasks_toggle_btn.configure(text="Tasks")
        except Exception:  # noqa: BLE001
            pass

    def _restore_assistant_layout(self) -> None:
        """Re-apply the Unified Canvas + drawer overlays after tab switches remount the page."""
        tab = self._assistant_tab()
        if tab is None:
            return
        frame = getattr(self, "_canvas_frame", None)
        if frame is not None:
            try:
                frame.pack(fill="both", expand=True, padx=14, pady=(10, 4))
            except Exception:  # noqa: BLE001
                pass
        if bool(getattr(self, "_tasks_drawer_visible", False)):
            self._expand_tasks_drawer()
        if bool(getattr(self, "_dag_drawer_visible", False)):
            self._expand_dag_drawer()
        if bool(getattr(self, "_spec_approval_visible", False)):
            host = getattr(self, "_spec_approval_host", None)
            if host is not None:
                try:
                    host.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
                    host.lift()
                except Exception:  # noqa: BLE001
                    pass

    def _select_tab(self, name: str) -> None:
        """Switch notebook by tab name (stable across reorder)."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        try:
            tabs.set(str(name))
        except Exception:  # noqa: BLE001
            pass
        seg = getattr(self, "_header_seg", None)
        if seg is not None:
            try:
                seg.set(str(name))
            except Exception:  # noqa: BLE001
                pass
        # CTkTabview remounts pages — re-lock the 65/35 Assistant layout.
        if str(name) == "Assistant & Tasks":
            try:
                self.after(10, self._restore_assistant_layout)
            except Exception:  # noqa: BLE001
                self._restore_assistant_layout()

    def _build_unified_canvas(self, tab) -> None:  # noqa: ANN001
        """Unified Agent Canvas — single 60/40 split-pane dashboard.

        Left (60%): Conversation + input. Right (40%): Neural Stream telemetry
        (top) + Artifact Viewer (bottom). Task Tracker and the DAG monitor are
        preserved as toggleable overlay drawers (place/place_forget) so the
        header's existing Tasks / DAG controls keep working without a
        permanent third column competing for width.
        """
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        app_frame = ctk.CTkFrame(tab, fg_color="transparent")
        app_frame.pack(fill="both", expand=True, padx=14, pady=(10, 4))
        self._canvas_frame = app_frame
        try:
            app_frame.grid_columnconfigure(0, weight=6)
            app_frame.grid_columnconfigure(1, weight=4)
            app_frame.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        # ---- Left Pane (60%) — Chat & Interaction --------------------------
        left = ctk.CTkFrame(app_frame, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._assistant_main = left
        try:
            left.grid_columnconfigure(0, weight=1)
            left.grid_rowconfigure(0, weight=1)
            left.grid_rowconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass

        # Status / wake chrome live in the header HUD (mic + system labels).
        self.status_value = None
        self.wake_value = None
        self._engine_status_lbl = None
        self._engine_warn_lbl = None
        self._vault_warn_lbl = None
        # Keep header-created mic / system labels — do not null them here.

        # Ensure monitor bus exists so background graph streams can publish.
        try:
            from dana.graph.monitor_bus import get_monitor_bus

            get_monitor_bus(create=True)
        except Exception:  # noqa: BLE001
            pass

        # _make_card packs its shell into parent; re-grid the shell so siblings
        # on ``left`` can use grid (Tk forbids mixing pack+grid on one parent).
        chat_card = self._make_card(
            left, title="Conversation", padx=0, pady=(0, 0), expand=True
        )
        chat_shell = getattr(chat_card, "master", None)
        try:
            if chat_shell is not None:
                chat_shell.pack_forget()
                chat_shell.grid(row=0, column=0, sticky="nsew")
        except Exception:  # noqa: BLE001
            try:
                chat_card.grid(row=0, column=0, sticky="nsew")
            except Exception:  # noqa: BLE001
                pass

        self._chat_view = None
        try:
            from dana.ui.chat_view import ChatBubbleView

            self._chat_view = ChatBubbleView(chat_card, wraplength=480)
            self._chat_view.pack(fill="both", expand=True)
            self.transcript_box = self._chat_view.transcript_box
            try:
                left.bind(
                    "<Configure>",
                    lambda e: self._on_chat_host_configure(e),
                    add="+",
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: ChatBubbleView unavailable ({exc})")
            self.transcript_box = ctk.CTkTextbox(
                chat_card,
                wrap="word",
                font=ctk.CTkFont(family="Segoe UI", size=14),
                fg_color=_UI_CANVAS,
                corner_radius=12,
                border_width=0,
            )
            self.transcript_box.pack(fill="both", expand=True)
        self._init_persona_transcript_tags()
        welcome = "Type below or say Dana, then speak."
        try:
            self.transcript_box.configure(state="normal")
            self.transcript_box.insert("1.0", f"[Dana] {welcome}\n\n")
            self.transcript_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        if self._chat_view is not None:
            try:
                self._chat_view.append_bubble("Dana", welcome, agent_id="broker")
            except Exception:  # noqa: BLE001
                pass

        # Input row — entry + send button, bottom of the left pane's stack.
        input_row = ctk.CTkFrame(left, fg_color="transparent")
        input_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.bottom_input_frame = input_row
        try:
            input_row.grid_columnconfigure(0, weight=1)
            input_row.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        self.chat_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="Type below or say Dana, then speak.",
            height=36,
            corner_radius=10,
            fg_color=_UI_GHOST,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            placeholder_text_color=_UI_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", self.submit_text_command)
        self._chat_send_btn = ctk.CTkButton(
            input_row,
            text="Send",
            width=92,
            height=36,
            corner_radius=999,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.submit_text_command,
        )
        self._chat_send_btn.grid(row=0, column=1, sticky="e")
        # Brief STANDBY toast ("Please Engage Engine First.") — see
        # _flash_engine_warning; empty text reserves the row so it doesn't
        # reflow the input row when a warning flashes/clears.
        self._engine_warn_lbl = ctk.CTkLabel(
            input_row,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F59E0B",
            anchor="w",
        )
        self._engine_warn_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        # Persistent banner (not a toast — stays until unlocked) so a locked
        # vault at startup is visible instead of the wake-word thread just
        # never starting with no explanation.
        self._vault_warn_lbl = ctk.CTkLabel(
            input_row,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#EF4444",
            anchor="w",
        )
        self._vault_warn_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # DAG monitor — collapsible overlay drawer above the input row.
        self._build_dag_monitor_section(left)
        try:
            self.bind("<Control-Shift-D>", lambda _e: self._toggle_dag_drawer())
            self.bind("<Control-Shift-d>", lambda _e: self._toggle_dag_drawer())
        except Exception:  # noqa: BLE001
            pass

        # ---- Right Pane (40%) — Workspace Inspector -------------------------
        right = ctk.CTkFrame(app_frame, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._workspace_inspector = right
        try:
            right.grid_columnconfigure(0, weight=1)
            right.grid_rowconfigure(0, weight=1)
            right.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        self._build_neural_stream_pane(right)
        self._build_artifact_viewer_pane(right)

        # Task Tracker — collapsible overlay drawer covering the inspector pane.
        self._build_task_tracker_section(right)

        try:
            self.engage_engine()
        except Exception:  # noqa: BLE001
            try:
                self._refresh_engine_ui()
            except Exception:  # noqa: BLE001
                pass

    def _build_neural_stream_pane(self, parent) -> None:  # noqa: ANN001
        """Top half of the Workspace Inspector — live color-coded telemetry."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        try:
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            card,
            text="Neural Stream",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        body = tk.Frame(card, bg=_UI_CANVAS, highlightthickness=0)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        try:
            body.grid_columnconfigure(0, weight=1)
            body.grid_rowconfigure(0, weight=1)
        except Exception:  # noqa: BLE001
            pass

        text = tk.Text(
            body,
            wrap="word",
            height=10,
            bg=_UI_CANVAS,
            fg=_UI_TEXT,
            insertbackground=_UI_TEXT,
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
        )
        text.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        text.tag_configure("error", foreground="#ff4444")
        text.tag_configure("tool", foreground="#00cc66")
        text.tag_configure("thought", foreground="#3399ff")
        text.configure(state="disabled")
        self._neural_stream_text = text

    def _build_artifact_viewer_pane(self, parent) -> None:  # noqa: ANN001
        """Bottom half of the Workspace Inspector — workspace code / file preview."""
        card = ctk.CTkFrame(
            parent,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        try:
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            card,
            text="Artifact Viewer",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        box = ctk.CTkTextbox(
            card,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=_UI_CANVAS,
            corner_radius=8,
            border_width=0,
        )
        box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        box.insert("1.0", "// No file selected.\n")
        box.configure(state="disabled")
        self.artifact_viewer = box

    def show_artifact(self, title: str, content: str) -> None:
        """Preview a file/code snippet in the Artifact Viewer pane."""
        box = getattr(self, "artifact_viewer", None)
        if box is None:
            return
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", f"# {title}\n\n{content}")
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _build_dag_monitor_section(self, main) -> None:  # noqa: ANN001
        """Collapsible live DAG drawer — overlays the conversation pane when expanded."""
        shell = ctk.CTkFrame(
            main,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            height=220,
        )
        self.dag_monitor_frame = shell
        self._dag_drawer_visible = False
        try:
            shell.grid_propagate(False)
            shell.grid_columnconfigure(0, weight=1)
            shell.grid_rowconfigure(0, weight=0)
            shell.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        hdr = ctk.CTkFrame(shell, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        try:
            hdr.grid_columnconfigure(0, weight=1)
            hdr.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            hdr,
            text="DAG Execution",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        right_hdr = ctk.CTkFrame(hdr, fg_color="transparent")
        right_hdr.grid(row=0, column=1, sticky="e")
        try:
            from dana.graph.cloud_planner import planner_mode_label

            _planner_mode = planner_mode_label()
        except Exception:  # noqa: BLE001
            _planner_mode = "LOCAL"
        self._dag_planner_mode_lbl = ctk.CTkLabel(
            right_hdr,
            text=f"Planner Mode: [{_planner_mode}]",
            font=ctk.CTkFont(size=10),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._dag_planner_mode_lbl.pack(side="top", anchor="e")
        self._dag_status_lbl = ctk.CTkLabel(
            right_hdr,
            text="Idle",
            font=ctk.CTkFont(size=11),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._dag_status_lbl.pack(side="top", anchor="e")

        try:
            from dana.ui.dag_monitor_view import DagMonitorView
            from dana.ui.tooltips import attach_tooltip

            self.dag_monitor_view = DagMonitorView(
                shell,
                poll_ms=250,
                show_header=False,
                status_label=self._dag_status_lbl,
                on_multistep=self._on_dag_multistep,
                on_complete=self._on_dag_complete,
                external_tick=True,
            )
            self.dag_monitor_view.grid(
                row=1, column=0, sticky="nsew", padx=4, pady=(0, 8)
            )
            attach_tooltip(
                self.dag_monitor_view,
                "Live supervisor plan, worker status, and tool staging events.",
            )
        except Exception as exc:  # noqa: BLE001
            self.dag_monitor_view = None
            log("UI", f"WARNING: DagMonitorView unavailable ({exc})")
            ctk.CTkLabel(
                shell,
                text="DAG monitor unavailable",
                text_color=_UI_MUTED,
            ).grid(row=1, column=0, sticky="nw", padx=12, pady=12)

        # Start collapsed (unplaced); multi-step plans auto-expand via place().

    def _on_dag_multistep(self) -> None:
        """Auto-expand the DAG drawer when a multi-step plan arrives."""
        try:
            self.after(0, self._expand_dag_drawer)
        except Exception:  # noqa: BLE001
            self._expand_dag_drawer()

    def _on_dag_complete(self, summary: str) -> None:
        try:
            log("UI", f"DAG complete: {summary}")
        except Exception:  # noqa: BLE001
            pass

    def _toggle_dag_drawer(self) -> None:
        if bool(getattr(self, "_dag_drawer_visible", False)):
            self._collapse_dag_drawer()
        else:
            self._expand_dag_drawer()

    def _collapse_dag_drawer(self) -> None:
        frame = getattr(self, "dag_monitor_frame", None)
        if frame is None:
            self._dag_drawer_visible = False
            return
        self._dag_drawer_visible = False
        try:
            frame.place_forget()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._dag_toggle_btn is not None:
                self._dag_toggle_btn.configure(text="DAG ▸")
        except Exception:  # noqa: BLE001
            pass

    def _expand_dag_drawer(self) -> None:
        frame = getattr(self, "dag_monitor_frame", None)
        if frame is None:
            self._dag_drawer_visible = True
            return
        self._dag_drawer_visible = True
        try:
            frame.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0, height=260)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._dag_toggle_btn is not None:
                self._dag_toggle_btn.configure(text="DAG")
        except Exception:  # noqa: BLE001
            pass
        view = getattr(self, "dag_monitor_view", None)
        if view is not None:
            try:
                view.refresh()
            except Exception:  # noqa: BLE001
                pass

    def start_dag_monitor_stream(
        self,
        user_prompt: str,
        *,
        tool_fn: Any | None = None,
        planner: Any | None = None,
    ) -> None:
        """Run ``stream_dag_supervisor`` on a daemon thread (UI stays responsive)."""
        import threading

        def _run() -> None:
            try:
                from dana.graph.builder import stream_dag_supervisor
                from dana.graph.monitor_bus import get_monitor_bus

                get_monitor_bus(create=True)
                for _chunk in stream_dag_supervisor(
                    user_prompt,
                    planner=planner,
                    tool_fn=tool_fn,
                    monitor=True,
                ):
                    pass
            except Exception as exc:  # noqa: BLE001
                try:
                    log("UI", f"DAG stream error: {exc}")
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=_run, name="dana-dag-monitor", daemon=True)
        self._dag_stream_thread = t
        t.start()

    def _build_spec_approval_host(self, tab) -> None:  # noqa: ANN001
        """HITL Spec Approval Card host (row 1) — hidden until a draft is ready."""
        host = ctk.CTkFrame(tab, fg_color="transparent")
        self._spec_approval_host = host
        self._spec_approval_visible = False
        self._spec_approval_card = None
        self._pending_spec_payload: dict[str, Any] | None = None
        try:
            from dana.ui.spec_approval_view import SpecApprovalCard

            card = SpecApprovalCard(
                host,
                on_approve=self._on_spec_approve,
                on_edit=self._on_spec_edit,
                on_cancel=self._on_spec_cancel,
            )
            card.pack(fill="x", expand=False)
            self._spec_approval_card = card
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: SpecApprovalCard unavailable ({exc})")
            self._spec_approval_card = None
        # Start collapsed (unplaced) — shown as a bottom overlay on demand.

    def show_spec_approval(self, payload: dict[str, Any]) -> None:
        """Present a compiled ``/broker`` draft and wait for Approve / Edit / Cancel."""
        self._pending_spec_payload = dict(payload or {})
        card = getattr(self, "_spec_approval_card", None)
        host = getattr(self, "_spec_approval_host", None)
        if card is None or host is None:
            return
        try:
            self._select_tab("Assistant & Tasks")
        except Exception:  # noqa: BLE001
            pass
        try:
            card.present(self._pending_spec_payload)
        except Exception:  # noqa: BLE001
            pass
        self._spec_approval_visible = True
        try:
            host.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            host.lift()
        except Exception:  # noqa: BLE001
            pass
        try:
            emit_live_transcript(
                "Dana",
                "Spec compiled — review the Approval Card, then Approve & Run.",
            )
        except Exception:  # noqa: BLE001
            pass

    def hide_spec_approval(self) -> None:
        host = getattr(self, "_spec_approval_host", None)
        self._spec_approval_visible = False
        self._pending_spec_payload = None
        if host is None:
            return
        try:
            host.place_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_approve(self, compiled_spec: str) -> None:
        """Dispatch the approved macro to Meta-Broker (approved=True bypasses HITL)."""
        macro = str(compiled_spec or "").strip()
        self.hide_spec_approval()
        if not macro:
            return
        try:
            emit_live_transcript("Dana", "Approved — dispatching Meta-Broker…")
        except Exception:  # noqa: BLE001
            pass

        def _run() -> None:
            try:
                obs = execute_tool_call(
                    ToolCall(
                        tool_id="meta_broker",
                        arguments={"prompt": macro, "approved": True},
                        raw_text=macro,
                        confidence=0.99,
                    )
                )
                spoken = str(obs or "")
                if spoken.startswith("OK:"):
                    lines = [
                        ln.strip()
                        for ln in spoken.splitlines()
                        if ln.strip() and not ln.startswith("epic_log:")
                    ]
                    spoken = lines[1] if len(lines) > 1 else lines[0]
                if len(spoken) > 420:
                    spoken = spoken[:417] + "..."
                try:
                    log_conversation("Dana", spoken)
                    emit_live_transcript("Dana", spoken)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                try:
                    emit_live_transcript(
                        "Dana", f"Meta-Broker dispatch failed: {exc}"
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            import threading

            threading.Thread(
                target=_run, name="dana-spec-approve", daemon=True
            ).start()
        except Exception:  # noqa: BLE001
            _run()

    def _on_spec_edit(self, compiled_spec: str) -> None:
        """Copy the compiled macro into the chat input for manual tweaking."""
        macro = str(compiled_spec or "").strip()
        self.hide_spec_approval()
        entry = getattr(self, "chat_entry", None)
        if entry is not None and macro:
            try:
                entry.delete(0, "end")
                entry.insert(0, macro)
                entry.focus_set()
            except Exception:  # noqa: BLE001
                pass
        try:
            emit_live_transcript(
                "Dana",
                "Macro copied to input — edit freely, then send to re-submit.",
            )
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_cancel(self) -> None:
        self.hide_spec_approval()
        try:
            emit_live_transcript("Dana", "Spec approval cancelled — Meta-Broker not started.")
        except Exception:  # noqa: BLE001
            pass

    def _on_chat_host_configure(self, event: Any) -> None:
        """Keep bubble wraplength proportional to Conversation width."""
        view = getattr(self, "_chat_view", None)
        if view is None or event is None:
            return
        try:
            width = int(getattr(event, "width", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        if width < 240:
            return
        # Leave room for bubble chrome / scrollbar; cap so text stays inside pane.
        wrap = max(220, min(560, width - 96))
        try:
            setter = getattr(view, "set_wraplength", None)
            if callable(setter):
                setter(wrap)
            else:
                view._wraplength = wrap  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def submit_text_command(self, event=None):  # noqa: ANN001
        """Stage 8.10 — inject typed text as the next user utterance (no STT)."""
        if not self._require_engine():
            return "break" if event is not None else None
        entry = self.chat_entry
        if entry is None:
            return "break" if event is not None else None
        try:
            text = str(entry.get() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            return "break" if event is not None else None
        try:
            entry.delete(0, "end")
        except Exception:  # noqa: BLE001
            pass
        # Instant distinct echo — conversation path will not re-log.
        try:
            self.log_transcript("User (Text)", text)
        except Exception:  # noqa: BLE001
            pass
        try:
            set_injected_question(text, source="text", already_logged=True)
            is_recording.set()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: silent text inject failed ({exc})")
            self._flash_engine_warning("Could not dispatch text command.")
            return "break" if event is not None else None
        preview = text if len(text) <= 120 else text[:117] + "..."
        log("UI", f'Silent text → LangGraph: "{preview}"')
        try:
            set_subtitle(f'User (Text): "{text}"')
        except Exception:  # noqa: BLE001
            pass
        return "break" if event is not None else None

    def _build_task_tracker_section(self, tab) -> None:  # noqa: ANN001
        """Task Tracker overlay drawer — covers the Workspace Inspector when expanded."""
        side = ctk.CTkFrame(
            tab,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        self._assistant_side = side
        self.task_tracker_frame = side
        try:
            side.grid_columnconfigure(0, weight=1)
            side.grid_rowconfigure(0, weight=0)
            side.grid_rowconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass
        # Start collapsed (unplaced); Tasks header toggle brings it forward.

        # Permanent chrome — never pack_forget / grid_forget these inners.
        hdr = ctk.CTkFrame(side, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        try:
            hdr.grid_columnconfigure(0, weight=1)
            hdr.grid_columnconfigure(1, weight=0)
        except Exception:  # noqa: BLE001
            pass
        ctk.CTkLabel(
            hdr,
            text="Task Tracker",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._tasks_empty_lbl = ctk.CTkLabel(
            hdr,
            text="No active tasks",
            font=ctk.CTkFont(size=11),
            text_color=_UI_MUTED,
            anchor="e",
        )
        self._tasks_empty_lbl.grid(row=0, column=1, sticky="e")

        try:
            from dana.ui.task_tracker_view import TaskTrackerView
            from dana.ui.tooltips import attach_tooltip

            self.task_tracker_view = TaskTrackerView(
                side,
                poll_ms=400,
                show_header=False,
                status_label=self._tasks_empty_lbl,
                external_tick=True,
            )
            self.task_tracker_view.grid(
                row=1, column=0, sticky="nsew", padx=4, pady=(0, 8)
            )
            attach_tooltip(
                self.task_tracker_view,
                "Shows real-time status of background Python sandbox scripts "
                "and research swarms.",
            )
        except Exception as exc:  # noqa: BLE001
            self.task_tracker_view = None
            log("UI", f"WARNING: TaskTrackerView unavailable ({exc})")
            ctk.CTkLabel(
                side,
                text="Task Tracker unavailable",
                text_color=_UI_MUTED,
            ).grid(row=1, column=0, sticky="nw", padx=12, pady=12)

    def _build_developer_diagnostics(self, tab) -> None:  # noqa: ANN001
        """Collapsible Live Trace / LangGraph diagnostics (Settings tab)."""
        self._diag_expanded = False
        self.live_trace = None
        self.trace_scroll = None
        self._diag_expander = None
        shell = ctk.CTkFrame(tab, fg_color="transparent")
        shell.pack(fill="x", expand=False, padx=8, pady=(4, 8))
        self._diag_shell = shell
        self._diag_btn = ctk.CTkButton(
            shell,
            text="▸ Developer Diagnostics",
            anchor="w",
            height=28,
            corner_radius=8,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
            command=self._toggle_developer_diagnostics,
        )
        self._diag_btn.pack(fill="x")
        self._diag_body = ctk.CTkFrame(shell, fg_color=_UI_CANVAS, height=220)
        # Hidden until toggled.
        try:
            from dana.ui.trace_window import LiveTracePanel

            self.live_trace = LiveTracePanel(
                self._diag_body, poll_ms=80, external_tick=True
            )
            self.live_trace.pack(fill="both", expand=True, padx=2, pady=2)
            self.trace_scroll = self.live_trace.timeline
        except Exception:  # noqa: BLE001
            self.live_trace = None
            fallback = self._make_card(self._diag_body, title="Pipeline stages")
            self.trace_scroll = ctk.CTkScrollableFrame(fallback, fg_color="transparent")
            self.trace_scroll.pack(fill="both", expand=True)
        self._diag_expander = self

    def _toggle_developer_diagnostics(self) -> None:
        body = getattr(self, "_diag_body", None)
        btn = getattr(self, "_diag_btn", None)
        if body is None:
            return
        self._diag_expanded = not bool(getattr(self, "_diag_expanded", False))
        try:
            if self._diag_expanded:
                body.pack(fill="both", expand=True, pady=(6, 0))
                if btn is not None:
                    btn.configure(text="▾ Developer Diagnostics")
            else:
                body.pack_forget()
                if btn is not None:
                    btn.configure(text="▸ Developer Diagnostics")
        except Exception:  # noqa: BLE001
            pass

    def _build_perception_tab(self, tab) -> None:  # noqa: ANN001
        """Compact Perception idle; preview expands only when vision is active."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        self._perception_grid = grid
        self._perception_preview_active = False

        # Compact idle banner (always visible).
        standby_card = ctk.CTkFrame(
            grid,
            fg_color=_UI_CARD,
            corner_radius=14,
            border_width=1,
            border_color=_UI_CARD_BORDER,
        )
        standby_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 8))
        self._vision_standby_lbl = ctk.CTkLabel(
            standby_card,
            text="Vision Standby — Ready for screen OCR / YOLO",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        )
        self._vision_standby_lbl.pack(fill="x", padx=14, pady=(12, 4))
        self._vision_status_lbl = ctk.CTkLabel(
            standby_card,
            text="Hybrid grounding: idle · ROI overlay: standby",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=640,
            justify="left",
        )
        self._vision_status_lbl.pack(fill="x", padx=14, pady=(0, 8))
        btn_row = ctk.CTkFrame(standby_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(
            btn_row,
            text="Refresh status",
            width=120,
            height=28,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            command=self._refresh_perception_status,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="Inspect UIA tree",
            width=130,
            height=28,
            corner_radius=999,
            command=self._inspect_uia_tree,
        ).pack(side="left")

        # Expandable workspace — hidden until a visual task is active.
        workspace = ctk.CTkFrame(grid, fg_color="transparent")
        self._perception_workspace = workspace
        workspace.grid_columnconfigure(0, weight=1, uniform="perc")
        workspace.grid_columnconfigure(1, weight=1, uniform="perc")
        workspace.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(workspace, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(workspace, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)

        roi_card = self._make_card(left, title="Live Perception feed", padx=4, pady=(0, 0))
        self._roi_preview_lbl = ctk.CTkLabel(
            roi_card,
            text="Live screen feed idle — open this tab to stream (~8 FPS).",
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=360,
            justify="left",
        )
        self._roi_preview_lbl.pack(fill="x", pady=(0, 8))
        self._roi_overlay_lbl = ctk.CTkLabel(
            roi_card,
            text="Grounding overlays: standby",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=360,
            justify="left",
        )
        self._roi_overlay_lbl.pack(fill="x", pady=(0, 8))
        self._roi_canvas = ctk.CTkFrame(
            roi_card,
            fg_color=_UI_CANVAS,
            corner_radius=12,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            height=220,
        )
        self._roi_canvas.pack(fill="both", expand=True, pady=(0, 4))
        self._perception_feed_lbl = ctk.CTkLabel(
            self._roi_canvas,
            text="Screen / ROI preview",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._perception_feed_lbl.pack(expand=True, fill="both", pady=8, padx=8)
        self._perception_feed_img = None

        tree_card = self._make_card(right, title="Win32 UIA tree", padx=4, pady=(0, 0))
        self._uia_tree_box = ctk.CTkTextbox(
            tree_card,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=12,
            border_width=0,
        )
        self._uia_tree_box.pack(fill="both", expand=True)
        self._uia_tree_box.insert(
            "1.0",
            "Click “Inspect UIA tree” to dump the foreground window hierarchy "
            "(best-effort; requires Windows UIA backends).\n",
        )
        self._uia_tree_box.configure(state="disabled")
        try:
            self.after(600, self._refresh_perception_status)
        except Exception:  # noqa: BLE001
            pass

    def _set_perception_preview_visible(self, active: bool) -> None:
        """Show ROI/UIA workspace only while a visual task is live."""
        workspace = getattr(self, "_perception_workspace", None)
        if workspace is None:
            return
        want = bool(active)
        prev = bool(getattr(self, "_perception_preview_active", False))
        self._perception_preview_active = want
        if want != prev:
            try:
                if want:
                    workspace.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
                else:
                    workspace.grid_remove()
            except Exception:  # noqa: BLE001
                pass
        standby = getattr(self, "_vision_standby_lbl", None)
        if standby is not None:
            try:
                if want:
                    standby.configure(
                        text="Vision Active — screen OCR / YOLO",
                        text_color=_UI_EMERALD,
                    )
                else:
                    standby.configure(
                        text="Vision Standby — Ready for screen OCR / YOLO",
                        text_color=_UI_MUTED,
                    )
            except Exception:  # noqa: BLE001
                pass

    def _refresh_perception_status(self) -> None:
        bits: list[str] = []
        overlay_txt = "Grounding overlays: standby"
        vision_active = False
        try:
            from dana.graph.nodes.vision import get_hybrid_grounding

            grounder = get_hybrid_grounding()
            bits.append(f"Hybrid grounding: {type(grounder).__name__}")
        except Exception as exc:  # noqa: BLE001
            bits.append(f"Hybrid grounding: unavailable ({type(exc).__name__})")
        try:
            from dana.vision.overlay import get_overlay

            overlay = get_overlay()
            visible = bool(getattr(overlay, "_visible", False))
            current = getattr(overlay, "_current", None)
            vision_active = bool(visible or current)
            bits.append(
                f"ROI overlay: {'visible' if visible else 'standby'}"
                + (f" @ {current}" if current else "")
            )
            overlay_txt = (
                f"Grounding overlays: {'visible' if visible else 'standby'}"
                + (f" @ {current}" if current else "")
            )
            roi_lbl = getattr(self, "_roi_preview_lbl", None)
            if roi_lbl is not None:
                if current:
                    label = str(getattr(overlay, "_label", "") or "")
                    roi_lbl.configure(
                        text=f"Last ROI {current}"
                        + (f" — {label}" if label else "")
                    )
                else:
                    roi_lbl.configure(
                        text="No ROI captured yet. Grounding hits appear here."
                    )
        except Exception as exc:  # noqa: BLE001
            bits.append(f"ROI overlay: unavailable ({type(exc).__name__})")
            overlay_txt = f"Grounding overlays: unavailable ({type(exc).__name__})"
        lbl = getattr(self, "_vision_status_lbl", None)
        if lbl is not None:
            try:
                lbl.configure(text=" · ".join(bits) if bits else "Vision idle")
            except Exception:  # noqa: BLE001
                pass
        ol = getattr(self, "_roi_overlay_lbl", None)
        if ol is not None:
            try:
                ol.configure(text=overlay_txt)
            except Exception:  # noqa: BLE001
                pass
        try:
            self._set_perception_preview_visible(vision_active)
        except Exception:  # noqa: BLE001
            pass

    def _format_uia_tree_text(self, raw: Any) -> str:
        """Indent UIA dump lines for readability."""
        if isinstance(raw, (list, tuple)):
            lines: list[str] = []
            for item in list(raw)[:120]:
                s = str(item)
                # Heuristic depth from leading spaces / pipe / depth keys.
                if isinstance(item, dict):
                    depth = int(item.get("depth") or item.get("level") or 0)
                    name = (
                        item.get("name")
                        or item.get("control_type")
                        or item.get("AutomationId")
                        or s
                    )
                    lines.append(("  " * max(0, depth)) + str(name))
                else:
                    stripped = s.lstrip()
                    lead = len(s) - len(stripped)
                    depth = lead // 2 if lead else 0
                    # Nested markers like "└─" / "|-" keep as-is; else indent.
                    if stripped.startswith(("└", "├", "|", "+", "-")):
                        lines.append(s)
                    else:
                        lines.append(("  " * depth) + stripped)
            return "\n".join(lines) or "(no controls)"
        text = str(raw)
        out: list[str] = []
        for line in text.splitlines():
            stripped = line.lstrip(" \t")
            lead = len(line) - len(stripped)
            indent = "  " * (lead // 2) if lead else ""
            out.append(indent + stripped if lead else line)
        return "\n".join(out) if out else text

    def _inspect_uia_tree(self) -> None:
        box = getattr(self, "_uia_tree_box", None)
        if box is None:
            return

        def _worker() -> None:
            text = "(empty)"
            try:
                from dana.vision.uia_provider import Win32UIAProvider

                provider = Win32UIAProvider()
                dump = getattr(provider, "dump_tree", None) or getattr(
                    provider, "list_controls", None
                )
                if callable(dump):
                    result = dump()
                    text = self._format_uia_tree_text(result)
                else:
                    # Best-effort: probe find_element path existence.
                    text = (
                        "UIA provider loaded. No dump_tree API — "
                        f"provider={type(provider).__name__}. "
                        "Hybrid grounding still uses UIA hit-testing at runtime."
                    )
            except Exception as exc:  # noqa: BLE001
                text = f"UIA inspect failed: {type(exc).__name__}: {exc}"

            def _ui() -> None:
                try:
                    box.configure(state="normal")
                    box.delete("1.0", "end")
                    box.insert("1.0", text + "\n")
                    box.configure(state="disabled")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._refresh_perception_status()
                except Exception:  # noqa: BLE001
                    pass
                # Keep workspace open after inspect even if ROI is idle.
                try:
                    self._perception_preview_active = False
                    self._set_perception_preview_visible(True)
                except Exception:  # noqa: BLE001
                    pass

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        try:
            self._perception_preview_active = False
            self._set_perception_preview_visible(True)
        except Exception:  # noqa: BLE001
            pass
        threading.Thread(target=_worker, name="UIAInspect", daemon=True).start()

    def _perception_tab_visible(self) -> bool:
        """True when the Perception tab is the active notebook page."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return False
        try:
            return str(tabs.get()) == "Perception"
        except Exception:  # noqa: BLE001
            return False

    def _schedule_perception_feed(self) -> None:
        """Lightweight ~8 FPS mss feed while Perception tab is visible."""
        if not self.winfo_exists():
            return
        try:
            if self._perception_tab_visible():
                self._set_perception_preview_visible(True)
                if not bool(getattr(self, "_perception_feed_busy", False)):
                    self._perception_feed_busy = True
                    threading.Thread(
                        target=self._capture_perception_frame,
                        name="PerceptionFeed",
                        daemon=True,
                    ).start()
            else:
                # Idle when tab hidden — keep UI responsive.
                lbl = getattr(self, "_roi_preview_lbl", None)
                if lbl is not None:
                    try:
                        lbl.configure(
                            text="Live screen feed idle — open this tab to stream (~8 FPS)."
                        )
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        try:
            self._perception_feed_job = self.after(125, self._schedule_perception_feed)
        except Exception:  # noqa: BLE001
            self._perception_feed_job = None

    def _capture_perception_frame(self) -> None:
        """Background: grab primary monitor via mss, downscale, push to UI."""
        pil_img = None
        err = ""
        try:
            import mss
            from PIL import Image

            factory = getattr(mss, "mss", None) or getattr(mss, "MSS", None)
            with factory() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                # Downscale for ~5–10 FPS UI (non-blocking).
                resample = getattr(
                    getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR
                )
                img.thumbnail((480, 270), resample)
                pil_img = img
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        def _apply() -> None:
            self._perception_feed_busy = False
            feed = getattr(self, "_perception_feed_lbl", None)
            if feed is None or not self.winfo_exists():
                return
            if pil_img is None:
                try:
                    feed.configure(image=None, text=err or "Capture unavailable")
                    self._perception_feed_img = None
                except Exception:  # noqa: BLE001
                    pass
                return
            try:
                ctk_img = ctk.CTkImage(
                    light_image=pil_img,
                    dark_image=pil_img,
                    size=pil_img.size,
                )
                self._perception_feed_img = ctk_img
                feed.configure(image=ctk_img, text="")
                status = getattr(self, "_roi_preview_lbl", None)
                if status is not None:
                    status.configure(text=f"Live feed {pil_img.size[0]}×{pil_img.size[1]}")
            except Exception as exc:  # noqa: BLE001
                try:
                    feed.configure(text=f"Feed error: {type(exc).__name__}")
                except Exception:  # noqa: BLE001
                    pass

        try:
            self.after(0, _apply)
        except Exception:  # noqa: BLE001
            self._perception_feed_busy = False

    def _build_memory_appearance_row(self, tab) -> None:  # noqa: ANN001
        """Row 3 — Episodic Memory search + Appearance theme."""
        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(fill="x", expand=False, padx=4, pady=(4, 4))
        try:
            grid.grid_columnconfigure(0, weight=1, uniform="mem3")
            grid.grid_columnconfigure(1, weight=1, uniform="mem3")
        except Exception:  # noqa: BLE001
            pass
        left = ctk.CTkFrame(grid, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(grid, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self._build_episodic_memory_section(left)
        self._build_appearance_card(right)

    def _build_updates_dictation_row(self, tab) -> None:  # noqa: ANN001
        """Row 4 — System Updates + Dictation Latch (no live session telemetry)."""
        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(fill="x", expand=False, padx=4, pady=(4, 8))
        try:
            grid.grid_columnconfigure(0, weight=1, uniform="mem4")
            grid.grid_columnconfigure(1, weight=1, uniform="mem4")
        except Exception:  # noqa: BLE001
            pass
        left = ctk.CTkFrame(grid, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        right = ctk.CTkFrame(grid, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self._build_system_updates_card(left)
        self._build_dictation_tab(right)

    def _build_memory_settings_grid(self, tab) -> None:  # noqa: ANN001
        """Legacy entrypoint — delegates to the reordered Memory & Settings rows."""
        self._build_memory_appearance_row(tab)
        self._build_updates_dictation_row(tab)

    def _build_system_updates_card(self, tab) -> None:  # noqa: ANN001
        """Stage 9.3 — System Updates card (left column) + Phase 1 OTA chrome."""
        updates = self._make_card(
            tab, title="System Updates", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            updates,
            text=(
                "Fetch from GitHub, compare revisions, then update dependencies "
                "and restart Dānā in one click."
            ),
            anchor="w",
            justify="left",
            wraplength=360,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))

        # Auto-Update Mode (Silent / Manual) — wired to OTAManifestManager.
        mode_row = ctk.CTkFrame(updates, fg_color="transparent")
        mode_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            mode_row,
            text="Auto-Update Mode",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        initial_mode = "Manual"
        try:
            from dana.updater.manifest import get_ota_manager

            self._ota_manager = get_ota_manager()
            initial_mode = (
                "Silent"
                if self._ota_manager.auto_update_mode == "silent"
                else "Manual"
            )
        except Exception:  # noqa: BLE001
            self._ota_manager = None
        self._ota_mode_var = ctk.StringVar(value=initial_mode)
        self._ota_mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=["Silent", "Manual"],
            variable=self._ota_mode_var,
            width=110,
            height=28,
            corner_radius=8,
            text_color=_UI_TEXT,
            command=self._on_ota_mode_changed,
        )
        self._ota_mode_menu.pack(side="right")

        self._ota_pill_lbl = ctk.CTkLabel(
            updates,
            text="[UP TO DATE]",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._ota_pill_lbl.pack(fill="x", pady=(0, 4))

        # Phase 2B — Active slot + staging health pills.
        _slot_active_color = _UI_EMERALD
        try:
            _slot_active_color = getattr(_UI_THEME, "STATUS_SLOT_ACTIVE", _UI_EMERALD)
        except Exception:  # noqa: BLE001
            _slot_active_color = _UI_EMERALD
        self._ota_slot_lbl = ctk.CTkLabel(
            updates,
            text="Active: Slot A (v0.0.0)",
            anchor="w",
            text_color=_slot_active_color,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._ota_slot_lbl.pack(fill="x", pady=(0, 2))
        self._ota_staging_lbl = ctk.CTkLabel(
            updates,
            text="Staging: idle",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._ota_staging_lbl.pack(fill="x", pady=(0, 4))

        self._update_status_lbl = ctk.CTkLabel(
            updates,
            text="Status: idle",
            anchor="w",
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self._update_status_lbl.pack(fill="x", pady=(0, 8))
        btn_row = ctk.CTkFrame(updates, fg_color="transparent")
        btn_row.pack(fill="x", side="bottom")
        self._update_check_btn = ctk.CTkButton(
            btn_row,
            text="Check for Updates",
            width=150,
            height=30,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_check_for_updates,
        )
        self._update_check_btn.pack(side="right", padx=(8, 0))
        self._update_apply_btn = ctk.CTkButton(
            btn_row,
            text="Update & Restart",
            width=150,
            height=30,
            corner_radius=999,
            fg_color=_UI_AMBER,
            hover_color="#D97706",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_apply_update_and_restart,
        )
        # Hidden until check_for_updates() reports True.
        self._ota_hot_apply_btn = ctk.CTkButton(
            btn_row,
            text="Hot Apply",
            width=110,
            height=30,
            corner_radius=999,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_ota_hot_apply,
        )
        # Hidden until OTAManifestManager reports a staged patch.
        try:
            self._refresh_ota_ui()
        except Exception:  # noqa: BLE001
            pass

    def _build_episodic_memory_section(self, tab) -> None:  # noqa: ANN001
        """Episodic memory keyword search (SQLite store)."""
        card = self._make_card(tab, title="Episodic Memory", padx=4, pady=(0, 8), expand=False)
        self._memory_query = ctk.CTkEntry(
            card,
            placeholder_text="Search preferences & facts…",
            height=32,
            corner_radius=8,
            fg_color=_UI_GHOST,
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
        )
        self._memory_query.pack(fill="x", pady=(0, 8))
        self._memory_query.bind("<Return>", lambda _e: self._run_episodic_search())
        self._memory_results = ctk.CTkTextbox(
            card,
            wrap="word",
            height=140,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=_UI_CANVAS,
            text_color=_UI_TEXT,
            corner_radius=10,
        )
        self._memory_results.pack(fill="x", pady=(0, 8))
        self._memory_results.insert("1.0", "(enter a query to search episodic memory)\n")
        self._memory_results.configure(state="disabled")
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x")
        ctk.CTkButton(
            actions,
            text="Search",
            width=88,
            height=30,
            corner_radius=999,
            command=self._run_episodic_search,
        ).pack(side="right")

    def _run_episodic_search(self) -> None:
        box = getattr(self, "_memory_results", None)
        entry = getattr(self, "_memory_query", None)
        if box is None:
            return
        try:
            query = str(entry.get() if entry is not None else "").strip()
        except Exception:  # noqa: BLE001
            query = ""
        try:
            from dana.memory.store import get_episodic_store

            facts = get_episodic_store().search_facts(query, limit=24)
            lines = []
            for fact in facts:
                key = fact.get("key") or ""
                val = fact.get("value") or ""
                cat = fact.get("category") or ""
                lines.append(f"[{cat}] {key} = {val}")
            text = "\n".join(lines) if lines else "(no matching facts)\n"
        except Exception as exc:  # noqa: BLE001
            text = f"Memory search failed: {type(exc).__name__}: {exc}\n"
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text if text.endswith("\n") else text + "\n")
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _build_settings_tab(self, tab) -> None:  # noqa: ANN001
        """Row 1 — Engine Runtime (Hybrid Broker, Wake Word, Startup)."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        stats_card = self._make_card(
            tab, title="Engine Runtime", padx=8, pady=(8, 4), expand=False
        )
        ctk.CTkLabel(
            stats_card,
            text="Wake word",
            anchor="w",
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))
        self._settings_wake_lbl = ctk.CTkLabel(
            stats_card,
            text="Active wake word: Dana",
            anchor="w",
            justify="left",
            wraplength=640,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=_UI_EMERALD,
        )
        self._settings_wake_lbl.pack(fill="x", pady=(0, 12))
        try:
            from dana.settings import is_open_window_on_startup

            open_on_start = bool(is_open_window_on_startup())
        except Exception:  # noqa: BLE001
            open_on_start = True
        self._open_window_var = ctk.BooleanVar(value=open_on_start)
        self._open_window_chk = ctk.CTkCheckBox(
            stats_card,
            text="Open window on startup",
            variable=self._open_window_var,
            command=self._on_open_window_startup_toggle,
            text_color=_UI_MUTED,
        )
        self._open_window_chk.pack(anchor="w", pady=(0, 8))
        try:
            from dana.settings import is_hybrid_planner_enabled

            hybrid_on = bool(is_hybrid_planner_enabled())
        except Exception:  # noqa: BLE001
            hybrid_on = False
        self._hybrid_planner_var = ctk.BooleanVar(value=hybrid_on)
        self._hybrid_planner_chk = ctk.CTkCheckBox(
            stats_card,
            text="Hybrid Broker (Cloud Planner)",
            variable=self._hybrid_planner_var,
            command=self._on_hybrid_planner_toggle,
            text_color=_UI_MUTED,
        )
        self._hybrid_planner_chk.pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            stats_card,
            text=(
                "Off by default (fully local). When on, Meta-Broker epic split "
                "and Supervisor DAG JSON may use Gemini if GEMINI_API_KEY is set "
                "in .env. Workers and the runtime harness always stay local."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            stats_card,
            text=(
                "Mic and speaker follow the OS System Default automatically. "
                "Pipeline mode is shown in the header and Assistive Orb."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 4))
        # Compatibility stubs (menus removed — autonomous System Default).
        self.mic_menu = None
        self.speaker_menu = None
        self.save_btn = None

        remote_card = self._make_card(
            tab, title="Remote Access", padx=8, pady=(0, 4), expand=False
        )
        ctk.CTkLabel(
            remote_card,
            text=(
                "Pushover push notifications and a personal Telegram bot let "
                "Dana reach you (or take commands from you) while running "
                "unattended in the background."
            ),
            anchor="w",
            justify="left",
            wraplength=640,
            text_color=_UI_MUTED,
        ).pack(fill="x", pady=(0, 8))
        self._integrations_setup_btn = ctk.CTkButton(
            remote_card,
            text="Integrations Setup (Pushover & Telegram)",
            height=32,
            corner_radius=8,
            command=self._show_integrations_setup_guide,
        )
        self._integrations_setup_btn.pack(anchor="w")

    def _show_integrations_setup_guide(self) -> None:  # noqa: ANN001
        """Popup with the Pushover/Telegram setup guide (Settings tab)."""
        try:
            from dana.ui.settings import get_integrations_setup_text

            guide_text = get_integrations_setup_text()
        except Exception:  # noqa: BLE001
            guide_text = "Integrations setup guide is unavailable right now."

        try:
            win = ctk.CTkToplevel(self)
            win.title("Integrations Setup")
            win.geometry("640x520")
            box = ctk.CTkTextbox(
                win,
                wrap="word",
                font=ctk.CTkFont(family="Segoe UI", size=13),
                fg_color=_UI_CANVAS,
                corner_radius=8,
                border_width=0,
            )
            box.pack(fill="both", expand=True, padx=12, pady=12)
            box.insert("1.0", guide_text)
            box.configure(state="disabled")
            win.lift()
            win.focus_force()
        except Exception:  # noqa: BLE001
            pass

    def _build_appearance_card(self, tab) -> None:  # noqa: ANN001
        """Appearance / theme picker (Memory & Settings row 3)."""
        appear_card = self._make_card(
            tab, title="Appearance", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            appear_card, text="UI Theme", anchor="w", text_color=_UI_MUTED
        ).pack(fill="x", pady=(0, 4))
        try:
            from dana.ui.theme import THEME_NAMES, active_theme_name

            theme_values = list(THEME_NAMES)
            initial_theme = active_theme_name()
        except Exception:  # noqa: BLE001
            theme_values = ["Obsidian Mint", "Cyber Amber", "Ghost Light"]
            initial_theme = "Obsidian Mint"
        self._theme_var = ctk.StringVar(value=initial_theme)
        self._theme_menu = ctk.CTkOptionMenu(
            appear_card,
            values=theme_values,
            variable=self._theme_var,
            corner_radius=10,
            command=self._on_ui_theme_changed,
        )
        self._theme_menu.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            appear_card,
            text="Obsidian Mint · Cyber Amber · Ghost Light — switches instantly.",
            anchor="w",
            justify="left",
            wraplength=320,
            text_color=_UI_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", pady=(0, 4))
        self.apply_note = ctk.CTkLabel(
            appear_card,
            text="Audio: System Default (Auto)",
            text_color=_UI_MUTED,
            anchor="w",
            wraplength=320,
            justify="left",
            font=ctk.CTkFont(size=11),
        )
        self.apply_note.pack(fill="x", pady=(4, 0))

    def _sync_ui_theme_aliases(self) -> None:
        """Refresh module-level ``_UI_*`` aliases from ``dana.ui.theme``."""
        global _UI_CANVAS, _UI_CARD, _UI_CARD_BORDER, _UI_GHOST, _UI_MUTED
        global _UI_TEXT, _UI_ACCENT, _UI_ACCENT_HOVER, _UI_EMERALD, _UI_EMERALD_HOVER
        global _UI_ROSE, _UI_ROSE_HOVER, _UI_AMBER
        try:
            from dana.ui import theme as T

            _UI_CANVAS = T.BG
            _UI_CARD = T.CARD
            _UI_CARD_BORDER = T.BORDER
            _UI_GHOST = T.GHOST
            _UI_MUTED = T.MUTED
            _UI_TEXT = T.TEXT
            _UI_ACCENT = T.ACCENT
            _UI_ACCENT_HOVER = T.ACCENT_HOVER
            _UI_EMERALD = T.EMERALD
            _UI_EMERALD_HOVER = T.EMERALD_HOVER
            _UI_ROSE = T.ROSE
            _UI_ROSE_HOVER = T.ROSE_HOVER
            _UI_AMBER = T.AMBER
        except Exception:  # noqa: BLE001
            pass

    def _on_ui_theme_changed(self, choice: str) -> None:
        """Runtime theme switch — recolor dashboard tree."""
        try:
            from dana.ui.theme import set_theme

            set_theme(str(choice), root=self)
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: theme switch failed ({exc})")
            return
        self._sync_ui_theme_aliases()
        try:
            self.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass
        note = getattr(self, "apply_note", None)
        if note is not None:
            try:
                note.configure(text=f"Theme: {choice}")
            except Exception:  # noqa: BLE001
                pass
        log("UI", f"UI Theme → {choice}")

    def _set_update_status(self, text: str, *, color: str | None = None) -> None:
        lbl = self._update_status_lbl
        if lbl is None:
            return
        try:
            kwargs: dict[str, Any] = {"text": str(text)}
            if color:
                kwargs["text_color"] = color
            lbl.configure(**kwargs)
        except Exception:  # noqa: BLE001
            pass

    def _ota_mgr(self) -> Any:
        mgr = getattr(self, "_ota_manager", None)
        if mgr is not None:
            return mgr
        try:
            from dana.updater.manifest import get_ota_manager

            self._ota_manager = get_ota_manager()
            return self._ota_manager
        except Exception:  # noqa: BLE001
            return None

    def _on_ota_mode_changed(self, choice: str) -> None:
        mgr = self._ota_mgr()
        if mgr is None:
            return
        try:
            mgr.set_auto_update_mode("silent" if str(choice).lower() == "silent" else "manual")
        except Exception as exc:  # noqa: BLE001
            log("Updater", f"auto_update_mode change failed: {exc}")
        self._refresh_ota_ui()

    def _refresh_ota_ui(self) -> None:
        """Sync OTA status pill + Hot Apply visibility (headless-safe)."""
        mgr = self._ota_mgr()
        pill = getattr(self, "_ota_pill_lbl", None)
        slot_lbl = getattr(self, "_ota_slot_lbl", None)
        staging_lbl = getattr(self, "_ota_staging_lbl", None)
        btn = getattr(self, "_ota_hot_apply_btn", None)

        def _token(name: str, fallback: str) -> str:
            try:
                return str(getattr(_UI_THEME, name, fallback))
            except Exception:  # noqa: BLE001
                return fallback

        if mgr is None:
            if pill is not None:
                try:
                    pill.configure(
                        text="[UP TO DATE]",
                        text_color=_token("STATUS_IDLE", _UI_MUTED),
                    )
                except Exception:  # noqa: BLE001
                    pass
            return
        try:
            st = mgr.state()
        except Exception:  # noqa: BLE001
            return
        pill_text = st.status_pill()
        health = str(getattr(st, "staging_health", "idle") or "idle")
        if pill is not None:
            try:
                if health == "checking":
                    color = _token("STATUS_STAGING", _UI_AMBER)
                elif health == "failed":
                    color = _token("STATUS_FAILED", _UI_ROSE)
                elif health == "healthy":
                    color = _token("STATUS_HEALTHY", _UI_EMERALD)
                elif st.staged_version:
                    color = _token("STATUS_UPDATE_READY", _UI_EMERALD)
                elif st.update_available:
                    color = _token("STATUS_UPDATE_AVAILABLE", _UI_ACCENT)
                else:
                    color = _token("STATUS_IDLE", _UI_MUTED)
                pill.configure(text=pill_text, text_color=color)
            except Exception:  # noqa: BLE001
                pass
        if slot_lbl is not None:
            try:
                label = str(getattr(st, "active_slot_label", "") or "").strip()
                if not label:
                    label = f"Slot A (v{st.local_version.lstrip('vV')})"
                slot_lbl.configure(
                    text=f"Active: {label}",
                    text_color=_token("STATUS_SLOT_ACTIVE", _UI_EMERALD),
                )
            except Exception:  # noqa: BLE001
                pass
        if staging_lbl is not None:
            try:
                staging_colors = {
                    "idle": _token("STATUS_IDLE", _UI_MUTED),
                    "checking": _token("STATUS_STAGING", _UI_AMBER),
                    "healthy": _token("STATUS_HEALTHY", _UI_EMERALD),
                    "failed": _token("STATUS_FAILED", _UI_ROSE),
                }
                staging_lbl.configure(
                    text=f"Staging: {health}",
                    text_color=staging_colors.get(health, _UI_MUTED),
                )
            except Exception:  # noqa: BLE001
                pass
        if btn is None:
            return
        try:
            if st.staged_version:
                if not btn.winfo_ismapped():
                    check = self._update_check_btn
                    if check is not None and check.winfo_ismapped():
                        btn.pack(side="right", padx=(0, 8), before=check)
                    else:
                        btn.pack(side="right", padx=(0, 8))
            else:
                btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_ota_hot_apply(self) -> None:
        """Promote staged OTA via blue-green health gate + tool reload."""
        if self._update_busy:
            return
        mgr = self._ota_mgr()
        if mgr is None:
            self._set_update_status("OTA manager unavailable.", color="#F87171")
            return
        # Attach sidecar IPC for hot_restart when available.
        try:
            client = getattr(self, "_daemon_client", None)
            if client is not None and getattr(mgr, "_ipc_client", None) is None:
                mgr._ipc_client = client
        except Exception:  # noqa: BLE001
            pass
        self._update_busy = True
        try:
            if self._ota_hot_apply_btn is not None:
                self._ota_hot_apply_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Blue-green promote + health check…", color=_UI_AMBER)

        def _worker() -> None:
            err = ""
            version = ""
            active_label = ""
            try:
                result = mgr.hot_apply()
                version = str((result or {}).get("version") or "")
                active_label = str((result or {}).get("active_label") or "")
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log("Updater", f"hot_apply failed: {err}")

            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._ota_hot_apply_btn is not None:
                        self._ota_hot_apply_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                if err:
                    self._set_update_status(f"Hot Apply failed: {err}", color="#F87171")
                else:
                    suffix = f" — Active: {active_label}" if active_label else ""
                    self._set_update_status(
                        f"Hot Apply complete — v{version or '?'}{suffix}",
                        color="#66BB6A",
                    )
                self._refresh_ota_ui()

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="OTAHotApply", daemon=True).start()

    def _set_update_available(self, available: bool) -> None:
        btn = self._update_apply_btn
        if btn is None:
            return
        try:
            if available:
                if not btn.winfo_ismapped():
                    check = self._update_check_btn
                    if check is not None and check.winfo_ismapped():
                        btn.pack(side="right", padx=(0, 8), before=check)
                    else:
                        btn.pack(side="right", padx=(0, 8))
            else:
                btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_check_for_updates(self) -> None:
        """Stage 9.3 — background git fetch + rev compare (non-blocking UI)."""
        if self._update_busy:
            return
        self._update_busy = True
        try:
            if self._update_check_btn is not None:
                self._update_check_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Checking GitHub…", color=_UI_ACCENT)
        self._set_update_available(False)

        def _worker() -> None:
            available = False
            err = ""
            try:
                from dana.utils.updater import check_for_updates

                available = bool(check_for_updates())
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log("Updater", f"check_for_updates raised: {err}")

            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._update_check_btn is not None:
                        self._update_check_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                if err:
                    self._set_update_status(
                        f"Update check failed: {err}",
                        color="#F87171",
                    )
                    self._set_update_available(False)
                    return
                if available:
                    self._set_update_status(
                        "Update available — review then Update & Restart.",
                        color="#FB8C00",
                    )
                    self._set_update_available(True)
                else:
                    self._set_update_status(
                        "System is up to date.",
                        color="#66BB6A",
                    )
                    self._set_update_available(False)

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="UpdateCheck", daemon=True).start()

    def _on_apply_update_and_restart(self) -> None:
        """Stage 9.3 — git pull + pip install + relaunch (background)."""
        if self._update_busy:
            return
        self._update_busy = True
        try:
            if self._update_check_btn is not None:
                self._update_check_btn.configure(state="disabled")
            if self._update_apply_btn is not None:
                self._update_apply_btn.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
        self._set_update_status("Updating from GitHub…", color="#FB8C00")

        def _worker() -> None:
            from dana.utils.updater import apply_update_and_restart

            result = apply_update_and_restart(restart=True)

            # Only reached on failure (success calls sys.exit).
            def _ui() -> None:
                self._update_busy = False
                try:
                    if self._update_check_btn is not None:
                        self._update_check_btn.configure(state="normal")
                    if self._update_apply_btn is not None:
                        self._update_apply_btn.configure(state="normal")
                except Exception:  # noqa: BLE001
                    pass
                msg = result.message or "Update Failed."
                self._set_update_status(msg, color="#F87171")
                if result.stderr:
                    log("Updater", f"stderr:\n{result.stderr[:2000]}")

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                _ui()

        threading.Thread(target=_worker, name="UpdateApply", daemon=True).start()

    def _flash_engine_warning(self, message: str = "Please Engage Engine First.") -> None:
        """Brief Dashboard toast when a task is attempted in STANDBY."""
        lbl = self._engine_warn_lbl
        if lbl is None:
            return
        if self._engine_warn_job is not None:
            try:
                self.after_cancel(self._engine_warn_job)
            except Exception:  # noqa: BLE001
                pass
            self._engine_warn_job = None
        try:
            lbl.configure(text=str(message))
        except Exception:  # noqa: BLE001
            pass

        def _clear() -> None:
            self._engine_warn_job = None
            try:
                if self._engine_warn_lbl is not None:
                    self._engine_warn_lbl.configure(text="")
            except Exception:  # noqa: BLE001
                pass

        try:
            self._engine_warn_job = self.after(2800, _clear)
        except Exception:  # noqa: BLE001
            pass

    def _require_engine(self) -> bool:
        """Return True when engine is ACTIVE; else flash warning and return False."""
        if self.engine_active and state.engine_engaged.is_set():
            return True
        self._select_tab("Assistant & Tasks")
        self._flash_engine_warning("Please Engage Engine First.")
        return False

    def _set_vault_status(self, message: str) -> None:
        """Persistent banner (empty clears it) — see ``_vault_warn_lbl``."""
        lbl = self._vault_warn_lbl
        if lbl is None:
            return
        try:
            lbl.configure(text=str(message))
        except Exception:  # noqa: BLE001
            pass

    def _on_vault_unlock_request(self, reason: str) -> None:
        """shared_state listener callback — runs on the AgentLoop thread.

        Hands off to the Tk main thread via ``after()`` since CTk widgets may
        only be created/touched there. ``reason == ""`` means "unlocked" —
        clear the banner / close any open prompt instead of showing one.
        """
        try:
            if reason:
                self.after(0, lambda: self._show_vault_unlock_dialog(reason))
            else:
                self.after(0, self._clear_vault_unlock_prompt)
        except Exception:  # noqa: BLE001
            pass

    def _on_spec_approval_requested(self, payload: dict) -> None:
        """shared_state listener — runs on the AgentLoop thread; hand off to Tk."""
        try:
            if hasattr(self, "show_spec_approval"):
                self.after(0, lambda p=payload: self.show_spec_approval(p))
        except Exception:  # noqa: BLE001
            pass

    def _on_dictation_sessions_changed(self) -> None:
        """shared_state listener — runs on the AgentLoop thread; hand off to Tk."""
        try:
            if hasattr(self, "refresh_dictation_sessions"):
                self.after(0, self.refresh_dictation_sessions)
        except Exception:  # noqa: BLE001
            pass

    def _clear_vault_unlock_prompt(self) -> None:
        self._set_vault_status("")
        win = self._vault_unlock_win
        self._vault_unlock_win = None
        if win is not None:
            try:
                win.destroy()
            except Exception:  # noqa: BLE001
                pass

    def _show_vault_unlock_dialog(self, reason: str) -> None:
        """Modal passcode prompt; unblocks ``unlock_dana_memory()`` on submit/cancel.

        Runs on the Tk main thread (scheduled by ``_on_vault_unlock_request``).
        """
        self._set_vault_status("Vault Locked — Enter Passcode to Unlock")
        try:
            self.show_window()
        except Exception:  # noqa: BLE001
            pass

        existing = self._vault_unlock_win
        if existing is not None:
            try:
                existing.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._vault_unlock_win = None

        def _submit(password: Optional[str]) -> None:
            win = self._vault_unlock_win
            self._vault_unlock_win = None
            if win is not None:
                try:
                    win.destroy()
                except Exception:  # noqa: BLE001
                    pass
            if password:
                self._set_vault_status("Vault Locked — Unlocking...")
            supply_vault_unlock_response(password)

        try:
            win = ctk.CTkToplevel(self)
            self._vault_unlock_win = win
            win.title("Vault Locked")
            win.geometry("460x220")
            win.resizable(False, False)
            ctk.CTkLabel(
                win,
                text="Vault Locked — Enter Passcode to Unlock",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color="#EF4444",
            ).pack(padx=16, pady=(16, 4), anchor="w")
            ctk.CTkLabel(
                win,
                text=str(reason),
                wraplength=420,
                justify="left",
                anchor="w",
                text_color=_UI_MUTED,
            ).pack(padx=16, pady=(0, 10), anchor="w", fill="x")
            entry = ctk.CTkEntry(
                win,
                show="*",
                placeholder_text="Master Password / Recovery Key",
                width=420,
            )
            entry.pack(padx=16, pady=(0, 12))
            entry.focus_set()

            btn_row = ctk.CTkFrame(win, fg_color="transparent")
            btn_row.pack(padx=16, pady=(0, 12), fill="x")
            ctk.CTkButton(
                btn_row,
                text="Cancel",
                fg_color="transparent",
                border_width=1,
                command=lambda: _submit(None),
            ).pack(side="left")
            ctk.CTkButton(
                btn_row,
                text="Unlock",
                command=lambda: _submit(entry.get().strip() or None),
            ).pack(side="right")
            entry.bind("<Return>", lambda _e: _submit(entry.get().strip() or None))
            win.protocol("WM_DELETE_WINDOW", lambda: _submit(None))
            win.transient(self)
            win.grab_set()
            win.lift()
            win.focus_force()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: vault unlock dialog failed to open ({exc})")
            self._vault_unlock_win = None
            supply_vault_unlock_response(None)

    def _apply_behavior_mixer_payload(self) -> dict[str, int]:
        """Push current Behavior sliders into Blackboard persona_mixer."""
        values: dict[str, int] = {}
        for key, slider in self._behavior_sliders.items():
            try:
                values[key] = int(round(float(slider.get())))
            except Exception:  # noqa: BLE001
                continue
        if not values:
            try:
                from dana.memory.blackboard import get_persona_mixer

                return dict(get_persona_mixer())
            except Exception:  # noqa: BLE001
                return {}
        try:
            from dana.memory.blackboard import set_persona_mixer

            return dict(set_persona_mixer(values))
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: engage mixer apply failed ({exc})")
            return values

    def _refresh_engine_ui(self) -> None:
        """Sync Engage/Standby toggle chrome with ``engine_active``."""
        active = bool(self.engine_active)
        stopped = bool(getattr(self, "_engine_stopped", False)) and not active
        status = self._engine_status_lbl
        if status is not None:
            try:
                if active:
                    status.configure(
                        text="  ● ACTIVE | Local Engine  ",
                        text_color=_UI_EMERALD,
                        fg_color=_UI_GHOST,
                    )
                elif stopped:
                    status.configure(
                        text="  ● STOPPED | Local Engine  ",
                        text_color=_UI_ROSE,
                        fg_color=_UI_GHOST,
                    )
                else:
                    status.configure(
                        text="  ● STANDBY | Local Engine  ",
                        text_color=_UI_MUTED,
                        fg_color=_UI_GHOST,
                    )
            except Exception:  # noqa: BLE001
                pass
        header_status = getattr(self, "_header_status_lbl", None)
        if header_status is not None:
            try:
                if active:
                    header_status.configure(text="• ACTIVE", text_color=_UI_EMERALD)
                elif stopped:
                    header_status.configure(text="• STOPPED", text_color=_UI_ROSE)
                else:
                    header_status.configure(text="• STANDBY", text_color=_UI_MUTED)
            except Exception:  # noqa: BLE001
                pass
        toggle = getattr(self, "_engage_toggle_btn", None) or self._engage_btn
        try:
            if toggle is not None:
                if active:
                    toggle.configure(
                        text="Engaged",
                        state="normal",
                        fg_color=_UI_EMERALD,
                        hover_color="#059669",
                        text_color="#ECFDF5",
                    )
                else:
                    toggle.configure(
                        text="Standby",
                        state="normal",
                        fg_color=_UI_GHOST,
                        hover_color="#475569",
                        text_color=_UI_TEXT,
                    )
        except Exception:  # noqa: BLE001
            pass

    def toggle_engine_engage(self) -> None:
        """Single HUD control: Engaged ↔ Standby."""
        if bool(self.engine_active) and state.engine_engaged.is_set():
            self.standby_engine()
        else:
            self.engage_engine()

    def engage_engine(self) -> None:
        """Stage 8.9.7 — arm engine, apply mixer, lock Behavior sliders."""
        applied = self._apply_behavior_mixer_payload()
        self.engine_active = True
        self._engine_stopped = False
        state.engine_engaged.set()
        self._set_behavior_controls_locked(True)
        self._refresh_engine_ui()
        try:
            if self._engine_warn_lbl is not None:
                self._engine_warn_lbl.configure(text="")
        except Exception:  # noqa: BLE001
            pass
        log(
            "UI",
            f"ENGAGE engine — behavior locked mixer={list(applied.keys())}",
        )
        try:
            self.log_transcript(
                "Dana",
                "Engine ENGAGED — Behavior variables locked. Ready for chat.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass
        # Lazy warm: LangGraph import only — Florence/YOLO stay JIT.
        try:
            threading.Thread(
                target=_warm_heavy_runtime_assets,
                name="HeavyWarm",
                daemon=True,
            ).start()
        except Exception:  # noqa: BLE001
            pass

    def standby_engine(self) -> None:
        """Stage 8.9.7 — soft pause loops, unlock Behavior (GUI stays alive)."""
        # Drop dictation latch if hot — mixer must be editable in STANDBY.
        if self._dictation_active:
            try:
                from dana.management.dictation import toggle_dictation_mode

                toggle_dictation_mode(False)
            except Exception:  # noqa: BLE001
                pass
            self._dictation_active = False
            try:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#34344A",
                    border_color=_UI_CARD_BORDER,
                    text_color="#F87171",
                )
            except Exception:  # noqa: BLE001
                pass
        self.engine_active = False
        state.engine_engaged.clear()
        # Soft-pause: clear pending mic latch; do NOT touch stop_event / STOP DANA.
        try:
            is_recording.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_tts_audio_state("standby_engine", ui_state="idle")
        except Exception:  # noqa: BLE001
            pass
        self._set_behavior_controls_locked(False)
        self._refresh_engine_ui()
        log("UI", "STANDBY engine — behavior unlocked (soft pause)")
        try:
            self.log_transcript(
                "Dana",
                "Engine STANDBY — Behavior variables unlocked.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _dashboard_start_chat(self) -> None:
        """Quick action — focus Assistant silent chat entry (no transcript spam)."""
        if not self._require_engine():
            return
        self._select_tab("Assistant & Tasks")
        try:
            self.lift()
            self.focus_force()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.chat_entry is not None:
                self.chat_entry.focus_set()
        except Exception:  # noqa: BLE001
            pass

    def _dashboard_trigger_dictation(self) -> None:
        """Quick action — open Memory & Settings and arm the latch if cold."""
        if not self._require_engine():
            return
        self._select_tab("Memory & Settings")
        if not self._dictation_active:
            try:
                self._toggle_dictation_mode()
            except Exception:  # noqa: BLE001
                pass

    def _dashboard_open_trace(self) -> None:
        """Open a dedicated Diagnostics / Live Trace overlay (not Memory & Settings)."""
        try:
            existing = getattr(self, "_diag_overlay", None)
            if existing is not None and bool(existing.winfo_exists()):
                try:
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    return
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            from dana.ui.trace_window import LiveTraceWindow

            win = LiveTraceWindow(self)
            self._diag_overlay = win
            try:
                win.title("Dānā — Diagnostics / Live Trace")
            except Exception:  # noqa: BLE001
                pass
            try:
                win.lift()
                win.focus_force()
            except Exception:  # noqa: BLE001
                pass
            return
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: Diagnostics overlay unavailable ({exc})")
        # Last resort: expand embedded Developer Diagnostics without stealing
        # the Memory & Settings tab identity from the header segment.
        try:
            self._select_tab("Memory & Settings")
            if not bool(getattr(self, "_diag_expanded", False)):
                self._toggle_developer_diagnostics()
        except Exception:  # noqa: BLE001
            pass

    def _transcript_tk(self):
        """Return the raw ``tk.Text`` inside CTkTextbox (for tag_configure)."""
        box = getattr(self, "transcript_box", None)
        if box is None:
            return None
        return getattr(box, "_textbox", None) or getattr(box, "textbox", None)

    def _init_persona_transcript_tags(self) -> None:
        """Stage 8.5.2 — register persona styles via Tk ``tag_configure``."""
        tk_text = self._transcript_tk()
        if tk_text is None:
            return
        try:
            # Readable slate palette — no electric cyan / neon green.
            tk_text.tag_configure(
                "jason",
                foreground="#C084FC",
                font=("Segoe UI", 14, "bold"),
            )
            tk_text.tag_configure(
                "llama",
                foreground=_UI_TEXT,
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "deepseek",
                foreground=_UI_ROSE,
                font=("Courier New", 10),
            )
            tk_text.tag_configure(
                "vision",
                foreground=_UI_EMERALD,
                font=("Segoe UI", 14),
            )
            tk_text.tag_configure(
                "typist",
                foreground=_UI_AMBER,
                font=("Segoe UI", 14, "italic"),
            )
            # Stage 8.10 — silent Dashboard text (distinct from Whisper).
            tk_text.tag_configure(
                "user_text",
                foreground=_UI_ACCENT,
                font=("Segoe UI", 14, "italic"),
            )
            # Theme-safe default when no agent_id is provided.
            tk_text.tag_configure(
                "default",
                foreground=_UI_TEXT,
                font=("Segoe UI", 14),
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: persona tag_configure failed ({exc})")

    @staticmethod
    def _persona_tag_for_agent(
        agent_id: str | None,
        *,
        speaker: str = "",
    ) -> str:
        """Map agent_id / speaker heuristics → transcript tag name."""
        key = (agent_id or "").strip().lower()
        aliases = {
            "jason": "jason",
            "jason_cto": "jason",
            "cto": "jason",
            "llama": "llama",
            "llama3": "llama",
            "broker": "llama",
            "chat": "llama",
            "chat_node": "llama",
            "receptionist": "llama",
            "dana": "llama",
            "dana": "llama",
            "deepseek": "deepseek",
            "moa": "deepseek",
            "moa_reasoner": "deepseek",
            "reasoner": "deepseek",
            "vision": "vision",
            "yolo": "vision",
            "florence": "vision",
            "ocr": "vision",
            "vision_agent": "vision",
            "typist": "typist",
            "ghost": "typist",
            "ghost_typist": "typist",
            "keystroke": "typist",
            "nav": "typist",
            "navigation": "typist",
        }
        if key in aliases:
            return aliases[key]
        sp = (speaker or "").strip().lower()
        if "jason" in sp:
            return "jason"
        if "deepseek" in sp or "moa" in sp:
            return "deepseek"
        if any(x in sp for x in ("vision", "yolo", "florence", "ocr")):
            return "vision"
        if any(x in sp for x in ("typist", "ghost", "keystroke", "nav")):
            return "typist"
        if sp.startswith(("dana", "dana")) or "ollama" in sp or "llama" in sp:
            return "llama"
        if sp.startswith("user") and "text" in sp:
            return "user_text"
        if sp.startswith("user"):
            return "default"
        return "default"

    def _build_dictation_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.5 — pill toggle dictation + recent sessions card."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        control_card = self._make_card(
            tab, title="Dictation Latch", padx=4, pady=(0, 8), expand=False
        )
        ctk.CTkLabel(
            control_card,
            text=(
                "When ON, every utterance is logged with Florence OCR visual state. "
                "You can also say “dictate …” while OFF."
            ),
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=360,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        controls = ctk.CTkFrame(control_card, fg_color="transparent")
        controls.pack(fill="x", pady=(0, 4))
        # Unified pill toggle — OFF: dim gray + rose pill; ON: accent glow + DICTATING.
        self.dictation_btn = ctk.CTkButton(
            controls,
            text="  ●  OFF  ",
            width=140,
            height=36,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_ROSE,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_dictation_mode,
        )
        self.dictation_btn.pack(side="left")
        # Compatibility stub — pill on dictation_btn is the sole latch chrome
        # (Stage 8.9.4 removed the duplicate Status: label beside the toggle).
        self.dictation_status = ctk.CTkLabel(
            controls,
            text="● OFF",
            anchor="w",
            text_color=_UI_ROSE,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=_UI_CANVAS,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        actions = ctk.CTkFrame(control_card, fg_color="transparent")
        actions.pack(fill="x", pady=(8, 0))
        ctk.CTkButton(
            actions,
            text="Refresh status",
            width=110,
            height=30,
            corner_radius=999,
            fg_color=_UI_GHOST,
            hover_color="#475569",
            border_width=1,
            border_color=_UI_CARD_BORDER,
            text_color=_UI_TEXT,
            font=ctk.CTkFont(size=11),
            command=self.refresh_dictation_sessions,
        ).pack(side="right")
        # Recent Sessions telemetry lives in the Diagnostics overlay only.
        # Keep attribute stub so refresh_dictation_sessions stays safe.
        if getattr(self, "dictation_list", None) is None:
            self.dictation_list = None
        # Sync latch from Blackboard (may already be on).
        try:
            from dana.memory.blackboard import is_dictation_mode

            self._set_dictation_ui(bool(is_dictation_mode()))
        except Exception:  # noqa: BLE001
            self._set_dictation_ui(False)

    def _build_behavior_tab(self, tab) -> None:  # noqa: ANN001
        """Stage 8.5 — Behavior Mixer 2×2 grid → Blackboard persona_mixer."""
        try:
            tab.configure(fg_color=_UI_CANVAS)
        except Exception:  # noqa: BLE001
            pass

        card = self._make_card(tab, title="Behavior Mixer")
        ctk.CTkLabel(
            card,
            text="Autonomy · Verbosity · Creativity · Tech Depth",
            anchor="w",
            text_color=_UI_MUTED,
            wraplength=640,
            justify="left",
        ).pack(fill="x", pady=(0, 4))
        # Neutral hint (no jarring lock warning — overlay handles engage lock).
        self._behavior_lock_hint = ctk.CTkLabel(
            card,
            text="Adjust traits while engine is on standby, then Apply & Save",
            anchor="w",
            text_color="#888888",
            font=ctk.CTkFont(size=12),
        )
        self._behavior_lock_hint.pack(fill="x", pady=(0, 8))

        specs = (
            ("Autonomy", "autonomy"),
            ("Verbosity", "verbosity"),
            ("Creativity", "creativity"),
            ("Tech Depth", "technical_depth"),
        )
        try:
            from dana.memory.blackboard import (
                PERSONA_MIXER_DEFAULTS,
                get_persona_mixer,
            )

            state = get_persona_mixer()
        except Exception:  # noqa: BLE001
            PERSONA_MIXER_DEFAULTS = {
                "autonomy": 40,
                "verbosity": 50,
                "creativity": 50,
                "technical_depth": 80,
            }
            state = dict(PERSONA_MIXER_DEFAULTS)

        mixer_host = ctk.CTkFrame(card, fg_color="transparent")
        mixer_host.pack(fill="both", expand=True, pady=(0, 4))
        self._behavior_mixer_host = mixer_host
        mixer_host.grid_columnconfigure(0, weight=1)
        mixer_host.grid_columnconfigure(1, weight=1)
        mixer_host.grid_rowconfigure(0, weight=1)
        mixer_host.grid_rowconfigure(1, weight=1)

        self._static_behavior_widgets = []
        for idx, (label, key) in enumerate(specs):
            r, c = divmod(idx, 2)
            cell = ctk.CTkFrame(mixer_host, fg_color="transparent")
            cell.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
            row = ctk.CTkFrame(cell, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row, text=label, anchor="w", text_color="#E5E7EB"
            ).pack(side="left")
            val = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            val_lbl = ctk.CTkLabel(
                row, text=str(val), width=36, text_color="#F9FAFB"
            )
            val_lbl.pack(side="right")
            self._behavior_labels[key] = val_lbl
            slider = ctk.CTkSlider(
                cell,
                from_=0,
                to=100,
                number_of_steps=100,
                command=lambda v, k=key: self._on_behavior_drag(k, v),
            )
            slider.set(float(val))
            slider.pack(fill="x", pady=(6, 0))
            slider.bind(
                "<ButtonRelease-1>",
                lambda _e, k=key: self._commit_behavior(k, force=True),
            )
            self._behavior_sliders[key] = slider
            self._static_behavior_widgets.append(slider)

        self._behavior_reload_btn = ctk.CTkButton(
            card,
            text="Apply & Save Traits",
            width=160,
            height=30,
            corner_radius=8,
            fg_color=_UI_EMERALD,
            hover_color=_UI_EMERALD_HOVER,
            text_color="#ECFDF5",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.update_behavior_traits,
        )
        self._behavior_reload_btn.pack(pady=(8, 4), anchor="w")
        self._static_behavior_widgets.append(self._behavior_reload_btn)

        # Dim overlay when engine is engaged (no red warning copy).
        self._behavior_lock_overlay = ctk.CTkFrame(
            mixer_host,
            fg_color=("#0a0e17", "#0a0e17"),
            corner_radius=12,
            border_width=0,
        )
        try:
            self._behavior_lock_overlay.configure(cursor="arrow")
        except Exception:  # noqa: BLE001
            pass
        overlay_lbl = ctk.CTkLabel(
            self._behavior_lock_overlay,
            text="Mixer locked",
            font=ctk.CTkFont(size=12),
            text_color=_UI_MUTED,
        )
        overlay_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _set_behavior_controls_locked(self, locked: bool) -> None:
        """Grey out Behavior Mixer + place dim overlay when engine is hot."""
        self._behavior_locked = bool(locked)
        state = "disabled" if self._behavior_locked else "normal"
        for widget in list(self._static_behavior_widgets):
            try:
                widget.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        # Hard-disable sliders again (some CTk builds ignore batch configure).
        for slider in list(self._behavior_sliders.values()):
            try:
                slider.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        overlay = getattr(self, "_behavior_lock_overlay", None)
        if overlay is not None:
            try:
                if self._behavior_locked:
                    overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
                    try:
                        overlay.lift()
                    except Exception:  # noqa: BLE001
                        pass
                    # Soft dim — CTk has no true alpha; use near-canvas fill.
                    overlay.configure(fg_color=("#131b2e", "#131b2e"))
                else:
                    overlay.place_forget()
            except Exception:  # noqa: BLE001
                pass
        hint = self._behavior_lock_hint
        if hint is not None:
            try:
                if self._behavior_locked:
                    hint.configure(
                        text="Mixer locked — standby / idle required to edit traits",
                        text_color=_UI_AMBER,
                    )
                else:
                    hint.configure(
                        text="Adjust traits while engine is on standby, then Apply & Save",
                        text_color="#888888",
                    )
            except Exception:  # noqa: BLE001
                pass

    def _sync_behavior_lock_from_engine_state(self) -> None:
        """Lock mixer when Engaged/Active or while the pipeline is processing."""
        engaged = bool(getattr(self, "engine_active", False))
        processing = bool(getattr(self, "_vad_processing", False))
        self._set_behavior_controls_locked(engaged or processing)

    def update_behavior_traits(self) -> None:
        """Apply & Save Traits — write current slider values to Blackboard."""
        if self._behavior_locked:
            hint = self._behavior_lock_hint
            if hint is not None:
                try:
                    hint.configure(
                        text="Cannot save — put engine in Standby first",
                        text_color=_UI_ROSE,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return
        applied = self._apply_behavior_mixer_payload()
        try:
            self._reload_behavior_sliders()
        except Exception:  # noqa: BLE001
            pass
        hint = self._behavior_lock_hint
        if hint is not None:
            try:
                keys = ", ".join(f"{k}={v}" for k, v in sorted(applied.items()))
                hint.configure(
                    text=f"Traits saved — {keys}" if keys else "Traits saved",
                    text_color=_UI_EMERALD,
                )
            except Exception:  # noqa: BLE001
                pass
        log("UI", f"Apply & Save Traits → {applied}")

    def _set_dictation_ui(self, active: bool) -> None:
        self._dictation_active = bool(active)
        try:
            if self._dictation_active:
                self.dictation_btn.configure(
                    text="  ●  DICTATING  ",
                    fg_color=_UI_EMERALD,
                    hover_color="#059669",
                    border_color=_UI_EMERALD,
                    text_color="#ECFDF5",
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● DICTATING",
                        text_color=_UI_EMERALD,
                    )
            else:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#475569",
                    border_color=_UI_CARD_BORDER,
                    text_color=_UI_ROSE,
                )
                if self.dictation_status is not None:
                    self.dictation_status.configure(
                        text="● OFF",
                        text_color=_UI_ROSE,
                    )
        except Exception:  # noqa: BLE001
            pass
        # Stage 8.9.7 — Behavior lock follows engine ignition (not dictation alone).
        self._sync_behavior_lock_from_engine_state()
        # Keep top status bar glowing badge in sync with latch.
        try:
            self._set_mode_indicator(self._header_mode)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _toggle_dictation_mode(self) -> None:
        """Non-blocking GUI latch → Blackboard dictation_mode + mixer lock."""
        # Turning ON requires ENGAGE; turning OFF is always allowed.
        if (not self._dictation_active) and (not self._require_engine()):
            return
        try:
            from dana.management.dictation import toggle_dictation_mode

            active = toggle_dictation_mode(not self._dictation_active)
            self._set_dictation_ui(active)
            log(
                "UI",
                f"Dictation mode -> {'on' if active else 'off'} "
                f"(behavior_locked={self._behavior_locked})",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: dictation toggle failed ({exc})")

    def refresh_dictation_sessions(self) -> None:
        """Query Blackboard dictation_sessions on the Tk thread."""
        if not self.winfo_exists():
            return
        box = getattr(self, "dictation_list", None)
        if box is None:
            return
        try:
            from dana.memory.blackboard import list_dictation_sessions

            rows = list_dictation_sessions(limit=40)
        except Exception as exc:  # noqa: BLE001
            rows = []
            log("UI", f"WARNING: list_dictation_sessions failed ({exc})")
        lines: list[str] = []
        if not rows:
            lines.append("(no dictation sessions yet)\n")
        else:
            for r in rows:
                ts = str(r.get("timestamp") or "")[:19].replace("T", " ")
                cmd = str(r.get("command_text") or "").replace("\n", " ")
                if len(cmd) > 90:
                    cmd = cmd[:87] + "..."
                vis_n = len(str(r.get("visual_state_reference") or ""))
                sid = str(r.get("session_id") or "")[:8]
                st = str(r.get("status") or "recorded")
                lines.append(f"[{ts}] {sid}  {st}\n  {cmd}\n  visual_chars={vis_n}\n\n")
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", "".join(lines))
            box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _refresh_system_log(self) -> None:
        box = getattr(self, "_system_log_box", None)
        if box is None:
            return
        text = "(log unavailable)\n"
        try:
            from dana.logging import RUNTIME_LOG_PATH

            path = RUNTIME_LOG_PATH
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                text = "".join(lines[-40:]) or "(log empty)\n"
            else:
                legacy = os.path.join(os.path.dirname(path), "dana_runtime.log")
                if os.path.isfile(legacy):
                    with open(legacy, "r", encoding="utf-8", errors="replace") as fh:
                        lines = fh.readlines()
                    text = "".join(lines[-40:]) or "(log empty)\n"
                else:
                    text = f"(no log yet at {path})\n"
        except Exception as exc:  # noqa: BLE001
            text = f"Could not read log: {exc}\n"
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text)
            box.configure(state="disabled")
            box.see("end")
        except Exception:  # noqa: BLE001
            pass

    def _on_behavior_drag(self, trait: str, value: float) -> None:
        if self._behavior_locked:
            return
        n = int(round(float(value)))
        lbl = self._behavior_labels.get(trait)
        if lbl is not None:
            try:
                lbl.configure(text=str(n))
            except Exception:  # noqa: BLE001
                pass
        now = time.monotonic()
        if now - float(self._behavior_last_write.get(trait, 0.0)) >= 0.15:
            self._commit_behavior(trait, force=False)

    def _commit_behavior(self, trait: str, *, force: bool) -> None:
        if self._behavior_locked:
            return
        slider = self._behavior_sliders.get(trait)
        if slider is None:
            return
        n = int(round(float(slider.get())))
        now = time.monotonic()
        if not force and (now - float(self._behavior_last_write.get(trait, 0.0))) < 0.15:
            return
        try:
            from dana.memory.blackboard import set_persona_trait

            set_persona_trait(trait, n)
            self._behavior_last_write[trait] = now
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: behavior slider write failed ({exc})")

    def _reload_behavior_sliders(self) -> None:
        if not self.winfo_exists():
            return
        if self._behavior_locked:
            return
        try:
            from dana.memory.blackboard import (
                PERSONA_MIXER_DEFAULTS,
                get_persona_mixer,
            )

            state = get_persona_mixer()
        except Exception:  # noqa: BLE001
            return
        for key, slider in self._behavior_sliders.items():
            n = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            try:
                slider.set(float(n))
                lbl = self._behavior_labels.get(key)
                if lbl is not None:
                    lbl.configure(text=str(n))
            except Exception:  # noqa: BLE001
                pass

    def _telemetry_had_activity(self) -> bool:
        """Cheap, side-effect-free peek: is there new telemetry to render?

        Called synchronously from ``_master_telemetry_tick`` — main thread
        only. Touches only thread-safe sources (``queue.Queue``, the
        lock-protected ``AsyncRingBuffer``, and ``MonitorBus.pending()``)
        without draining anything itself; the dispatch below does the real
        draining. Feeds ``AdaptivePoller.note_activity()`` so the shared
        heartbeat speeds back up under load and rests while idle.

        NOTE: this does *not* run on ``AdaptivePoller``'s background thread.
        An earlier version of this dispatcher tried exactly that (per the
        textbook "marshal via self.after(0, ...)" pattern) and it does not
        work: registering a Tk callback is itself a Tcl/Tk call, and
        CPython's Tkinter (3.12+) raises ``RuntimeError: main thread is not
        in main loop`` — or simply stalls the poller thread — the instant a
        non-main thread calls ``widget.after()``, ``self.after(0, ...)``
        included. See ``AdaptivePoller``'s docstring. Everything telemetry
        related in ``DanaGUI`` therefore stays on the Tk main thread; only
        the backoff *interval math* is delegated to ``AdaptivePoller``.
        """
        had_activity = False
        try:
            had_activity = not gui_telemetry_queue.empty()
        except Exception:  # noqa: BLE001
            pass
        if not had_activity:
            try:
                had_activity = len(self._telemetry_buffer.snapshot()) != self._neural_rendered
            except Exception:  # noqa: BLE001
                pass
        if not had_activity:
            try:
                from dana.graph.monitor_bus import get_monitor_bus

                bus = get_monitor_bus(create=False)
                had_activity = bool(bus is not None and bus.pending() > 0)
            except Exception:  # noqa: BLE001
                pass
        return had_activity

    def _master_telemetry_tick(self) -> None:
        """Single Tk-main-thread dispatcher for every telemetry consumer.

        A conventional self-rescheduling ``self.after()`` chain — like every
        other poller in this file — except the *delay* it re-arms itself
        with adapts each tick via ``self._adaptive_poller.note_activity()``
        (50ms while busy, backing off toward 500ms while idle) instead of a
        fixed interval. Replaces what used to be five independent
        ``self.after()`` loops (``LiveTracePanel`` 80ms, ``process_telemetry``
        100ms, ``_poll_state_changes`` 100ms, ``DagMonitorView`` 250ms,
        ``TaskTrackerView`` 400ms) with one shared heartbeat. Each consumer
        still only fires at its own cadence — tracked via monotonic elapsed
        time in ``self._telemetry_last`` — so a backed-off (idle) heartbeat
        never causes drift or double-fires; a slow heartbeat just means a
        consumer's next check waits longer.
        """
        if not self.winfo_exists():
            return
        try:
            had_activity = self._telemetry_had_activity()
        except Exception:  # noqa: BLE001
            had_activity = False
        now = time.monotonic()
        last = self._telemetry_last
        cadences = _TELEMETRY_CADENCES_S

        live_trace = getattr(self, "live_trace", None)
        if live_trace is not None and now - last.get("live_trace", 0.0) >= cadences["live_trace"]:
            last["live_trace"] = now
            try:
                live_trace.tick()
            except Exception:  # noqa: BLE001
                pass

        if now - last.get("process_telemetry", 0.0) >= cadences["process_telemetry"]:
            last["process_telemetry"] = now
            try:
                self.process_telemetry()
            except Exception:  # noqa: BLE001
                pass

        if now - last.get("state_changes", 0.0) >= cadences["state_changes"]:
            last["state_changes"] = now
            try:
                self._poll_state_changes()
            except Exception:  # noqa: BLE001
                pass

        dag_view = getattr(self, "dag_monitor_view", None)
        if dag_view is not None and now - last.get("dag_monitor", 0.0) >= cadences["dag_monitor"]:
            last["dag_monitor"] = now
            try:
                dag_view.refresh()
            except Exception:  # noqa: BLE001
                pass

        task_view = getattr(self, "task_tracker_view", None)
        if task_view is not None and now - last.get("task_tracker", 0.0) >= cadences["task_tracker"]:
            last["task_tracker"] = now
            try:
                task_view.tick()
            except Exception:  # noqa: BLE001
                pass

        try:
            next_s = self._adaptive_poller.note_activity(had_activity)
        except Exception:  # noqa: BLE001
            next_s = self._adaptive_poller.t_min
        try:
            self.after(max(1, int(next_s * 1000)), self._master_telemetry_tick)
        except Exception:  # noqa: BLE001
            pass

    def process_telemetry(self) -> None:
        """Drain legacy ``gui_telemetry_queue`` on the Tk main thread.

        Called from ``_master_telemetry_tick`` (see its cadence in
        ``_TELEMETRY_CADENCES_S["process_telemetry"]``) — does not
        reschedule itself; the master dispatcher owns cadence for all five
        telemetry consumers now. Keeps header mode / fallback TraceCells in
        sync, and mirrors every event into the Neural Stream ring buffer for
        the Unified Canvas.
        """
        if not self.winfo_exists():
            return
        try:
            while True:
                try:
                    event = gui_telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(event, dict):
                    continue
                stage = str(event.get("stage") or "stage")
                status = str(event.get("status") or "active")
                message = str(event.get("message") or stage)
                mode = event.get("mode")
                if mode:
                    self._set_mode_indicator(str(mode))
                try:
                    self._telemetry_emitter.emit(
                        stage, {"message": f"[{stage}] {message}", "status": status}
                    )
                except Exception:  # noqa: BLE001
                    pass
                # When LiveTracePanel is mounted, skip duplicate TraceCell rows.
                if getattr(self, "live_trace", None) is not None:
                    continue
                accent = self._mode_accent(
                    str(mode) if mode else self._header_mode
                )
                cell = self._trace_cells.get(stage)
                if cell is None:
                    cell = TraceCell(
                        self.trace_scroll,
                        stage=stage,
                        message=message,
                        status=status,
                    )
                    cell.pack(fill="x", padx=4, pady=4)
                    self._trace_cells[stage] = cell
                cell.update_status(status, message=message, accent=accent)
                try:
                    self.trace_scroll._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            self._render_neural_stream()
        except Exception:  # noqa: BLE001
            pass

    def _render_neural_stream(self) -> None:
        """Flush new ``AsyncRingBuffer`` events into the Neural Stream Text widget.

        Applies keyword color tags and a tail-drop limiter (cap 500 lines) so
        a busy session can never grow the widget large enough to lag Tk.
        """
        text = getattr(self, "_neural_stream_text", None)
        buffer = getattr(self, "_telemetry_buffer", None)
        if text is None or buffer is None or not self.winfo_exists():
            return
        events = buffer.snapshot()
        rendered = getattr(self, "_neural_rendered", 0)
        new_events = events[rendered:]
        if not new_events:
            return
        self._neural_rendered = len(events)
        try:
            text.configure(state="normal")
            for event in new_events:
                payload = event.get("payload") or {} if isinstance(event, dict) else {}
                message = str(payload.get("message") or event.get("type") or "").strip()
                if not message:
                    continue
                upper = message.upper()
                if "EXECUTION ERROR" in upper or str(payload.get("status") or "") == "error":
                    tag = "error"
                elif "THOUGHT:" in upper:
                    tag = "thought"
                elif "TOOL" in upper:
                    tag = "tool"
                else:
                    tag = None
                line = f"{message}\n"
                if tag:
                    text.insert("end", line, (tag,))
                else:
                    text.insert("end", line)
            # Tail-Drop Limiter — keep at most 500 lines so Tk never lags.
            line_count = int(text.index("end-1c").split(".")[0])
            if line_count > 500:
                text.delete("1.0", f"{line_count - 500}.0")
            text.see("end")
            text.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _poll_state_changes(self) -> None:
        """Drain STATE_CHANGE bus → VAD mic + System Status line.

        Called from ``_master_telemetry_tick`` — does not reschedule itself.
        """
        if not self.winfo_exists():
            return
        try:
            from dana.ui.status_bus import drain_state_changes

            # Drain until empty (capped) so bursts between ticks never stall.
            events: list = []
            for _ in range(8):
                batch = drain_state_changes(max_items=64)
                if not batch:
                    break
                events.extend(batch)
                if len(events) >= 256:
                    break
            if events:
                # Latest wins for widgets; full drain guarantees no dropped tip.
                self._apply_state_change(events[-1])
        except Exception:  # noqa: BLE001
            pass

    def _apply_state_change(self, event: dict) -> None:
        """Update VAD mic pip + System Status line from a STATE_CHANGE payload."""
        try:
            from dana.ui.status_bus import format_system_status_line
        except Exception:  # noqa: BLE001
            return
        status = str(event.get("status") or "idle").strip().lower()
        tool = str(event.get("tool") or "")
        message = str(event.get("message") or "")
        self._vad_listening = status == "listening"
        self._vad_processing = status in {"processing", "routing", "executing"}
        # Instantly lock / unlock Behavior Mixer with pipeline activity.
        try:
            self._sync_behavior_lock_from_engine_state()
        except Exception:  # noqa: BLE001
            pass
        line = format_system_status_line(
            status, tool=tool, message=message
        )
        mic_text = "● Idle"
        mic_color = _UI_MUTED
        if status == "listening":
            mic_text = "● Listening"
            mic_color = _UI_EMERALD
        elif status == "processing":
            mic_text = "● Processing"
            mic_color = _UI_AMBER
        elif status == "routing":
            mic_text = "● Processing"
            mic_color = _UI_AMBER
        elif status == "executing":
            mic_text = "● Processing"
            mic_color = _UI_ACCENT
        lbl = getattr(self, "_system_status_lbl", None)
        if lbl is not None:
            try:
                color = _UI_MUTED
                if status == "routing":
                    color = _UI_AMBER
                elif status == "executing":
                    color = _UI_ACCENT
                elif status == "listening":
                    color = _UI_EMERALD
                elif status == "processing":
                    color = _UI_AMBER
                elif status == "idle":
                    color = _UI_MUTED
                if tool == "proactive_briefing":
                    color = _UI_AMBER
                lbl.configure(text=line or "Idle", text_color=color)
            except Exception:  # noqa: BLE001
                pass
        if tool == "proactive_briefing":
            hdr = getattr(self, "_header_status_lbl", None)
            if hdr is not None:
                try:
                    hdr.configure(text="• BRIEFING", text_color=_UI_AMBER)
                except Exception:  # noqa: BLE001
                    pass
            badge = getattr(self, "daemon_badge", None)
            if badge is not None and message:
                try:
                    badge.configure(text="● UPDATE")
                except Exception:  # noqa: BLE001
                    pass
        mic = getattr(self, "_vad_mic_lbl", None)
        if mic is not None:
            try:
                mic.configure(text=mic_text, text_color=mic_color)
            except Exception:  # noqa: BLE001
                pass

    def _pulse_active_cells(self) -> None:
        if not self.winfo_exists():
            return
        self._pulse_on = not self._pulse_on
        accent = self._mode_accent()
        dim = "#4B5563"
        for cell in self._trace_cells.values():
            if cell.current_status != "active":
                continue
            try:
                cell.configure(
                    border_color=accent if self._pulse_on else dim
                )
            except Exception:  # noqa: BLE001
                pass
        # Pulsating VAD mic pip while listening (theme emerald ↔ muted).
        mic = getattr(self, "_vad_mic_lbl", None)
        if mic is not None and getattr(self, "_vad_listening", False):
            self._vad_pulse_on = not getattr(self, "_vad_pulse_on", False)
            try:
                mic.configure(
                    text="● Listening",
                    text_color=_UI_EMERALD
                    if self._vad_pulse_on
                    else _UI_ACCENT,
                )
            except Exception:  # noqa: BLE001
                pass
        elif mic is not None and getattr(self, "_vad_processing", False):
            self._vad_pulse_on = not getattr(self, "_vad_pulse_on", False)
            try:
                mic.configure(
                    text="● Processing",
                    text_color=_UI_AMBER if self._vad_pulse_on else _UI_ACCENT,
                )
            except Exception:  # noqa: BLE001
                pass
        try:
            self.after(500, self._pulse_active_cells)
        except Exception:  # noqa: BLE001
            pass

    def _on_transcript_event(
        self, speaker: str, text: str, agent_id: str | None
    ) -> None:
        """dana.core.shared_state transcript-listener adapter for log_transcript."""
        try:
            self.log_transcript(speaker, text, agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: live transcript update failed ({exc})")

    def log_transcript(
        self,
        speaker: str,
        text: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Append a speaker line to Conversation (bubbles + mirror textbox)."""
        line = f"[{speaker}] {text}\n\n"
        tag = self._persona_tag_for_agent(agent_id, speaker=speaker)
        try:
            self._telemetry_emitter.emit("transcript", {"message": f"[{speaker}] {text}"})
        except Exception:  # noqa: BLE001
            pass

        def _append() -> None:
            try:
                if not self.winfo_exists():
                    return
                box = getattr(self, "transcript_box", None)
                if box is not None:
                    box.configure(state="normal")
                    # Prefer tagged insert on underlying Text; fall back to CTk API.
                    tk_text = self._transcript_tk()
                    if tk_text is not None:
                        tk_text.insert("end", line, (tag,))
                    else:
                        try:
                            box.insert("end", line, tag)
                        except TypeError:
                            box.insert("end", line)
                    try:
                        box.see("end")
                    except Exception:  # noqa: BLE001
                        pass
                    box.configure(state="disabled")
                chat = getattr(self, "_chat_view", None)
                if chat is not None:
                    try:
                        from dana.ui.chat_view import _classify_role

                        role = _classify_role(speaker, agent_id)
                        if tag == "vision":
                            role = "system"
                        chat.append_bubble(
                            speaker, text, agent_id=agent_id, role=role
                        )
                        try:
                            chat._scroll_to_latest()
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:
                pass

        try:
            self.after(0, _append)
        except Exception:
            pass

    def _reload_device_menus(self) -> None:
        """No-op — Mic/Speaker menus removed; streams always use System Default."""
        try:
            from dana.audio.devices import SYSTEM_DEFAULT_LABEL
        except Exception:  # noqa: BLE001
            SYSTEM_DEFAULT_LABEL = "System Default (Auto)"
        self._mic_labels = [SYSTEM_DEFAULT_LABEL]
        self._speaker_labels = [SYSTEM_DEFAULT_LABEL]
        self._mic_by_label = {SYSTEM_DEFAULT_LABEL: None}
        self._speaker_by_label = {SYSTEM_DEFAULT_LABEL: None}

    def _refresh_stats(self) -> None:
        if not self.winfo_exists():
            return
        raw = get_ui_state()
        label = _UI_STATE_LABELS.get(raw, raw.title())
        try:
            if self.status_value is not None:
                self.status_value.configure(text=label)
        except Exception:  # noqa: BLE001
            pass
        wake = ", ".join(WAKEWORD_MODELS) if WAKEWORD_MODELS else "—"
        _wake_display = {"dana": "Dana", "alexa": "Alexa"}
        if wake != "—":
            parts = [w.strip() for w in wake.split(",") if w.strip()]
            wake_disp = ", ".join(_wake_display.get(p.lower(), p.title()) for p in parts)
        else:
            wake_disp = wake
        try:
            if self.wake_value is not None:
                self.wake_value.configure(text=f"Wake: {wake_disp}")
        except Exception:  # noqa: BLE001
            pass
        settings_wake = getattr(self, "_settings_wake_lbl", None)
        if settings_wake is not None:
            try:
                settings_wake.configure(text=f"Active wake word: {wake_disp}")
            except Exception:  # noqa: BLE001
                pass
        try:
            self._set_mode_indicator(get_dana_mode())
        except Exception:  # noqa: BLE001
            pass
        self.after(500, self._refresh_stats)

    def _save_and_apply_audio(self) -> None:
        """Compatibility stub — audio always binds System Default (device=None)."""
        state.AUDIO_INPUT_DEVICE = None
        state.AUDIO_OUTPUT_DEVICE = None
        state.AUDIO_INPUT_RATE = _device_rate(None)
        try:
            save_audio_settings(None, None)
        except Exception:  # noqa: BLE001
            pass
        request_mic_ingest_restart()
        ensure_mic_ingest_thread()
        note = getattr(self, "apply_note", None)
        if note is not None:
            try:
                note.configure(text="Audio: System Default (Auto)")
            except Exception:  # noqa: BLE001
                pass
        log("Audio", "GUI audio → System Default (autonomous; menus removed)")

    def show_window(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_force()
        try:
            from dana.ui.logo import schedule_window_icon

            schedule_window_icon(self, delay_ms=100)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _on_open_window_startup_toggle(self) -> None:
        """Persist Settings → Open window on startup immediately."""
        try:
            from dana.settings import set_open_window_on_startup

            enabled = bool(self._open_window_var.get())
            set_open_window_on_startup(enabled)
            log(
                "UI",
                f"open_window_on_startup={'True' if enabled else 'False'} (saved)",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: could not save open_window_on_startup ({exc})")

    def _on_hybrid_planner_toggle(self) -> None:
        """Persist Settings → Hybrid Broker (Cloud Planner); refresh DAG label."""
        try:
            from dana.settings import set_hybrid_planner_enabled

            enabled = bool(self._hybrid_planner_var.get())
            set_hybrid_planner_enabled(enabled)
            log(
                "UI",
                f"hybrid_planner_enabled={'True' if enabled else 'False'} (saved)",
            )
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: could not save hybrid_planner_enabled ({exc})")
        self._refresh_planner_mode_label(warn_missing_key=True)

    def _refresh_planner_mode_label(self, *, warn_missing_key: bool = False) -> None:
        """Update DAG drawer 'Planner Mode: […]' from settings + API key presence."""
        try:
            from dana.graph.cloud_planner import (
                planner_mode_label,
                publish_planner_mode,
            )

            mode = publish_planner_mode(warn_missing_key=warn_missing_key)
            mode = mode or planner_mode_label()
        except Exception:  # noqa: BLE001
            mode = "LOCAL"
        lbl = getattr(self, "_dag_planner_mode_lbl", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=f"Planner Mode: [{mode}]")
        except Exception:  # noqa: BLE001
            pass
        view = getattr(self, "dag_monitor_view", None)
        if view is not None and hasattr(view, "set_planner_mode"):
            try:
                view.set_planner_mode(mode)
            except Exception:  # noqa: BLE001
                pass

    def kill_dana_processes(self) -> dict[str, Any]:
        """Stage 8.9.2 — launch ``stop_dana.vbs`` / ``stop_dana.bat`` (non-blocking).

        Prefers the VBS silent runner (no console flash). Uses a detached
        ``subprocess.Popen`` so teardown can finish after this GUI process dies.
        """
        try:
            from dana.paths import PROJECT_ROOT

            root = Path(PROJECT_ROOT)
        except Exception:  # noqa: BLE001
            root = Path(__file__).resolve().parents[1]
        launchers = root / "scripts" / "launchers"
        vbs = launchers / "stop_dana.vbs"
        bat = launchers / "stop_dana.bat"
        if not vbs.is_file() and not bat.is_file():
            # Fallback: thin root wrappers / legacy layout / unit-test tmp roots.
            vbs = root / "stop_dana.vbs"
            bat = root / "stop_dana.bat"
        runner = vbs if vbs.is_file() else bat
        if not runner.is_file():
            msg = (
                f"stop_dana.vbs / stop_dana.bat not found under "
                f"{launchers} or {root}"
            )
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": "FileNotFoundError", "message": msg}
        try:
            creationflags = 0
            startupinfo = None
            if sys.platform == "win32":
                # Hide wscript/cmd host for stop_dana (no flashing console).
                try:
                    from dana.vault_service import windows_no_window_creationflags

                    creationflags |= windows_no_window_creationflags()
                except Exception:  # noqa: BLE001
                    creationflags |= int(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                    )
                if hasattr(subprocess, "DETACHED_PROCESS"):
                    creationflags |= int(subprocess.DETACHED_PROCESS)
                if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                    creationflags |= int(subprocess.CREATE_NEW_PROCESS_GROUP)
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
            # shell=True + absolute path — matches Windows .vbs/.bat launch semantics.
            proc = subprocess.Popen(  # noqa: S603
                f'"{runner}"',
                cwd=str(root),
                shell=True,
                creationflags=creationflags,
                startupinfo=startupinfo,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            log("UI", f"STOP DANA — launched {runner.name} pid={proc.pid}")
            return {"ok": True, "pid": int(proc.pid), "path": str(runner)}
        except FileNotFoundError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": "FileNotFoundError", "message": msg}
        except OSError as exc:
            msg = f"Failed to launch {runner.name}: {exc}"
            log("UI", f"WARNING: {msg}")
            return {"ok": False, "error": type(exc).__name__, "message": msg}

    def _halt_engine_full(self) -> None:
        """In-process full halt: terminate workers, cancel audio/LLM, STOPPED pill.

        Kill-switch ``stop_dana.*`` launch remains separate (see
        ``_on_stop_dana_clicked``); this path must run even if the batch is
        missing so the Local Engine goes inactive immediately.
        """
        # Drop dictation latch if hot.
        if self._dictation_active:
            try:
                from dana.management.dictation import toggle_dictation_mode

                toggle_dictation_mode(False)
            except Exception:  # noqa: BLE001
                pass
            self._dictation_active = False
            try:
                self.dictation_btn.configure(
                    text="  ●  OFF  ",
                    fg_color=_UI_GHOST,
                    hover_color="#34344A",
                    border_color=_UI_CARD_BORDER,
                    text_color="#F87171",
                )
            except Exception:  # noqa: BLE001
                pass
        self.engine_active = False
        self._engine_stopped = True
        state.engine_engaged.clear()
        # Termination latch — workers / conversation loop exit.
        try:
            stop_event.set()
        except Exception:  # noqa: BLE001
            pass
        # Cancel VAD / mic latch + pending TTS.
        try:
            is_recording.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            vad_capture_active.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_tts_audio_state("stop_dana", ui_state="idle")
        except Exception:  # noqa: BLE001
            pass
        # Abort pending actuators / Ghost Typist / in-flight LLM actions.
        try:
            from dana.middleware.kill_switch import trigger_halt

            trigger_halt(reason="stop_dana")
        except Exception:  # noqa: BLE001
            pass
        # Detach daemon sidecar reconnect (do not hot_restart — hard stop).
        try:
            client = getattr(self, "_daemon_client", None)
            if client is not None:
                try:
                    client.stop_auto_reconnect()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    client._drop_socket()
                except Exception:  # noqa: BLE001
                    pass
                self._daemon_client = None
        except Exception:  # noqa: BLE001
            pass
        self._set_behavior_controls_locked(False)
        self._refresh_engine_ui()
        try:
            from dana.ui.status_bus import emit_state_change

            emit_state_change("idle", message="● STOPPED | Local Engine")
        except Exception:  # noqa: BLE001
            pass
        log("UI", "STOP DANA — engine halted (STOPPED)")
        try:
            self.log_transcript(
                "Dana",
                "Engine STOPPED — Local Engine halted.",
                agent_id="broker",
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.assistive_orb is not None:
                self.assistive_orb.refresh_controls()
        except Exception:  # noqa: BLE001
            pass

    def _on_stop_dana_clicked(self) -> None:
        """Halt engine in-process, show TERMINATING…, then fire kill switch."""
        btn = getattr(self, "stop_dana_btn", None)
        try:
            self._halt_engine_full()
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: in-process engine halt failed ({exc})")
        try:
            if btn is not None:
                btn.configure(
                    text="TERMINATING...",
                    state="disabled",
                    fg_color=_UI_ROSE_HOVER,
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            self.update_idletasks()
        except Exception:  # noqa: BLE001
            pass

        def _launch() -> None:
            result = self.kill_dana_processes()
            if not result.get("ok"):
                try:
                    if btn is not None:
                        btn.configure(
                            text="STOP DANA",
                            state="normal",
                            fg_color=_UI_ROSE,
                        )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    log(
                        "UI",
                        f"STOP DANA aborted: {result.get('message') or 'unknown'}",
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            # Brief paint of TERMINATING… before the batch tears us down.
            self.after(80, _launch)
        except Exception:  # noqa: BLE001
            _launch()

    def _start_assistive_orb(self) -> None:
        """Stage 8.7 — spawn frameless topmost orb on the Tk main thread.

        DISABLED: experimental orb window is not launched at engine start.
        """
        # Experimental AssistiveTouch orb — leave commented until re-enabled.
        return
        # if self.assistive_orb is not None:
        #     return
        # try:
        #     from dana.ui.assistive_orb import AssistiveTouchOrb
        #
        #     def _active_agent() -> str:
        #         try:
        #             from dana.audio.multi_voice_tts import get_active_tts_agent
        #
        #             return str(get_active_tts_agent() or "broker")
        #         except Exception:  # noqa: BLE001
        #             return "broker"
        #
        #     self.assistive_orb = AssistiveTouchOrb(
        #         self,
        #         on_toggle_dictation=self._toggle_dictation_mode,
        #         on_open_dashboard=self.show_window,
        #         on_approve_ticket=self._orb_approve_ticket,
        #         on_deny_ticket=self._orb_deny_ticket,
        #         dictation_getter=lambda: bool(self._dictation_active),
        #         mode_getter=lambda: str(self._header_mode or "chat"),
        #         accent_getter=lambda: self._mode_accent(),
        #         agent_getter=_active_agent,
        #     )
        # except Exception as exc:  # noqa: BLE001
        #     log("UI", f"WARNING: AssistiveTouch orb failed to start ({exc})")
        #     self.assistive_orb = None

    def open_github_issue(
        self,
        ticket_content: dict[str, Any] | str | None = None,
        jason_critique: str = "",
    ) -> str:
        """Stage 8.9.3 — open a pre-filled GitHub issue in the default browser."""
        try:
            from dana.middleware.hitl_ticket import get_pending
            from dana.ui.github_escalation import open_github_issue as _open

            pending = ticket_content if ticket_content is not None else get_pending()
            return _open(pending, jason_critique or str((pending or {}).get("jason_critique") or ""))
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: open_github_issue failed ({exc})")
            return ""

    def _orb_approve_ticket(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(True, action="approve")
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: orb approve failed ({exc})")
        try:
            if getattr(self, "live_trace", None) is not None:
                self.live_trace._set_hitl_visible(False)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _orb_deny_ticket(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(False, action="deny")
        except Exception as exc:  # noqa: BLE001
            log("UI", f"WARNING: orb deny failed ({exc})")
        try:
            if getattr(self, "live_trace", None) is not None:
                self.live_trace._set_hitl_visible(False)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _on_close_to_tray(self) -> None:
        # Dashboard hides; AssistiveTouch orb stays visible as the always-on control.
        self.withdraw()
