"""CustomTkinter Live Trace panel — drains TraceEventBus on the Tk main thread."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from typing import Any

import customtkinter as ctk

from dana.ui.trace_bus import TraceEvent, get_trace_bus


def _startup_log_path() -> str:
    """Locate OS-level dana_startup.log written by login/startup launchers."""
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ.get("TEMP", os.environ.get("TMP", ".")), "dana_startup.log")
    return "/tmp/dana_startup.log"


def _project_root() -> str:
    try:
        from dana.paths import PROJECT_ROOT

        return os.path.abspath(str(PROJECT_ROOT))
    except Exception:  # noqa: BLE001
        return os.path.abspath(os.getcwd())


def _open_startup_log() -> str:
    """Open the startup log in the native text editor (worker-thread safe)."""
    log_path = _startup_log_path()
    if not os.path.isfile(log_path):
        try:
            with open(log_path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            return f"Startup log unavailable ({log_path}): {exc}"
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(log_path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.call(["open", log_path])
        else:
            subprocess.call(["xdg-open", log_path])
    except Exception as exc:  # noqa: BLE001
        return f"Failed to open startup log: {exc}"
    return f"Opened startup log: {log_path}"


def _spawn_headless_boot_terminal() -> str:
    """Spawn a native terminal running ``python run.py --no-gui`` (keeps window open)."""
    root = _project_root()
    system = platform.system()
    try:
        if system == "Windows":
            # ``start cmd /k`` keeps the console open after crash/traceback.
            subprocess.Popen(
                f'start cmd /k "cd /d {root} && python run.py --no-gui"',
                shell=True,
            )
        elif system == "Darwin":
            script = (
                f'cd {root} && python run.py --no-gui; '
                r'read -p "Press Enter to close"'
            )
            # Escape for AppleScript string literal.
            escaped = script.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'tell application "Terminal" to do script "{escaped}"',
                ]
            )
        else:
            subprocess.Popen(
                [
                    "x-terminal-emulator",
                    "-e",
                    f'bash -c "cd {root} && python run.py --no-gui; exec bash"',
                ]
            )
    except Exception as exc:  # noqa: BLE001
        return f"Failed to spawn headless terminal: {exc}"
    return "Spawned headless boot terminal (python run.py --no-gui)."

_MODE_COLORS = {
    "chat": "#10b981",
    "developer": "#F59E0B",
    "agentic": "#F59E0B",
    "vision": "#10b981",
    "research": "#F59E0B",
    "dictation": "#A855F7",
    "idle": "#94A3B8",
    "routing": "#F59E0B",
    "tool": "#8B5CF6",
    "synthesis": "#10b981",
}
_CARD_BG = "#131b2e"
_CANVAS_BG = "#0a0e17"
_GHOST_BG = "#1e293b"
_GHOST_BORDER = "#1e293b"

_STATUS_PILLS = {
    "idle": ("[IDLE]", "#94A3B8"),
    "routing": ("[ROUTING]", "#F59E0B"),
    "tool": ("[TOOL]", "#8B5CF6"),
    "synthesis": ("[SYNTHESIS]", "#10b981"),
    "active": ("[ACTIVE]", "#10b981"),
}


class LiveTracePanel(ctk.CTkFrame):
    """Dark Live Trace dashboard: status pill, timeline, payload viewer."""

    def __init__(
        self, master: Any, *, poll_ms: int = 50, external_tick: bool = False
    ) -> None:
        super().__init__(master, fg_color=_CANVAS_BG)
        self._poll_ms = max(30, int(poll_ms))
        self._external_tick = bool(external_tick)
        self._phase = "idle"
        self._mode = "chat"
        self._node_t0: dict[str, float] = {}
        self._timeline_rows: list[ctk.CTkFrame] = []
        self._max_rows = 80
        self._build()
        if not self._external_tick:
            self.after(self._poll_ms, self._drain_trace_queue)

    def _build(self) -> None:
        header = ctk.CTkFrame(
            self,
            fg_color=_CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=_GHOST_BORDER,
        )
        header.pack(fill="x", padx=10, pady=(10, 6))
        self.pill = ctk.CTkLabel(
            header,
            text="[IDLE]",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_STATUS_PILLS["idle"][1],
            fg_color=_GHOST_BG,
            corner_radius=999,
            padx=14,
            pady=6,
        )
        self.pill.pack(side="left", padx=12, pady=10)
        self.mode_label = ctk.CTkLabel(
            header,
            text="LangGraph Live Trace",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#E5E7EB",
            anchor="w",
        )
        self.mode_label.pack(side="left", padx=8, pady=10)
        ctk.CTkLabel(
            header,
            text="pipeline · payload",
            text_color="#6B7280",
            font=ctk.CTkFont(size=11),
        ).pack(side="right", padx=14, pady=10)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(
            body,
            fg_color=_CARD_BG,
            corner_radius=16,
            border_width=1,
            border_color=_GHOST_BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(
            left,
            text="State Graph Timeline",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F3F4F6",
        ).pack(fill="x", padx=14, pady=(12, 4))
        self.timeline = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.timeline.pack(fill="both", expand=True, padx=8, pady=(0, 10))

        right = ctk.CTkFrame(
            body,
            fg_color=_CARD_BG,
            corner_radius=16,
            border_width=1,
            border_color=_GHOST_BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ctk.CTkLabel(
            right,
            text="Payload Viewer",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F3F4F6",
        ).pack(fill="x", padx=14, pady=(12, 4))
        self.payload = ctk.CTkTextbox(
            right,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=_CANVAS_BG,
            border_width=0,
            corner_radius=10,
        )
        self.payload.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.payload.insert("1.0", "Waiting for LangGraph transitions…\n")
        self.payload.configure(state="disabled")

        # Stage 8.6 — HITL Approve / Deny (hidden until ticket_approval interrupt).
        self._hitl_bar = ctk.CTkFrame(
            self,
            fg_color=_CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color="#388e3c",
        )
        self._hitl_label = ctk.CTkLabel(
            self._hitl_bar,
            text="Ticket approval required",
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#E8F5E9",
        )
        self._hitl_label.pack(side="left", padx=(14, 8), pady=10)
        self._hitl_approve_btn = ctk.CTkButton(
            self._hitl_bar,
            text="Approve & Submit",
            width=150,
            height=32,
            corner_radius=999,
            fg_color="#388e3c",
            hover_color="#2E7D32",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_hitl_approve,
        )
        self._hitl_approve_btn.pack(side="right", padx=(4, 12), pady=8)
        self._hitl_deny_btn = ctk.CTkButton(
            self._hitl_bar,
            text="Deny / Edit",
            width=120,
            height=32,
            corner_radius=999,
            fg_color="#C62828",
            hover_color="#B71C1C",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_hitl_deny,
        )
        self._hitl_deny_btn.pack(side="right", padx=4, pady=8)
        # Stage 8.9.3 — shown only when consecutive_denials >= 2.
        self._hitl_github_btn = ctk.CTkButton(
            self._hitl_bar,
            text="\U0001f419 Report Issue on GitHub",
            width=200,
            height=32,
            corner_radius=999,
            fg_color="#24292F",
            hover_color="#1B1F23",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_hitl_github,
        )
        self._hitl_visible = False
        self._hitl_github_visible = False

        # Subtle footer toolbar — ghost secondary actions.
        self._dev_footer = ctk.CTkFrame(
            self,
            fg_color=_GHOST_BG,
            corner_radius=12,
            border_width=1,
            border_color=_GHOST_BORDER,
            height=44,
        )
        self._dev_footer.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(
            self._dev_footer,
            text="Developer Tools",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
        ).pack(side="left", padx=(12, 8), pady=8)
        for label, cmd in (
            ("View Startup Log", self._on_view_startup_log),
            ("Test Headless Boot", self._on_test_headless_boot),
        ):
            ctk.CTkButton(
                self._dev_footer,
                text=label,
                width=140,
                height=28,
                corner_radius=999,
                fg_color=_CANVAS_BG,
                hover_color="#34344A",
                border_width=1,
                border_color=_GHOST_BORDER,
                text_color="#D1D5DB",
                font=ctk.CTkFont(size=11),
                command=cmd,
            ).pack(side="left", padx=4, pady=8)
        # Stage 8.9.1 — truncate local HITL feedback JSONL (no app restart).
        ctk.CTkButton(
            self._dev_footer,
            text="Clear Logs",
            width=100,
            height=28,
            corner_radius=999,
            fg_color=_CANVAS_BG,
            hover_color="#4A1520",
            border_width=1,
            border_color="#d32f2f",
            text_color="#EF9A9A",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_clear_feedback_logs,
        ).pack(side="left", padx=(8, 4), pady=8)
        self._clear_logs_status = ctk.CTkLabel(
            self._dev_footer,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=10),
            text_color="#9CA3AF",
        )
        self._clear_logs_status.pack(side="left", padx=(2, 10), pady=8)
        self._clear_logs_status_job: str | None = None
        # Stage 8.9.8 — footer KILL SWITCH removed; header STOP DONNA is sole exit.

    def _on_clear_feedback_logs(self) -> None:
        """Tk-main-thread clear of feedback_logs.jsonl + 3s status toast."""
        try:
            from dana.memory.feedback_log import clear_feedback_logs

            result = clear_feedback_logs()
            msg = str(
                (result or {}).get("message")
                or "Logs Cleared (0 B)"
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Clear failed: {exc}"
        try:
            self._clear_logs_status.configure(text=msg, text_color="#EF9A9A")
        except Exception:  # noqa: BLE001
            pass
        self._append_timeline(f"Dev → {msg}", accent="#d32f2f")
        if self._clear_logs_status_job is not None:
            try:
                self.after_cancel(self._clear_logs_status_job)
            except Exception:  # noqa: BLE001
                pass
            self._clear_logs_status_job = None

        def _clear_toast() -> None:
            self._clear_logs_status_job = None
            try:
                self._clear_logs_status.configure(text="")
            except Exception:  # noqa: BLE001
                pass

        try:
            self._clear_logs_status_job = self.after(3000, _clear_toast)
        except Exception:  # noqa: BLE001
            pass

    def _run_dev_action(self, action: Any, label: str) -> None:
        """Run a developer action off the Tk main thread."""

        def _worker() -> None:
            try:
                result = action()
            except Exception as exc:  # noqa: BLE001
                result = f"{label} failed: {exc}"
            msg = str(result or label)

            def _ui() -> None:
                self._append_timeline(msg, accent=_MODE_COLORS.get("developer"))
                self._set_payload(msg)

            try:
                self.after(0, _ui)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_worker, name=f"donna-dev-{label}", daemon=True).start()

    def _on_view_startup_log(self) -> None:
        self._run_dev_action(_open_startup_log, "View Startup Log")

    def _on_test_headless_boot(self) -> None:
        self._run_dev_action(_spawn_headless_boot_terminal, "Test Headless Boot")

    def _set_hitl_visible(self, visible: bool) -> None:
        """Reveal or hide Approve / Deny controls (Tk main thread only)."""
        self._hitl_visible = bool(visible)
        try:
            if self._hitl_visible:
                self._sync_github_escalate_button()
                if not self._hitl_bar.winfo_ismapped():
                    footer = getattr(self, "_dev_footer", None)
                    if footer is not None:
                        self._hitl_bar.pack(
                            fill="x", padx=10, pady=(0, 6), before=footer
                        )
                    else:
                        self._hitl_bar.pack(fill="x", padx=10, pady=(0, 6))
            else:
                self._set_github_escalate_visible(False)
                self._hitl_bar.pack_forget()
        except Exception:  # noqa: BLE001
            try:
                if self._hitl_visible:
                    self._hitl_bar.pack(fill="x", padx=10, pady=(0, 6))
                else:
                    self._hitl_bar.pack_forget()
            except Exception:  # noqa: BLE001
                pass

    def _sync_github_escalate_button(self) -> None:
        """Stage 8.9.3 — reveal GitHub escalate when denials >= 2."""
        denials = 0
        try:
            from dana.middleware.hitl_ticket import (
                get_consecutive_denials,
                get_pending,
            )

            pending = get_pending() or {}
            denials = int(
                pending.get("consecutive_denials")
                if pending.get("consecutive_denials") is not None
                else get_consecutive_denials()
            )
        except Exception:  # noqa: BLE001
            denials = 0
        self._set_github_escalate_visible(denials >= 2)

    def _set_github_escalate_visible(self, visible: bool) -> None:
        self._hitl_github_visible = bool(visible)
        try:
            if self._hitl_github_visible:
                if not self._hitl_github_btn.winfo_ismapped():
                    self._hitl_github_btn.pack(
                        side="right", padx=(4, 4), pady=8, before=self._hitl_deny_btn
                    )
            else:
                self._hitl_github_btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _on_hitl_github(self) -> None:
        """Open a pre-filled GitHub issue for repeated HITL failures."""
        try:
            from dana.middleware.hitl_ticket import get_pending
            from dana.ui.github_escalation import open_github_issue

            pending = get_pending() or {}
            url = open_github_issue(
                pending,
                str(pending.get("jason_critique") or ""),
            )
            self._append_timeline(
                "HITL → Report Issue on GitHub",
                accent="#24292F",
            )
            if url:
                self._set_payload(f"GitHub issue draft opened:\n{url[:500]}")
        except Exception as exc:  # noqa: BLE001
            self._append_timeline(
                f"HITL → GitHub escalate failed: {exc}",
                accent="#C62828",
            )

    def _on_hitl_approve(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(True, action="approve")
        except Exception:  # noqa: BLE001
            pass
        self._set_hitl_visible(False)
        self._append_timeline("HITL → Approve & Submit", accent="#388e3c")

    def _on_hitl_deny(self) -> None:
        try:
            from dana.middleware.hitl_ticket import submit_decision

            submit_decision(False, action="deny")
        except Exception:  # noqa: BLE001
            pass
        self._set_hitl_visible(False)
        self._append_timeline("HITL → Deny / Edit", accent="#C62828")
        self._set_payload(
            "Ticket denied. Graph returned to idle — edit your request in chat "
            "and try again when ready."
        )

    def _set_pill(self, phase: str, *, tool: str = "") -> None:
        self._phase = phase
        if phase == "tool" and tool:
            label = f"[TOOL: {tool}]"
            color = _MODE_COLORS["tool"]
        else:
            label, color = _STATUS_PILLS.get(phase, _STATUS_PILLS["active"])
        try:
            self.pill.configure(text=label, text_color=color)
        except Exception:  # noqa: BLE001
            pass

    def _set_mode(self, mode: str) -> None:
        key = (mode or "chat").strip().lower() or "chat"
        if key == "agentic":
            key = "developer"
        self._mode = key
        # Global Mode indicator lives in DonnaGUI status bar; keep local title stable.

    def _append_timeline(self, line: str, *, accent: str | None = None) -> None:
        row = ctk.CTkFrame(
            self.timeline,
            fg_color=_GHOST_BG,
            corner_radius=10,
            border_width=1,
            border_color=_GHOST_BORDER,
        )
        row.pack(fill="x", padx=2, pady=3)
        ctk.CTkLabel(
            row,
            text=line,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=accent or ("gray20", "gray85"),
        ).pack(fill="x", padx=8, pady=6)
        self._timeline_rows.append(row)
        while len(self._timeline_rows) > self._max_rows:
            old = self._timeline_rows.pop(0)
            try:
                old.destroy()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.timeline._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _set_payload(self, text: str) -> None:
        snippet = (text or "").strip()
        if not snippet:
            return
        try:
            self.payload.configure(state="normal")
            self.payload.delete("1.0", "end")
            self.payload.insert("1.0", snippet + "\n")
            self.payload.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def _handle_event(self, event: TraceEvent) -> None:
        if event.mode:
            self._set_mode(event.mode)
        et = event.event_type
        node = event.node or "node"
        if et == "node_enter":
            self._node_t0[node] = time.perf_counter()
            phase = "routing" if "router" in node.lower() or node.lower() == "agent" else "active"
            if "tool" in node.lower():
                phase = "tool"
            if "synth" in node.lower() or "finish" in node.lower():
                phase = "synthesis"
            self._set_pill(phase, tool=event.tool)
            self._append_timeline(
                f"→ enter {node}",
                accent=_MODE_COLORS.get(self._mode),
            )
        elif et == "node_exit":
            t0 = self._node_t0.pop(node, None)
            ms = event.latency_ms
            if ms is None and t0 is not None:
                ms = (time.perf_counter() - t0) * 1000.0
            latency = f" ({ms:.0f}ms)" if ms is not None else ""
            self._append_timeline(f"← exit {node}{latency}")
            if event.payload:
                self._set_payload(event.payload)
        elif et == "tool_execution":
            tool = event.tool or node or "tool"
            self._set_pill("tool", tool=tool)
            ms = f" ({event.latency_ms:.0f}ms)" if event.latency_ms is not None else ""
            self._append_timeline(
                f"Router Node → Tool: {tool}{ms}",
                accent=_MODE_COLORS["tool"],
            )
            if event.payload:
                self._set_payload(event.payload)
            elif event.message:
                self._set_payload(event.message)
        elif et == "state_update":
            keys = ", ".join(event.state_keys) if event.state_keys else "state"
            self._append_timeline(f"state ← {keys}")
            if event.payload:
                self._set_payload(event.payload)
        elif et == "mode":
            self._set_mode(event.mode or event.message)
            self._append_timeline(event.message or f"mode={self._mode}")
        else:
            msg = str(event.message or "")
            # Stage 8.6 — reveal / hide HITL confirmation buttons.
            if "HITL_PENDING_APPROVAL" in msg or (
                event.node == "ticket_approval" and "awaiting" in msg.lower()
            ):
                self._set_pill("tool", tool="HITL")
                self._append_timeline(
                    "HITL → ticket approval required",
                    accent="#388e3c",
                )
                if event.payload:
                    self._set_payload(event.payload)
                self._set_hitl_visible(True)
                return
            if "HITL_RESOLVED" in msg or "HITL_RESUME" in msg:
                self._set_hitl_visible(False)
                self._append_timeline(msg or "HITL resolved")
                if event.payload:
                    self._set_payload(event.payload)
                return
            if event.message:
                self._append_timeline(event.message)
            if event.payload:
                self._set_payload(event.payload)

    def tick(self) -> None:
        """One polling step — safe to call from an external scheduler.

        Drains TraceEventBus on the Tk main thread (never from worker
        threads). Does not reschedule itself; that's the caller's job.
        """
        try:
            if not self.winfo_exists():
                return
        except Exception:  # noqa: BLE001
            return
        try:
            for event in get_trace_bus().drain(max_items=48):
                self._handle_event(event)
        except Exception:  # noqa: BLE001
            pass

    def _drain_trace_queue(self) -> None:
        """Self-scheduling wrapper around ``tick()`` (internal-poll mode only)."""
        if not self.winfo_exists():
            return
        self.tick()
        try:
            self.after(self._poll_ms, self._drain_trace_queue)
        except Exception:  # noqa: BLE001
            pass


class LiveTraceWindow(ctk.CTkToplevel):
    """Diagnostics overlay: Live Trace + System Log + Recent Sessions."""

    def __init__(self, master: Any | None = None) -> None:
        super().__init__(master)
        self.title("Dānā — Diagnostics / Live Trace")
        self.geometry("900x640")
        self.minsize(720, 520)
        ctk.set_appearance_mode("dark")

        self.panel = LiveTracePanel(self)
        self.panel.pack(fill="both", expand=True, padx=6, pady=(6, 2))

        extras = ctk.CTkFrame(self, fg_color="transparent")
        extras.pack(fill="both", expand=False, padx=6, pady=(2, 8))
        try:
            extras.grid_columnconfigure(0, weight=1)
            extras.grid_columnconfigure(1, weight=1)
        except Exception:  # noqa: BLE001
            pass

        log_card = ctk.CTkFrame(extras, fg_color=("#1e1e2e", "#1e1e2e"), corner_radius=10)
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ctk.CTkLabel(
            log_card,
            text="System Log",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 2))
        self.system_log_box = ctk.CTkTextbox(
            log_card,
            wrap="word",
            height=140,
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self.system_log_box.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.system_log_box.insert("1.0", "(log empty — click Refresh)\n")
        self.system_log_box.configure(state="disabled")
        ctk.CTkButton(
            log_card,
            text="Refresh log",
            width=100,
            height=26,
            command=self._refresh_log_via_master,
        ).pack(anchor="e", padx=8, pady=(0, 8))

        sess_card = ctk.CTkFrame(extras, fg_color=("#1e1e2e", "#1e1e2e"), corner_radius=10)
        sess_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ctk.CTkLabel(
            sess_card,
            text="Recent Sessions",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 2))
        self.sessions_box = ctk.CTkTextbox(
            sess_card,
            wrap="word",
            height=140,
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self.sessions_box.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.sessions_box.insert("1.0", "(no dictation sessions yet)\n")
        self.sessions_box.configure(state="disabled")
        ctk.CTkButton(
            sess_card,
            text="Refresh sessions",
            width=120,
            height=26,
            command=self._refresh_sessions_via_master,
        ).pack(anchor="e", padx=8, pady=(0, 8))

    def _refresh_log_via_master(self) -> None:
        master = self.master
        fn = getattr(master, "_refresh_system_log", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:  # noqa: BLE001
                pass
        self._local_refresh_log()

    def _refresh_sessions_via_master(self) -> None:
        master = self.master
        fn = getattr(master, "refresh_dictation_sessions", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:  # noqa: BLE001
                pass

    def _local_refresh_log(self) -> None:
        text = "(log unavailable)\n"
        try:
            from dana.logging import RUNTIME_LOG_PATH

            path = RUNTIME_LOG_PATH
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                text = "".join(lines[-40:]) or "(log empty)\n"
        except Exception as exc:  # noqa: BLE001
            text = f"Could not read log: {exc}\n"
        try:
            self.system_log_box.configure(state="normal")
            self.system_log_box.delete("1.0", "end")
            self.system_log_box.insert("1.0", text)
            self.system_log_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass
