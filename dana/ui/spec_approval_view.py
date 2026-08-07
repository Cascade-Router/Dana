"""Human-in-the-Loop Spec Approval Card for compiled ``/broker`` macros."""

from __future__ import annotations

from typing import Any, Callable

import customtkinter as ctk

from dana.ui import theme as T


class SpecApprovalCard(ctk.CTkFrame):
    """Draft-and-Approve card: show compiled macro + Approve / Edit / Cancel."""

    def __init__(
        self,
        master: Any,
        *,
        on_approve: Callable[[str], None] | None = None,
        on_edit: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            master,
            fg_color=T.CARD,
            corner_radius=12,
            border_width=1,
            border_color=T.AMBER,
        )
        self._on_approve = on_approve
        self._on_edit = on_edit
        self._on_cancel = on_cancel
        self._compiled_spec = ""
        self._payload: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            hdr,
            text="Spec Approval — PENDING_USER_APPROVAL",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=T.AMBER,
            anchor="w",
        ).pack(side="left")
        self._epic_lbl = ctk.CTkLabel(
            hdr,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            anchor="e",
        )
        self._epic_lbl.pack(side="right")

        ctk.CTkLabel(
            self,
            text="Compiled /broker macro (review before Meta-Broker dispatch)",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        self._spec_box = ctk.CTkTextbox(
            self,
            fg_color=T.BG,
            text_color=T.TEXT,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            height=120,
            activate_scrollbars=True,
        )
        self._spec_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(
            actions,
            text="Approve & Run",
            height=32,
            corner_radius=8,
            fg_color=T.EMERALD,
            hover_color=T.EMERALD_HOVER,
            text_color=T.TEXT_ON_ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._click_approve,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            actions,
            text="Edit Macro",
            height=32,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=T.BORDER,
            text_color=T.TEXT,
            font=ctk.CTkFont(size=12),
            command=self._click_edit,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            actions,
            text="Cancel",
            height=32,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=T.ROSE,
            text_color=T.ROSE,
            font=ctk.CTkFont(size=12),
            command=self._click_cancel,
        ).pack(side="left")

    def present(self, payload: dict[str, Any]) -> None:
        """Populate the card from a ``spec_approval_request`` payload."""
        self._payload = dict(payload or {})
        spec = str(self._payload.get("compiled_spec") or "").strip()
        self._compiled_spec = spec
        epics = list(self._payload.get("epics") or [])
        try:
            self._epic_lbl.configure(
                text=f"{len(epics)} epic(s)" if epics else "draft spec"
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self._spec_box.configure(state="normal")
            self._spec_box.delete("1.0", "end")
            self._spec_box.insert("1.0", spec or "(empty spec)")
            self._spec_box.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

    def compiled_spec(self) -> str:
        return self._compiled_spec

    def _click_approve(self) -> None:
        if self._on_approve is not None:
            try:
                self._on_approve(self._compiled_spec)
            except Exception:  # noqa: BLE001
                pass

    def _click_edit(self) -> None:
        if self._on_edit is not None:
            try:
                self._on_edit(self._compiled_spec)
            except Exception:  # noqa: BLE001
                pass

    def _click_cancel(self) -> None:
        if self._on_cancel is not None:
            try:
                self._on_cancel()
            except Exception:  # noqa: BLE001
                pass


__all__ = ("SpecApprovalCard",)
