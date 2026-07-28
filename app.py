"""Dānā · Hugging Face Space — tabless dark cybernetic dashboard."""

from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw

try:
    import spaces
except ImportError:  # Local / CI without ZeroGPU runtime

    class _SpacesFallback:
        @staticmethod
        def GPU(fn=None, **_kwargs):  # noqa: N802 — match HF API
            if fn is None:
                return lambda f: f
            return fn

    spaces = _SpacesFallback()  # type: ignore[assignment]

_CSS = """
:root {
  --dana-bg: #0b0f17;
  --dana-panel: #111827;
  --dana-border: #1f2937;
  --dana-cyan: #22d3ee;
  --dana-approve: #10b981;
  --dana-deny: #ef4444;
  --dana-text: #e5e7eb;
  --dana-muted: #94a3b8;
}
.gradio-container {
  background: var(--dana-bg) !important;
  color: var(--dana-text) !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
}
footer { display: none !important; }
.dana-hero h1 {
  font-size: 2.1rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  color: #f8fafc !important;
  margin-bottom: 0.35rem !important;
}
.dana-hero p, .dana-hero li {
  color: var(--dana-muted) !important;
}
.dana-panel {
  background: var(--dana-panel) !important;
  border: 1px solid var(--dana-border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.1rem !important;
}
.dana-status {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 0.95rem !important;
  padding: 0.65rem 0.9rem !important;
  border-radius: 10px !important;
  border: 1px solid var(--dana-border) !important;
  background: #0f172a !important;
  color: #e2e8f0 !important;
}
button.dana-approve {
  background: var(--dana-approve) !important;
  border-color: var(--dana-approve) !important;
  color: #04160f !important;
  font-weight: 650 !important;
}
button.dana-deny {
  background: var(--dana-deny) !important;
  border-color: var(--dana-deny) !important;
  color: #fff !important;
  font-weight: 650 !important;
}
button.dana-primary {
  background: linear-gradient(135deg, #0891b2, #0e7490) !important;
  border: none !important;
  color: #ecfeff !important;
  font-weight: 650 !important;
}
.dana-ticket-title {
  color: var(--dana-cyan) !important;
  font-weight: 600 !important;
  margin: 0 0 0.5rem 0 !important;
}
"""

DEFAULT_MEMORY_LEDGER: dict[str, Any] = {
    "user_identity": "Amirhosein",
    "active_project": "Dānā Agentic Architecture",
    "stored_preferences": {"theme": "dark", "patch_style": "minimal_diff"},
    "recent_context": "Validated e820f01 planner graph refactor",
}

_SAMPLE_PNG: Path | None = None


def _sample_desktop() -> Image.Image:
    """Synthetic desktop screenshot for one-click demos."""
    img = Image.new("RGB", (960, 540), "#0f172a")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 960, 40), fill="#1e293b")
    draw.rectangle((24, 72, 936, 500), fill="#111827", outline="#334155", width=2)
    draw.rectangle((40, 96, 280, 132), fill="#164e63", outline="#22d3ee", width=2)
    draw.text((52, 106), "Search Bar", fill="#ecfeff")
    draw.rectangle((760, 96, 900, 132), fill="#065f46", outline="#10b981", width=2)
    draw.text((792, 106), "Save", fill="#ecfdf5")
    draw.rectangle((40, 160, 620, 420), fill="#0b1220", outline="#1f2937", width=1)
    draw.text((56, 180), "Active Window — Notepad", fill="#94a3b8")
    draw.text((56, 220), "ERROR: TypeError on line 42", fill="#fca5a5")
    draw.text((56, 250), "WARNING: unused import json", fill="#fcd34d")
    return img


def _sample_desktop_path() -> str:
    """Persist sample desktop PNG for ``gr.Examples`` image inputs."""
    global _SAMPLE_PNG
    if _SAMPLE_PNG is None or not _SAMPLE_PNG.is_file():
        path = Path(tempfile.gettempdir()) / "dana_sample_desktop.png"
        _sample_desktop().save(path, format="PNG")
        _SAMPLE_PNG = path
    return str(_SAMPLE_PNG)


def _empty_annotated() -> None:
    return None


def _merge_memory_ledger(
    ledger: dict[str, Any] | None,
    command: str,
    *,
    vision_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update Active Memory Ledger for store / read / vision intents."""
    out = copy.deepcopy(ledger if isinstance(ledger, dict) else DEFAULT_MEMORY_LEDGER)
    prefs = dict(out.get("stored_preferences") or {})
    low = (command or "").lower()
    vision_meta = vision_meta or {}

    store_match = re.search(
        r"(?:store\s+preference|remember|save\s+preference)\s*:?\s*(.+)$",
        command or "",
        re.I,
    )
    if store_match:
        pref_text = store_match.group(1).strip()
        if "pep8" in low:
            prefs["python_formatting"] = "PEP8"
        prefs["last_stored"] = pref_text[:160]
        out["stored_preferences"] = prefs
        out["recent_context"] = f"Stored preference: {pref_text[:200]}"
        return out

    if any(k in low for k in ("read memory", "recall", "what do you know", "show memory")):
        out["recent_context"] = (
            f"Read ledger for {out.get('user_identity')}: "
            f"project={out.get('active_project')}; prefs={prefs}"
        )
        return out

    if any(k in low for k in ("summarize", "highlight", "active window", "ocr")):
        bbox = vision_meta.get("bounding_box_xyxy")
        out["recent_context"] = (
            f"Vision grounding for: {(command or '')[:140]} | bbox={bbox}"
        )
        return out

    if any(k in low for k in ("delete", "restart", "wipe", "rm -rf")):
        out["recent_context"] = (
            f"HITL-gated destructive intent (pending approval): {(command or '')[:160]}"
        )
        return out

    out["recent_context"] = f"Last command: {(command or '')[:160]}"
    return out


@spaces.GPU
def florence_vision_infer(
    image: Image.Image | None, target_prompt: str
) -> tuple[tuple[Image.Image, list] | None, dict[str, Any]]:
    """Florence-2 UI grounding simulation — ZeroGPU-decorated inference handler."""
    if image is None:
        return None, {
            "error": "No screenshot provided",
            "model": "Florence-2 (Simulated · ZeroGPU)",
        }

    width, height = image.size
    # Bias box toward "error" lines when the prompt asks for errors.
    if "error" in (target_prompt or "").lower() or "error" in str(target_prompt):
        x1, y1 = int(width * 0.04), int(height * 0.38)
        x2, y2 = int(width * 0.62), int(height * 0.52)
    else:
        x1, y1 = int(width * 0.25), int(height * 0.3)
        x2, y2 = int(width * 0.75), int(height * 0.5)
    label = f"Target: '{target_prompt or 'UI Element'}'"
    annotations = [((x1, y1, x2, y2), label)]
    meta = {
        "label": target_prompt or "Detected Element",
        "bounding_box_xyxy": [x1, y1, x2, y2],
        "confidence": 0.94,
        "model": "Florence-2 (Simulated · ZeroGPU)",
        "image_size": [width, height],
    }
    return (image, annotations), meta


def route_command(
    command: str,
    image: Image.Image | None,
    target_prompt: str,
    memory_ledger: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Route a desktop command through vision → memory → HITL → LangGraph trace."""
    cmd = (command or "").strip()
    ledger = copy.deepcopy(
        memory_ledger if isinstance(memory_ledger, dict) else DEFAULT_MEMORY_LEDGER
    )
    if not cmd:
        return (
            "⚪ Node: Idle — enter a desktop command",
            _empty_annotated(),
            {},
            {"error": "empty_command"},
            ledger,
            "Awaiting command.",
        )

    low = cmd.lower()
    needs_vision = any(
        k in low
        for k in ("summarize", "highlight", "window", "screen", "ocr", "vision")
    )
    is_memory = any(
        k in low for k in ("store preference", "remember", "read memory", "recall")
    )
    is_destructive = any(
        k in low for k in ("delete", "restart", "wipe", "daemon", "rm ")
    )

    annotated: Any = None
    vision_meta: dict[str, Any] = {}
    if needs_vision:
        # Auto-load sample desktop when vision examples omit an upload.
        frame = image if image is not None else _sample_desktop()
        target = target_prompt or (
            "errors" if "error" in low else "active window text"
        )
        annotated, vision_meta = florence_vision_infer(frame, target)
        status = "🟢 Node: Vision Grounding"
    elif is_memory:
        status = "🟢 Node: Memory Ledger Read/Write"
        annotated = None
        vision_meta = {"skipped": True, "reason": "memory_intent"}
    else:
        status = "🟡 Node: Intent Inspected"
        annotated = None
        vision_meta = {"skipped": True}

    ledger = _merge_memory_ledger(ledger, cmd, vision_meta=vision_meta)

    risk = "HIGH" if is_destructive else ("LOW" if is_memory and not needs_vision else "MEDIUM")
    ticket = {
        "ticket_id": "TICK-8042",
        "command": cmd,
        "ui_target": target_prompt or "UI Element",
        "proposed_action": (
            "vault_memory_write"
            if is_memory and not is_destructive
            else "win32_system_write / patch_ledger"
        ),
        "risk_level": risk,
        "requiring_approval": True,
        "status": "PENDING_USER_APPROVAL",
        "vision": vision_meta,
        "memory_touch": is_memory
        or any(k in low for k in ("store", "remember", "read memory", "recall")),
    }
    status_hitl = "🟡 Node: HITL Ticket Interrupted"

    trace = {
        "corridor": [
            {"t": "0.0s", "node": "INTAKE", "detail": f"Received prompt: {cmd}"},
            {"t": "0.1s", "node": "ROUTER", "detail": "Evaluating MoA corridor"},
            {
                "t": "0.2s",
                "node": "VISION" if needs_vision else "MEMORY",
                "detail": (
                    "Florence-2 UI grounding (ZeroGPU)"
                    if needs_vision
                    else "Active Memory Ledger touch"
                ),
                "bbox": vision_meta.get("bounding_box_xyxy"),
            },
            {
                "t": "0.3s",
                "node": "LANGGRAPH",
                "detail": "State → inspect_intent → ticket_validate",
            },
            {
                "t": "0.4s",
                "node": "HITL",
                "detail": "Ticket gate interrupted — awaiting Approve / Deny",
            },
        ],
        "active_intent": cmd,
        "halt": False,
        "ticket_validated": True,
        "memory_keys": list((ledger.get("stored_preferences") or {}).keys()),
    }

    note = (
        f"{status} → {status_hitl}\n"
        f"Memory recent_context: {ledger.get('recent_context')}\n"
        "Review the HITL ticket card, then Approve or Deny."
    )
    return status_hitl, annotated, ticket, trace, ledger, note


# Back-compat alias used by older call sites / docs.
run_pipeline = route_command


def resolve_ticket(ticket: dict | None, decision: str) -> tuple[str, dict, str]:
    if not ticket or "status" not in ticket:
        return (
            "⚪ Node: Idle — no active ticket",
            {},
            "No active ticket to resolve.",
        )

    updated = dict(ticket)
    if decision == "Approve":
        updated["status"] = "APPROVED (Executed)"
        status = "🟢 Node: Tools Executed (Fail-Open after HITL)"
        note = f"Ticket {updated.get('ticket_id')} APPROVED — corridor resumed."
    else:
        updated["status"] = "DENIED (Failed Closed)"
        status = "🔴 Node: Denied — Fail Closed"
        note = f"Ticket {updated.get('ticket_id')} DENIED — write aborted."
    return status, updated, note


_THEME = gr.themes.Soft(
    primary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#0b0f17",
    body_background_fill_dark="#0b0f17",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    border_color_primary="#1f2937",
    border_color_primary_dark="#1f2937",
    button_primary_background_fill="#0e7490",
    button_primary_background_fill_dark="#0e7490",
)

with gr.Blocks(
    title="Dānā · Cybernetic Dashboard",
    theme=_THEME,
    css=_CSS,
) as demo:
    with gr.Column(elem_classes=["dana-hero"]):
        gr.Markdown(
            """
# Dānā
Cybernetic control-plane simulator — LangGraph corridor · Florence-2 grounding · HITL fail-closed tickets.

Local by design. This Space is a ZeroGPU-backed preview; download the native Windows app for Distil-Whisper, Win32 ROI, and offline CUDA inference.
"""
        )
        gr.Button(
            "⬇ Download Dānā for Windows",
            link="https://github.com/Cascade-Router/Donna/releases",
            elem_classes=["dana-primary"],
        )

    status_badge = gr.Markdown(
        "⚪ Node: Idle — awaiting desktop command",
        elem_classes=["dana-status"],
    )

    with gr.Row(equal_height=True):
        # ── LEFT: command + vision canvas ──────────────────────────────
        with gr.Column(scale=1, elem_classes=["dana-panel"]):
            gr.Markdown("### Perception · Command & Florence-2 Canvas")
            command_input = gr.Textbox(
                label="Desktop command",
                placeholder="e.g., Summarize active window and create a desktop log ticket",
                value="Summarize active window and create a desktop log ticket",
                lines=2,
            )
            target_input = gr.Textbox(
                label="UI target (OCR / grounding)",
                value="Search Bar",
                placeholder="e.g., Save Button, Search Bar, Close Icon",
            )
            with gr.Row():
                sample_btn = gr.Button("Load sample desktop", size="sm")
                img_input = gr.Image(
                    type="pil",
                    label="Screenshot upload",
                    height=220,
                )
            annotated_out = gr.AnnotatedImage(
                label="Florence-2 bounding boxes",
                height=360,
            )
            submit_btn = gr.Button(
                "Execute corridor",
                variant="primary",
                elem_classes=["dana-primary"],
            )

        # ── RIGHT: status · HITL · memory · LangGraph JSON ─────────────
        with gr.Column(scale=1, elem_classes=["dana-panel"]):
            gr.Markdown(
                "### Pipeline · HITL Ticket · Live Trace",
                elem_classes=["dana-ticket-title"],
            )
            resolution_note = gr.Textbox(
                label="Visual pipeline status",
                value="Ready.",
                lines=3,
                interactive=False,
            )
            memory_ledger = gr.JSON(
                label="🧠 Active Memory Ledger",
                value=copy.deepcopy(DEFAULT_MEMORY_LEDGER),
            )
            gr.Markdown("#### Interactive HITL ticket")
            ticket_card = gr.JSON(label="Ticket payload (Jason review)")
            with gr.Row():
                approve_btn = gr.Button(
                    "Approve",
                    elem_classes=["dana-approve"],
                )
                deny_btn = gr.Button(
                    "Deny",
                    elem_classes=["dana-deny"],
                )
            gr.Markdown("#### Live LangGraph state")
            trace_json = gr.JSON(label="State corridor JSON")

    _pipeline_outputs = [
        status_badge,
        annotated_out,
        ticket_card,
        trace_json,
        memory_ledger,
        resolution_note,
    ]
    # 1-click presets under the command row — fill inputs + run corridor.
    # Providing ``fn`` + ``outputs`` runs the corridor when a preset is clicked
    # (Gradio 5 fills inputs, then invokes ``fn``).
    gr.Examples(
        examples=[
            [
                "Summarize active window text and highlight errors",
                _sample_desktop_path(),
                "errors",
            ],
            [
                "Store preference: Always use PEP8 formatting for python patches",
                None,
                "",
            ],
            [
                "Delete temporary build cache and restart daemon",
                None,
                "",
            ],
        ],
        inputs=[command_input, img_input, target_input],
        outputs=_pipeline_outputs,
        fn=lambda c, i, t: route_command(c, i, t, None),
        cache_examples=False,
        examples_per_page=3,
        label="1-click corridor presets",
    )

    sample_btn.click(fn=_sample_desktop, outputs=[img_input])
    submit_btn.click(
        fn=route_command,
        inputs=[command_input, img_input, target_input, memory_ledger],
        outputs=_pipeline_outputs,
    )
    approve_btn.click(
        fn=lambda t: resolve_ticket(t, "Approve"),
        inputs=[ticket_card],
        outputs=[status_badge, ticket_card, resolution_note],
    )
    deny_btn.click(
        fn=lambda t: resolve_ticket(t, "Deny"),
        inputs=[ticket_card],
        outputs=[status_badge, ticket_card, resolution_note],
    )

if __name__ == "__main__":
    demo.launch()
