"""Dānā · Hugging Face Space — tabless dark cybernetic dashboard."""

from __future__ import annotations

import json
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
    draw.text((56, 220), "Summarize active window…", fill="#cbd5e1")
    return img


def _empty_annotated() -> tuple[None, list]:
    return None, []


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


def run_pipeline(
    command: str, image: Image.Image | None, target_prompt: str
) -> tuple[Any, ...]:
    """Single submit: vision grounding → HITL ticket → LangGraph JSON trace."""
    cmd = (command or "").strip()
    if not cmd:
        return (
            "⚪ Node: Idle — enter a desktop command",
            _empty_annotated()[0],
            {},
            {"error": "empty_command"},
            "Awaiting command.",
        )

    # Stage 1 — vision
    annotated, vision_meta = florence_vision_infer(image, target_prompt)
    status_vision = "🟢 Node: Vision Grounding"

    # Stage 2 — HITL interrupt
    ticket = {
        "ticket_id": "TICK-8042",
        "command": cmd,
        "ui_target": target_prompt or "UI Element",
        "proposed_action": "win32_system_write / patch_ledger",
        "risk_level": "MEDIUM",
        "requiring_approval": True,
        "status": "PENDING_USER_APPROVAL",
        "vision": vision_meta,
    }
    status_hitl = "🟡 Node: HITL Ticket Interrupted"

    trace = {
        "corridor": [
            {"t": "0.0s", "node": "INTAKE", "detail": f"Received prompt: {cmd}"},
            {"t": "0.1s", "node": "ROUTER", "detail": "Evaluating MoA corridor"},
            {
                "t": "0.2s",
                "node": "VISION",
                "detail": "Florence-2 UI grounding (ZeroGPU)",
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
    }

    note = (
        f"{status_vision} → {status_hitl}\n"
        "Review the HITL ticket card, then Approve or Deny."
    )
    return status_hitl, annotated, ticket, trace, note


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

        # ── RIGHT: status · HITL · LangGraph JSON ─────────────────────
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

    sample_btn.click(fn=_sample_desktop, outputs=[img_input])
    submit_btn.click(
        fn=run_pipeline,
        inputs=[command_input, img_input, target_input],
        outputs=[
            status_badge,
            annotated_out,
            ticket_card,
            trace_json,
            resolution_note,
        ],
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
