"""Dānā · Hugging Face Space — headless Meta-Broker Gradio dashboard.

No Tkinter. Prompts run via ``run_meta_broker_isolated`` on a background thread;
telemetry is polled into the Status Indicator and Task Tracker panels.
"""

from __future__ import annotations

import os
import time
from typing import Any, Generator

# Headless flags before any Dānā imports.
os.environ.setdefault("DONNA_NO_GUI", "1")
os.environ.setdefault("DONNA_HEADLESS", "1")
os.environ.setdefault("DONNA_SKIP_BOOT_READY", "1")

import gradio as gr

from dana.web.headless_bridge import (
    assert_no_tkinter_loaded,
    get_bridge,
    status_label,
)

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --dana-bg0: #070b12;
  --dana-bg1: #0c1220;
  --dana-panel: rgba(17, 24, 39, 0.72);
  --dana-border: #1e293b;
  --dana-cyan: #22d3ee;
  --dana-emerald: #34d399;
  --dana-amber: #fbbf24;
  --dana-text: #e2e8f0;
  --dana-muted: #94a3b8;
}

.gradio-container {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(34, 211, 238, 0.12), transparent 55%),
    radial-gradient(900px 500px at 90% 0%, rgba(16, 185, 129, 0.08), transparent 50%),
    linear-gradient(180deg, var(--dana-bg0), var(--dana-bg1)) !important;
  color: var(--dana-text) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
}
footer { display: none !important; }

.dana-hero {
  padding: 1.25rem 0.25rem 0.5rem 0.25rem !important;
}
.dana-hero h1 {
  font-size: clamp(2.4rem, 5vw, 3.4rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.04em !important;
  color: #f8fafc !important;
  margin: 0 0 0.35rem 0 !important;
  line-height: 1.05 !important;
}
.dana-hero p {
  color: var(--dana-muted) !important;
  font-size: 1.05rem !important;
  max-width: 42rem !important;
  margin: 0 !important;
}

.dana-status-pill {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  padding: 0.7rem 1rem !important;
  border-radius: 999px !important;
  border: 1px solid var(--dana-border) !important;
  background: rgba(15, 23, 42, 0.9) !important;
  color: #e2e8f0 !important;
  display: inline-block !important;
  min-width: 11rem !important;
  text-align: center !important;
}
.dana-status-pill.idle { color: #94a3b8 !important; border-color: #334155 !important; }
.dana-status-pill.listening { color: var(--dana-emerald) !important; border-color: #065f46 !important; }
.dana-status-pill.processing { color: var(--dana-amber) !important; border-color: #92400e !important; }

.dana-panel {
  background: var(--dana-panel) !important;
  border: 1px solid var(--dana-border) !important;
  border-radius: 16px !important;
  padding: 0.85rem 1rem 1rem 1rem !important;
  backdrop-filter: blur(8px);
}
.dana-panel h3, .dana-panel .label-wrap span {
  color: #f1f5f9 !important;
  font-weight: 600 !important;
}
.dana-mono textarea, .dana-mono input {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-size: 0.82rem !important;
  line-height: 1.45 !important;
}
button.dana-primary {
  background: linear-gradient(135deg, #0891b2, #0e7490) !important;
  border: none !important;
  color: #ecfeff !important;
  font-weight: 650 !important;
}
button.dana-ghost {
  background: transparent !important;
  border: 1px solid var(--dana-border) !important;
  color: var(--dana-text) !important;
}
"""

_THEME = gr.themes.Soft(
    primary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#070b12",
    body_background_fill_dark="#070b12",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    border_color_primary="#1e293b",
    border_color_primary_dark="#1e293b",
    button_primary_background_fill="#0e7490",
    button_primary_background_fill_dark="#0e7490",
)


def _status_markdown(status: str) -> str:
    st = str(status or "idle").strip().lower()
    if st in {"routing", "executing", "planning"}:
        st = "processing"
    if st not in {"idle", "listening", "processing"}:
        st = "idle"
    label = status_label(st)
    return f'<div class="dana-status-pill {st}">{label}</div>'


def _assistant_reply(bridge_snap: dict[str, Any]) -> str:
    if bridge_snap.get("running"):
        return "Meta-Broker is running in an isolated process. Telemetry is streaming…"
    err = str(bridge_snap.get("error") or "").strip()
    rs = str(bridge_snap.get("result_status") or "").strip()
    if err:
        return f"Meta-Broker finished with error ({rs or 'failed'}):\n{err}"
    if rs:
        return f"Meta-Broker finished — status=`{rs}`.\nSee Task Tracker / telemetry for epic progress."
    return "Done."


def submit_prompt(
    message: str,
    history: list[dict[str, str]] | None,
) -> Generator[tuple[Any, ...], None, None]:
    """Non-blocking submit: background Meta-Broker + live telemetry yields."""
    history = list(history or [])
    text = (message or "").strip()
    if not text:
        bridge = get_bridge()
        yield (
            history,
            "",
            _status_markdown(bridge.status()),
            bridge.task_tracker_text(),
            bridge.log_text(),
        )
        return

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "Dispatching isolated Meta-Broker…"},
    ]
    bridge = get_bridge()
    ok, note = bridge.submit(text)
    if not ok:
        history[-1] = {"role": "assistant", "content": note}
        yield (
            history,
            "",
            _status_markdown(bridge.status()),
            bridge.task_tracker_text(),
            bridge.log_text(),
        )
        return

    yield (
        history,
        "",
        _status_markdown("processing"),
        bridge.task_tracker_text(),
        bridge.log_text() + f"\n[ui] {note}",
    )

    # Poll telemetry until the background worker completes (Gradio generator stream).
    deadline = time.time() + float(os.environ.get("DONNA_META_BROKER_TIMEOUT_S") or "600")
    last_log = ""
    while bridge.is_running and time.time() < deadline:
        _ = bridge.drain_telemetry(max_items=64)
        snap = bridge.snapshot()
        log = bridge.log_text()
        tracker = bridge.task_tracker_text()
        if log != last_log:
            history[-1] = {
                "role": "assistant",
                "content": (
                    "Meta-Broker running…\n\n"
                    + "\n".join(snap.get("log_lines") or [])[-1200:]
                ),
            }
            last_log = log
        yield (
            history,
            "",
            _status_markdown(snap.get("status") or "processing"),
            tracker,
            log,
        )
        time.sleep(0.35)

    # Final drain.
    _ = bridge.drain_telemetry(max_items=128)
    snap = bridge.snapshot()
    history[-1] = {"role": "assistant", "content": _assistant_reply(snap)}
    yield (
        history,
        "",
        _status_markdown(snap.get("status") or "idle"),
        bridge.task_tracker_text(),
        bridge.log_text(),
    )


def poll_panels() -> tuple[str, str, str]:
    """Timer tick: refresh status / tracker / log without submitting."""
    bridge = get_bridge()
    _ = bridge.drain_telemetry(max_items=32)
    return (
        _status_markdown(bridge.status()),
        bridge.task_tracker_text(),
        bridge.log_text(),
    )


def clear_chat() -> tuple[list, str, str, str]:
    bridge = get_bridge()
    return (
        [],
        _status_markdown(bridge.status()),
        bridge.task_tracker_text(),
        bridge.log_text(),
    )


# Optional ZeroGPU stub kept for Space compatibility (vision demos retired).
@spaces.GPU
def _gpu_warmup() -> str:
    return "ok"


with gr.Blocks(
    title="Dānā · Headless Control Plane",
    theme=_THEME,
    css=_CSS,
) as demo:
    with gr.Column(elem_classes=["dana-hero"]):
        gr.Markdown(
            """
# Dānā
Headless cybernetic control plane — isolated Meta-Broker, live telemetry, stdlib-first epics.
"""
        )

    status_html = gr.HTML(
        _status_markdown("idle"),
        elem_classes=["dana-status-wrap"],
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, elem_classes=["dana-panel"]):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(
                label="Session",
                height=440,
                type="messages",
                show_copy_button=True,
            )
            with gr.Row():
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        "/broker Epic 1: …  — or describe a multi-epic goal"
                    ),
                    lines=3,
                    scale=5,
                    autofocus=True,
                )
            with gr.Row():
                send_btn = gr.Button(
                    "Run Meta-Broker",
                    variant="primary",
                    elem_classes=["dana-primary"],
                )
                clear_btn = gr.Button("Clear", elem_classes=["dana-ghost"])

        with gr.Column(scale=2, elem_classes=["dana-panel"]):
            gr.Markdown("### Task Tracker")
            tracker_box = gr.Textbox(
                label="Activities",
                value="(no active tasks)",
                lines=12,
                max_lines=20,
                interactive=False,
                elem_classes=["dana-mono"],
            )
            gr.Markdown("### Telemetry")
            telemetry_box = gr.Textbox(
                label="IPC / Meta-Broker log",
                value="(no telemetry yet)",
                lines=12,
                max_lines=24,
                interactive=False,
                elem_classes=["dana-mono"],
            )

    gr.Examples(
        examples=[
            [
                "/broker Epic 1: Write hello_util.py with a greet(name) function. "
                "Epic 2: Write tests/test_hello_util.py that asserts greet returns a string."
            ],
            [
                "Plan and implement a small JSON key-value store with pytest coverage"
            ],
        ],
        inputs=[prompt_box],
        label="Example Meta-Broker prompts",
    )

    gr.Markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:0.75rem'>"
        "Runs <code>run_meta_broker_isolated</code> (multiprocessing) — no Tkinter. "
        "Local Ollama (<code>qwen2.5-coder:7b</code>) required for real codegen; "
        "otherwise telemetry will show the failure path."
        "</p>"
    )

    outputs = [chatbot, prompt_box, status_html, tracker_box, telemetry_box]
    send_btn.click(
        fn=submit_prompt,
        inputs=[prompt_box, chatbot],
        outputs=outputs,
    )
    prompt_box.submit(
        fn=submit_prompt,
        inputs=[prompt_box, chatbot],
        outputs=outputs,
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, status_html, tracker_box, telemetry_box],
    )

    # Background refresh while a job runs (and idle keepalive).
    try:
        timer = gr.Timer(1.0)
        timer.tick(
            fn=poll_panels,
            outputs=[status_html, tracker_box, telemetry_box],
        )
    except Exception:  # noqa: BLE001 — older Gradio without Timer
        pass


if __name__ == "__main__":
    assert_no_tkinter_loaded()
    demo.queue(default_concurrency_limit=2)
    demo.launch()
