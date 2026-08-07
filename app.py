"""Dānā Control Plane — Hugging Face Gradio Space (headless Meta-Broker).

No desktop GUI. Prompts route through ``dana.web.headless_bridge`` into an
isolated ``run_meta_broker_isolated`` process with live telemetry streaming.
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
    load_manifest_dict,
    status_label,
)

_RELEASE_URL = "https://github.com/Cascade-Router/Dana/releases"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --dana-bg0: #0d1117;
  --dana-bg1: #010409;
  --dana-panel: #161b22;
  --dana-border: #30363d;
  --dana-cyan: #00f0ff;
  --dana-amber: #ffb000;
  --dana-rose: #ff7b72;
  --dana-text: #e6edf3;
  --dana-muted: #8b949e;
  --dana-radius: 2px;
}

.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
  background:
    linear-gradient(180deg, var(--dana-bg0) 0%, var(--dana-bg1) 100%) !important;
  color: var(--dana-text) !important;
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding-bottom: 2rem !important;
}
.gradio-container, .gradio-container * {
  box-shadow: none !important;
}
footer { display: none !important; }

.dana-hero {
  padding: 1.35rem 0.25rem 0.85rem 0.25rem !important;
  border-bottom: 1px solid var(--dana-border);
  margin-bottom: 1rem !important;
}
.dana-hero h1 {
  font-size: clamp(2rem, 4.5vw, 2.85rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  color: #f0f6fc !important;
  margin: 0 0 0.35rem 0 !important;
  line-height: 1.1 !important;
}
.dana-hero .dana-sub {
  color: var(--dana-muted) !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.02em !important;
  margin: 0 0 0.75rem 0 !important;
}
.dana-hero .dana-tag {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  color: var(--dana-cyan) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  margin: 0 0 0.55rem 0 !important;
}
.dana-hero-row {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.65rem 0.85rem !important;
  align-items: center !important;
}

.dana-status-pill {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.85rem !important;
  border-radius: var(--dana-radius) !important;
  border: 1px solid var(--dana-border) !important;
  background: var(--dana-panel) !important;
  color: var(--dana-text) !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.45rem !important;
  min-width: 11.5rem !important;
  justify-content: center !important;
}
.dana-status-pill .dot {
  width: 0.5rem; height: 0.5rem; border-radius: 0;
  background: #6e7681; display: inline-block;
}
.dana-status-pill.idle .dot { background: #6e7681; }
.dana-status-pill.listening .dot { background: var(--dana-cyan); }
.dana-status-pill.processing .dot {
  background: var(--dana-amber);
  animation: dana-pulse 1.1s ease-in-out infinite;
}
.dana-status-pill.epic_executing .dot {
  background: var(--dana-cyan);
  animation: dana-pulse 0.85s ease-in-out infinite;
}
.dana-status-pill.idle { color: var(--dana-muted) !important; }
.dana-status-pill.listening { color: var(--dana-cyan) !important; border-color: #1f6f78 !important; }
.dana-status-pill.processing { color: var(--dana-amber) !important; border-color: #9a6700 !important; }
.dana-status-pill.epic_executing { color: var(--dana-cyan) !important; border-color: #1f6f78 !important; }
.dana-status-pill.pending_approval { color: var(--dana-amber) !important; border-color: #9a6700 !important; }
.dana-status-pill.pending_approval .dot {
  background: var(--dana-amber);
  box-shadow: 0 0 8px rgba(255, 176, 0, 0.55);
}

@keyframes dana-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

.dana-panel {
  background: var(--dana-panel) !important;
  border: 1px solid var(--dana-border) !important;
  border-radius: var(--dana-radius) !important;
  padding: 0.85rem 0.95rem 1rem 0.95rem !important;
}
.dana-panel h3 {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  color: var(--dana-cyan) !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  margin: 0 0 0.65rem 0 !important;
}

/* Terminal surfaces: broker, tracker, telemetry, manifest */
.dana-mono textarea,
.dana-mono input,
.dana-mono .prose,
.dana-broker textarea,
.dana-broker input,
.dana-panel textarea,
.dana-panel [data-testid="json"] {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.78rem !important;
  line-height: 1.55 !important;
  background: #0d1117 !important;
  border: 1px solid var(--dana-border) !important;
  border-radius: var(--dana-radius) !important;
  color: #c9d1d9 !important;
}

button.primary, button.secondary, button.dana-primary, button.dana-stop,
button.dana-ghost, button.dana-download, .gr-button {
  border-radius: var(--dana-radius) !important;
  box-shadow: none !important;
}
button.dana-primary {
  background: transparent !important;
  border: 1px solid var(--dana-cyan) !important;
  color: var(--dana-cyan) !important;
  font-weight: 600 !important;
}
button.dana-primary:hover {
  background: rgba(0, 240, 255, 0.08) !important;
}
button.dana-stop {
  background: transparent !important;
  border: 1px solid var(--dana-rose) !important;
  color: var(--dana-rose) !important;
  font-weight: 600 !important;
}
button.dana-ghost {
  background: transparent !important;
  border: 1px solid var(--dana-border) !important;
  color: var(--dana-text) !important;
}
button.dana-download {
  background: transparent !important;
  border: 1px solid var(--dana-amber) !important;
  color: var(--dana-amber) !important;
  font-weight: 600 !important;
}

.gradio-container .accordion,
.gradio-container details {
  border: 1px solid var(--dana-border) !important;
  border-radius: var(--dana-radius) !important;
  background: #0d1117 !important;
}

@media (max-width: 768px) {
  .gradio-container { max-width: 100% !important; }
  .dana-hero h1 { font-size: 1.85rem !important; }
}
"""

_THEME = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_none,
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="#30363d",
    block_border_color_dark="#30363d",
    border_color_primary="#30363d",
    border_color_primary_dark="#30363d",
    button_primary_background_fill="#161b22",
    button_primary_background_fill_dark="#161b22",
    button_primary_text_color="#00f0ff",
    button_primary_text_color_dark="#00f0ff",
    button_primary_border_color="#00f0ff",
    button_primary_border_color_dark="#00f0ff",
    shadow_drop="none",
    shadow_drop_lg="none",
    shadow_spread="0px",
)


def _status_html(status: str) -> str:
    st = str(status or "idle").strip().lower()
    if st in {"pending_approval", "pending_user_approval", "awaiting_approval"}:
        st = "pending_approval"
    elif st in {"dispatch_epic", "feedback"} or (
        "epic" in st and "pending" not in st
    ):
        st = "epic_executing"
    elif st in {"routing", "executing", "planning"}:
        st = "processing"
    if st not in {
        "idle",
        "listening",
        "processing",
        "epic_executing",
        "pending_approval",
    }:
        st = "idle"
    label = status_label(st)
    return (
        f'<div class="dana-status-pill {st}">'
        f'<span class="dot"></span><span>{label}</span></div>'
    )


def _approval_markdown() -> str:
    pending = get_bridge().pending_approval()
    if not pending:
        return (
            "_No compiled spec awaiting approval. "
            "Submit a plain-English goal or `/broker` macro to draft one._"
        )
    spec = str(pending.get("compiled_spec") or "").strip()
    epics = list(pending.get("epics") or [])
    return (
        "### Spec Approval — `PENDING_USER_APPROVAL`\n\n"
        f"**Epics:** {len(epics)}\n\n"
        f"```\n{spec or '(empty)'}\n```\n\n"
        "Click **Approve & Run** to dispatch Meta-Broker, "
        "**Edit Macro** to copy into the prompt box, or **Cancel**."
    )


def _chat_append_user(history: list | None, text: str) -> list:
    hist = list(history or [])
    if hist and isinstance(hist[0], dict):
        return hist + [{"role": "user", "content": text}]
    return hist + [[text, None]]


def _chat_append_assistant(history: list | None, text: str) -> list:
    hist = list(history or [])
    if not hist:
        if text:
            return [{"role": "assistant", "content": text}]
        return hist
    if isinstance(hist[0], dict):
        return hist + [{"role": "assistant", "content": text}]
    # Tuple/list pairs: fill empty assistant slot or append.
    last = hist[-1]
    if isinstance(last, (list, tuple)) and len(last) >= 2 and last[1] in (None, ""):
        hist = hist[:-1] + [[last[0], text]]
        return hist
    return hist + [[None, text]]


def _panels(selected: str | None = None) -> tuple[Any, ...]:
    bridge = get_bridge()
    _ = bridge.drain_telemetry(max_items=48)
    snap = bridge.snapshot()
    choices = bridge.epic_choices()
    value = selected if selected in choices else (choices[0] if choices else None)
    label, code = bridge.workspace_viewer()
    pending = bool(bridge.pending_approval())
    return (
        _status_html(str(snap.get("status") or "idle")),
        gr.update(choices=choices, value=value),
        bridge.epic_detail_markdown(value),
        bridge.log_text(),
        label,
        code,
        load_manifest_dict(),
        _approval_markdown(),
        gr.update(interactive=pending),
        gr.update(interactive=pending),
        gr.update(interactive=pending),
    )


def submit_command(
    message: str,
    force_local: bool,
    verbose: bool,
    history: list | None,
) -> Generator[tuple[Any, ...], None, None]:
    """Submit → background Meta-Broker; yield live panels. Prompt logged once."""
    bridge = get_bridge()
    bridge.configure(force_local=bool(force_local), verbose=bool(verbose))
    text = (message or "").strip()
    hist = list(history or [])

    def _pack(
        *,
        clear_prompt: bool = False,
        hist_out: list | None = None,
        selected: str | None = None,
        prompt_value: str | None = None,
    ) -> tuple[Any, ...]:
        (
            st,
            epic_dd,
            epic_md,
            log,
            wlabel,
            wcode,
            man,
            approval_md,
            appr_btn,
            edit_btn,
            cancel_btn,
        ) = _panels(selected)
        out_prompt = "" if clear_prompt else (
            text if prompt_value is None else prompt_value
        )
        return (
            st,
            epic_dd,
            epic_md,
            log,
            wlabel,
            wcode,
            man,
            approval_md,
            appr_btn,
            edit_btn,
            cancel_btn,
            out_prompt,
            hist_out if hist_out is not None else hist,
        )

    if not text:
        yield _pack()
        return

    # Append the user prompt EXACTLY ONCE to chat history.
    hist = _chat_append_user(hist, text)

    ok, note = bridge.submit_prompt(text)
    if not ok:
        hist = _chat_append_assistant(hist, note)
        yield _pack(hist_out=hist)
        return

    hist = _chat_append_assistant(hist, note)
    yield _pack(clear_prompt=True, hist_out=hist)

    deadline = time.time() + float(os.environ.get("DONNA_META_BROKER_TIMEOUT_S") or "600")
    while bridge.is_running and time.time() < deadline:
        yield _pack(clear_prompt=True, hist_out=hist)
        time.sleep(0.4)

    (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
    ) = _panels()
    snap = bridge.snapshot()
    if snap.get("error"):
        log = log + f"\n[ui] finished error={snap.get('error')}"
        hist = _chat_append_assistant(hist, f"Finished with error: {snap.get('error')}")
    elif snap.get("result_status"):
        log = log + f"\n[ui] finished status={snap.get('result_status')}"
        hist = _chat_append_assistant(hist, f"Finished: {snap.get('result_status')}")
    yield (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
        "",
        hist,
    )


def stop_execution(
    history: list | None = None,
    selected: str | None = None,
) -> tuple[Any, ...]:
    bridge = get_bridge()
    ok, note = bridge.stop()
    (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
    ) = _panels(selected)
    if ok:
        log = log + f"\n[ui] {note}"
    hist = _chat_append_assistant(history, note) if note else list(history or [])
    return (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
        hist,
    )


def poll_live(selected: str | None = None) -> tuple[Any, ...]:
    return _panels(selected)


def on_epic_select(choice: str) -> str:
    return get_bridge().epic_detail_markdown(choice)


def approve_spec_action(
    history: list | None = None,
    selected: str | None = None,
) -> tuple[Any, ...]:
    bridge = get_bridge()
    ok, note = bridge.approve_spec()
    (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
    ) = _panels(selected)
    if ok:
        log = log + f"\n[ui] {note}"
    hist = _chat_append_assistant(history, note) if note else list(history or [])
    return (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
        hist,
    )


def edit_spec_action(
    history: list | None = None,
    selected: str | None = None,
) -> tuple[Any, ...]:
    bridge = get_bridge()
    pending = bridge.pending_approval() or {}
    macro = str(pending.get("compiled_spec") or "").strip()
    ok, note = bridge.cancel_spec()
    (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
    ) = _panels(selected)
    msg = "Macro copied to prompt — edit, then Submit."
    if ok:
        log = log + f"\n[ui] {note}; {msg}"
    hist = _chat_append_assistant(history, msg)
    return (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
        macro,
        hist,
    )


def cancel_spec_action(
    history: list | None = None,
    selected: str | None = None,
) -> tuple[Any, ...]:
    bridge = get_bridge()
    ok, note = bridge.cancel_spec()
    (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
    ) = _panels(selected)
    if ok:
        log = log + f"\n[ui] {note}"
    hist = _chat_append_assistant(history, note) if note else list(history or [])
    return (
        st,
        epic_dd,
        epic_md,
        log,
        wlabel,
        wcode,
        man,
        approval_md,
        appr_btn,
        edit_btn,
        cancel_btn,
        hist,
    )


with gr.Blocks(
    title="Dānā Control Plane",
    theme=_THEME,
    css=_CSS,
) as demo:
    with gr.Column(elem_classes=["dana-hero"]):
        gr.Markdown(
            """
<p class="dana-tag">Cybernetic Control-Plane Simulator</p>
# Dānā Control Plane
<p class="dana-sub">Multi-Agent Orchestration · Meta-Broker Corridor · Cascade Router</p>
"""
        )
        with gr.Row(elem_classes=["dana-hero-row"]):
            status_html = gr.HTML(_status_html("idle"))
            gr.Button(
                "⬇ Download Dānā for Windows",
                link=_RELEASE_URL,
                elem_classes=["dana-download"],
            )

    with gr.Row(equal_height=False):
        # ── Left: command / chat (prompt appears once) ─────────────────
        with gr.Column(scale=5, elem_classes=["dana-panel"]):
            gr.Markdown("### Command & Broker")
            chat_hist = gr.Chatbot(
                label="Session",
                value=[],
                height=220,
                type="messages",
                elem_classes=["dana-mono"],
            )
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder=(
                    "/broker Epic 1: … Epic 2: … — or describe a multi-epic goal"
                ),
                lines=5,
                max_lines=12,
                autofocus=True,
                elem_classes=["dana-broker", "dana-mono"],
            )
            with gr.Row():
                force_local = gr.Checkbox(
                    label="DONNA_FORCE_LOCAL",
                    value=True,
                    info="Skip cloud / Gemini hops (local Ollama only)",
                )
                verbose = gr.Checkbox(
                    label="Verbose telemetry",
                    value=False,
                    info="Set DONNA_DEBUG=1 for richer logs",
                )
            with gr.Row():
                submit_btn = gr.Button(
                    "Submit Command",
                    variant="primary",
                    elem_classes=["dana-primary"],
                )
                stop_btn = gr.Button(
                    "Stop Execution",
                    elem_classes=["dana-stop"],
                )
            gr.Markdown("### Spec Approval Card")
            approval_md = gr.Markdown(
                value=(
                    "_No compiled spec awaiting approval. "
                    "Submit a plain-English goal or `/broker` macro to draft one._"
                )
            )
            with gr.Row():
                approve_btn = gr.Button(
                    "Approve & Run",
                    variant="primary",
                    interactive=False,
                    elem_classes=["dana-primary"],
                )
                edit_macro_btn = gr.Button("Edit Macro", interactive=False)
                cancel_spec_btn = gr.Button(
                    "Cancel",
                    interactive=False,
                    elem_classes=["dana-stop"],
                )
            gr.Examples(
                examples=[
                    [
                        "/broker Epic 1: Write hello_util.py with greet(name). "
                        "Epic 2: Write tests/test_hello_util.py asserting greet returns a string."
                    ],
                    [
                        "/broker Epic 1: Write rate_limiter.py with a TokenBucket class. "
                        "Epic 2: Write tests/test_rate_limiter.py for refill and deny paths."
                    ],
                ],
                inputs=[prompt_box],
                label="Example /broker prompts",
            )

        # ── Right: interactive tracker ─────────────────────────────────
        with gr.Column(scale=6, elem_classes=["dana-panel"]):
            gr.Markdown("### Epic Task Tracker")
            epic_radio = gr.Radio(
                choices=["(no epics yet)"],
                value="(no epics yet)",
                label="Select an epic for details",
                elem_classes=["dana-mono"],
            )
            epic_detail = gr.Markdown(
                value="_Submit a `/broker` command to populate epics._"
            )
            with gr.Accordion("Artifact Manifest (.dana_scratch/manifest.json)", open=False):
                manifest_json = gr.JSON(
                    label="Exports",
                    value=load_manifest_dict(),
                )

    # ── DAG / Workspace split-view ─────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["dana-panel"]):
            gr.Markdown("### DAG · Broker Telemetry")
            telemetry_box = gr.Textbox(
                label="IPC / stdout stream",
                value="(no telemetry yet)",
                lines=16,
                max_lines=28,
                interactive=False,
                elem_classes=["dana-mono"],
                autoscroll=True,
            )
        with gr.Column(scale=1, elem_classes=["dana-panel"]):
            gr.Markdown("### Live Workspace Viewer")
            workspace_label = gr.Textbox(
                label="Active file",
                value="(waiting for artifacts…)",
                interactive=False,
                elem_classes=["dana-mono"],
            )
            try:
                workspace_code = gr.Code(
                    label="Source",
                    value="# Generated files stream here during Meta-Broker runs.\n",
                    language="python",
                    lines=16,
                    interactive=False,
                )
            except TypeError:
                workspace_code = gr.Textbox(
                    label="Source",
                    value="# Generated files stream here during Meta-Broker runs.\n",
                    lines=16,
                    max_lines=28,
                    interactive=False,
                    elem_classes=["dana-mono"],
                )

    gr.Markdown(
        "<p style='color:#8b949e;font-size:0.85rem;margin-top:0.85rem'>"
        "Headless corridor: <code>headless_bridge</code> → "
        "<code>run_meta_broker_isolated</code>. "
        "Prompts appear once in Session. Click an epic to inspect status / harness."
        "</p>"
    )

    stream_outs = [
        status_html,
        epic_radio,
        epic_detail,
        telemetry_box,
        workspace_label,
        workspace_code,
        manifest_json,
        approval_md,
        approve_btn,
        edit_macro_btn,
        cancel_spec_btn,
        prompt_box,
        chat_hist,
    ]
    panel_outs = [
        status_html,
        epic_radio,
        epic_detail,
        telemetry_box,
        workspace_label,
        workspace_code,
        manifest_json,
        approval_md,
        approve_btn,
        edit_macro_btn,
        cancel_spec_btn,
    ]
    submit_btn.click(
        fn=submit_command,
        inputs=[prompt_box, force_local, verbose, chat_hist],
        outputs=stream_outs,
    )
    prompt_box.submit(
        fn=submit_command,
        inputs=[prompt_box, force_local, verbose, chat_hist],
        outputs=stream_outs,
    )
    stop_btn.click(
        fn=stop_execution,
        inputs=[chat_hist, epic_radio],
        outputs=panel_outs + [chat_hist],
    )
    approve_btn.click(
        fn=approve_spec_action,
        inputs=[chat_hist, epic_radio],
        outputs=panel_outs + [chat_hist],
    )
    edit_macro_btn.click(
        fn=edit_spec_action,
        inputs=[chat_hist, epic_radio],
        outputs=panel_outs + [prompt_box, chat_hist],
    )
    cancel_spec_btn.click(
        fn=cancel_spec_action,
        inputs=[chat_hist, epic_radio],
        outputs=panel_outs + [chat_hist],
    )
    epic_radio.change(
        fn=on_epic_select,
        inputs=[epic_radio],
        outputs=[epic_detail],
    )

    try:
        timer = gr.Timer(1.0)
        timer.tick(
            fn=poll_live,
            inputs=[epic_radio],
            outputs=panel_outs,
        )
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    assert_no_tkinter_loaded()
    demo.queue(default_concurrency_limit=2)
    demo.launch()
