"""Dana AI Copilot Sandbox — Gradio entry point.

Three tabs: a ReAct-style tool-broker chat, a CAD blueprint vision + 3D
viewer, and an interactive architecture/plugin explorer. See hf_sandbox/ for
the underlying (mostly real, partly mocked) logic — README.md documents
exactly which parts are real.
"""

from __future__ import annotations

from functools import partial

import gradio as gr
import spaces

from hf_sandbox import agent_bridge, architecture_docs, cad_visualizer, feature_flags

QUICK_PROMPTS = [
    "Build a mounting plate in FreeCAD",
    "Inspect viewport geometry",
    "Check active plugins",
    "System state",
]


def _side_panel_state() -> dict:
    return agent_bridge.TOOL_REGISTRY["system_state"]({})


def _side_panel_plugins() -> dict:
    return agent_bridge.TOOL_REGISTRY["check_plugin_registry"]({})


def _active_tools_view(enabled: dict, pinned: list | None = None) -> dict:
    active = feature_flags.filter_active_tools(enabled, agent_bridge.TOOL_REGISTRY)
    for tool_id in pinned or []:
        owner = feature_flags.tool_id_to_feature(tool_id)
        if owner is None or enabled.get(owner, True):
            active[tool_id] = agent_bridge.TOOL_REGISTRY.get(tool_id)
    return {"active_tool_ids": sorted(active.keys())}


def _on_feature_toggle(feature_id: str, value: bool, state: dict) -> tuple[dict, dict]:
    state = dict(state)
    state[feature_id] = bool(value)
    return state, _active_tools_view(state)


def _on_pin_tool(tool_id: str | None, pinned: list, enabled: dict) -> tuple[list, dict]:
    pinned = list(pinned or [])
    if tool_id and tool_id not in pinned:
        pinned.append(tool_id)
    return pinned, _active_tools_view(enabled, pinned)


def chat_fn(user_text: str, history: list, feature_state: dict):
    history = history or []
    if not user_text or not user_text.strip():
        return history, "", "", _side_panel_state(), _side_panel_plugins()

    turn = agent_bridge.run_turn(user_text, enabled_features=feature_state)
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": turn["assistant_text"]},
    ]

    log_lines = list(turn["reasoning_steps"])
    if turn["tool_result"] is not None:
        log_lines.append(f"payload: {turn['tool_result'].payload}")
    log_text = "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(log_lines))

    return history, "", log_text, _side_panel_state(), _side_panel_plugins()


@spaces.GPU
def analyze_and_preview(image):
    spec = cad_visualizer.parse_blueprint(image)
    stl_path = cad_visualizer.spec_to_stl(spec)
    return spec, stl_path


def regenerate_from_json(spec_json):
    if not isinstance(spec_json, dict) or "shape" not in spec_json:
        spec_json = cad_visualizer.DEFAULT_SPEC
    return cad_visualizer.spec_to_stl(spec_json)


with gr.Blocks(title="Dana AI Copilot Sandbox") as demo:
    gr.Markdown(
        "# 🤖 Dānā — AI Co-Pilot Sandbox\n"
        "Modular multimodal co-pilot: ReAct tool broker · zero-touch plugins · "
        "zero-focus CAD actuation. **This Space mocks OS/FreeCAD binaries** "
        "(no Windows/Win32 in this container) — see [README](README.md) for what's real."
    )

    feature_state = gr.State(value=dict(feature_flags.DEFAULT_ENABLED))
    pinned_tools_state = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("ReAct Co-Pilot Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(type="messages", height=460, label="Dana")
                    msg = gr.Textbox(
                        placeholder="Ask Dana to do something (e.g. 'create a box 40x25x15')...",
                        show_label=False,
                    )
                    with gr.Row():
                        quick_buttons = [gr.Button(p, size="sm") for p in QUICK_PROMPTS]

                with gr.Column(scale=2):
                    with gr.Accordion("Plugin & Feature Manager", open=False):
                        gr.Markdown(
                            "Toggle a feature off and Dana will refuse to dispatch its "
                            "tools and tell you it's disabled — try unchecking FreeCAD "
                            "then asking to build a box."
                        )
                        feature_checkboxes = {
                            feature.id: gr.Checkbox(
                                label=feature.label
                                + ("" if feature.implemented else " (not implemented)"),
                                value=feature_flags.DEFAULT_ENABLED[feature.id],
                                interactive=feature.implemented,
                            )
                            for feature in feature_flags.FEATURES.values()
                        }
                        add_tool_dropdown = gr.Dropdown(
                            choices=list(agent_bridge.TOOL_REGISTRY.keys()),
                            value=None,
                            label="+ Add Existing Feature (pin a tool for this session)",
                        )
                        active_tools_box = gr.JSON(
                            label="Active Tool Registry",
                            value=_active_tools_view(feature_flags.DEFAULT_ENABLED),
                        )
                    gr.Markdown("### Tool Dispatch Log")
                    tool_log = gr.Textbox(lines=10, interactive=False, show_label=False)
                    gr.Markdown("### System State")
                    state_box = gr.JSON(value=_side_panel_state())
                    gr.Markdown("### Active Plugin Registrations")
                    plugins_box = gr.JSON(value=_side_panel_plugins())

            for feature_id, checkbox in feature_checkboxes.items():
                checkbox.change(
                    partial(_on_feature_toggle, feature_id),
                    [checkbox, feature_state],
                    [feature_state, active_tools_box],
                )
            add_tool_dropdown.change(
                _on_pin_tool,
                [add_tool_dropdown, pinned_tools_state, feature_state],
                [pinned_tools_state, active_tools_box],
            )

            chat_outputs = [chatbot, msg, tool_log, state_box, plugins_box]
            msg.submit(chat_fn, [msg, chatbot, feature_state], chat_outputs)
            for button, prompt in zip(quick_buttons, QUICK_PROMPTS):
                button.click(partial(chat_fn, prompt), [chatbot, feature_state], chat_outputs)

        with gr.Tab("CAD Blueprint Vision & 3D Viewer"):
            gr.Markdown(
                "Upload an engineering drawing or a CAD viewport screenshot. "
                "Live multimodal analysis runs if this Space has an "
                "`ANTHROPIC_API_KEY` secret configured; otherwise a labeled "
                "heuristic mock stands in, so the JSON contract is always visible."
            )
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Blueprint / viewport image")
                    analyze_btn = gr.Button("Analyze Blueprint", variant="primary")
                    spec_json = gr.JSON(label="Extracted geometry spec", value=cad_visualizer.DEFAULT_SPEC)
                with gr.Column():
                    model3d = gr.Model3D(label="3D Preview", value=cad_visualizer.default_preview_stl())
                    regen_btn = gr.Button("Regenerate mesh from edited JSON")

            analyze_btn.click(analyze_and_preview, [image_in], [spec_json, model3d])
            regen_btn.click(regenerate_from_json, [spec_json], [model3d])

        with gr.Tab("System Architecture & Plugin Explorer"):
            gr.Markdown(architecture_docs.ARCHITECTURE_OVERVIEW_MD)
            gr.Markdown(architecture_docs.ZERO_FOCUS_MD)
            gr.Markdown(architecture_docs.SAFETY_GATES_MD)

            gr.Markdown("### Plugin manifest explorer (`dana/plugins/*/manifest.json`)")
            with gr.Row():
                plugin_dropdown = gr.Dropdown(
                    choices=architecture_docs.plugin_choices(),
                    value=architecture_docs.plugin_choices()[0],
                    label="Plugin",
                )
            manifest_view = gr.Code(
                value=architecture_docs.plugin_manifest_json(architecture_docs.plugin_choices()[0]),
                language="json",
                label="manifest.json",
            )
            plugin_dropdown.change(architecture_docs.plugin_manifest_json, [plugin_dropdown], [manifest_view])


if __name__ == "__main__":
    demo.launch()
