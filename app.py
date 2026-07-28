import gradio as gr
from PIL import Image
import json

# ZeroGPU Spaces preinstall `spaces`; local CPU simulator does not require it.
try:
    import spaces  # noqa: F401
except ImportError:  # pragma: no cover - optional on non-HF runtimes
    spaces = None

# --- Tab 1: LangGraph Routing Logic & HITL Gate ---
def simulate_routing(user_prompt):
    if not user_prompt.strip():
        return "Please enter a command.", {}, "Awaiting Input"
    
    state_log = (
        f"0.0s [INTAKE] Received prompt: '{user_prompt}'\n"
        f"0.1s [ROUTER] Evaluating MoA corridor...\n"
        f"0.2s [LANGGRAPH] State transitioning -> Node: inspect_intent\n"
        f"0.3s [SAFETY] HITL Ticket required before win32 patch write."
    )
    
    ticket = {
        "ticket_id": "TICK-8042",
        "command": user_prompt,
        "proposed_action": "win32_system_write / patch_ledger",
        "risk_level": "MEDIUM",
        "requiring_approval": True,
        "status": "PENDING_USER_APPROVAL"
    }
    
    return state_log, ticket, "🎫 Ticket Generated — Review Needed"

def resolve_ticket(ticket, decision):
    if not ticket or "status" not in ticket:
        return "No active ticket to resolve.", {}
    
    ticket["status"] = "APPROVED (Executed)" if decision == "Approve" else "DENIED (Failed Closed)"
    result_log = f"Ticket {ticket['ticket_id']} has been {ticket['status']}."
    return result_log, ticket

# --- Tab 2: Vision & Screen Grounding Logic ---
def simulate_vision_grounding(image, target_prompt):
    if image is None:
        return None, "Please upload or select an image."
    
    width, height = image.size
    x1, y1 = int(width * 0.25), int(height * 0.3)
    x2, y2 = int(width * 0.75), int(height * 0.5)
    
    label = f"Target: '{target_prompt or 'UI Element'}'"
    annotations = [((x1, y1, x2, y2), label)]
    
    bbox_info = {
        "label": target_prompt or "Detected Element",
        "bounding_box_xyxy": [x1, y1, x2, y2],
        "confidence": 0.94,
        "model": "Florence-2 (Simulated on CPU)"
    }
    
    return (image, annotations), json.dumps(bbox_info, indent=2)

# --- Gradio Blocks UI ---
with gr.Blocks(title="Dānā · Agent Simulator") as demo:
    gr.Markdown(
        """
        # Dānā · Cybernetic Agent Simulator
        
        > **Local by design.** This Hugging Face Space runs a lightweight CPU simulation of Dānā's **LangGraph routing corridor** and **Florence-2 UI grounding logic**.
        > 
        > ⚡ *To execute local Win32 hardware actions, Distil-Whisper speech, and CUDA float16 offline inference with 0 cloud latency, download the native Windows app below.*
        """
    )
    
    gr.Button("⬇️ Download Dānā for Windows (GitHub)", link="https://github.com/Cascade-Router/Donna/releases")
    
    with gr.Tabs():
        # Tab 1: LangGraph Router & HITL Gate
        with gr.Tab("1. LangGraph Routing & HITL Gate"):
            gr.Markdown("### Test the Human-In-The-Loop (HITL) Ticket Gate")
            prompt_input = gr.Textbox(
                label="Enter a Desktop Automation Command",
                placeholder="e.g., Summarize active window text and draft a local report",
                value="Summarize active window text and draft a local report"
            )
            route_btn = gr.Button("Route Command Through LangGraph", variant="primary")
            
            trace_output = gr.Textbox(label="LangGraph State Corridor Trace", lines=4)
            ticket_output = gr.JSON(label="Generated HITL Ticket (Jason Review)")
            status_banner = gr.Markdown("### Status: Idle")
            
            with gr.Row():
                approve_btn = gr.Button("✅ Approve Ticket", variant="success")
                deny_btn = gr.Button("❌ Deny Ticket (Fail Closed)", variant="stop")
            
            resolution_output = gr.Textbox(label="Ticket Resolution Result")
            
            route_btn.click(
                simulate_routing,
                inputs=[prompt_input],
                outputs=[trace_output, ticket_output, status_banner]
            )
            approve_btn.click(
                lambda t: resolve_ticket(t, "Approve"),
                inputs=[ticket_output],
                outputs=[resolution_output, ticket_output]
            )
            deny_btn.click(
                lambda t: resolve_ticket(t, "Deny"),
                inputs=[ticket_output],
                outputs=[resolution_output, ticket_output]
            )

        # Tab 2: Florence-2 Vision Sandbox
        with gr.Tab("2. Florence-2 Vision & UI Grounding"):
            gr.Markdown("### Screen ROI & UI Element Grounding")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Desktop Screenshot")
                    target_input = gr.Textbox(
                        label="UI Target Prompt",
                        placeholder="e.g., Save Button, Search Bar, Close Icon",
                        value="Search Bar"
                    )
                    ground_btn = gr.Button("Ground UI Element", variant="primary")
                with gr.Column():
                    annotated_out = gr.AnnotatedImage(label="Florence-2 Bounding Box Annotation")
                    bbox_json = gr.Code(label="Bounding Box JSON Output", language="json")
            
            ground_btn.click(
                simulate_vision_grounding,
                inputs=[img_input, target_input],
                outputs=[annotated_out, bbox_json]
            )

        # Tab 3: Architecture Specs
        with gr.Tab("3. Local Architecture & Safety"):
            gr.Markdown(
                """
                ### Core System Principles
                
                * **Zero-Cloud Cognition:** MoA routing, Distil-Whisper STT, and Florence-2 OCR remain on your local NVIDIA RTX GPU.
                * **Fail Closed:** Every system write or patch ledger update goes through an isolated ticket gate. Unapproved actions fail closed automatically.
                * **Hardware Control Plane:** Win32 ROI overlays, half-duplex audio, kill switch, and actuator jail.
                """
            )

demo.launch()
