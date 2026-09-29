# Demo script: prompt → FreeCAD bracket → standalone macro (2:00)

A shot list for a two-minute screen recording of the Dānā pipeline. It shows a local terminal, one bracket-generation prompt, the agent driving FreeCAD, and the exported macro rebuilding the part in stock FreeCAD.

---

## Before recording (off camera, ~10 min)

1. **Models.** Start Ollama and check that the local models are pulled:

   ```bash
   ollama serve                        # skip if the Ollama app is already running
   ollama pull qwen2.5-coder:14b       # local tool-calling fallback
   ollama pull qwen2.5vl:7b            # only needed for the optional image beat
   ```

2. **`.env`.** Start from `.env.example` and set the following:

   ```ini
   OLLAMA_URL=http://localhost:11434
   OLLAMA_BASE_URL=http://localhost:11434
   DEEPSEEK_API_KEY=...          # optional: cloud planning/geometry lane
   DANA_HEADLESS=false           # show the FreeCAD window updating live
   DANA_AUTO_APPROVE=1           # skip approval prompts for this recording only
   DANA_VISION_MOCK_FALLBACK=1   # only if this machine has no GPU/vision model
   ```

   To get a clean routing story, copy `routing_config.yaml.example` to `routing_config.yaml`.

3. **Demo client.** The UI has no macro-export button yet; the export is a WebSocket message. Save this as `demo_client.py` **outside the repo** (for example, next to your recording files):

   ```python
   """Drive one Dana turn over /ws/chat, then export the session's FreeCAD macro."""

   import asyncio
   import json
   import sys

   import websockets

   URL = "ws://localhost:8000/ws/chat"
   PROMPT = " ".join(sys.argv[1:]) or (
       "Design an L-shaped mounting bracket: a 60x40x5 mm base plate and a 40x5x50 mm "
       "upright fused along the back edge, with two 6 mm through-holes in the base."
   )


   async def main() -> None:
       async with websockets.connect(URL, max_size=None) as ws:
           ready = json.loads(await ws.recv())
           print(f"[demo] connected, session {ready['session_id']}")
           print(f"[demo] > {PROMPT}\n")
           await ws.send(json.dumps({"text": PROMPT}))

           while True:
               msg = json.loads(await ws.recv())
               kind = msg.get("type")
               if kind == "tool_dispatch_start":
                   print(f"  -> {msg.get('tool_name')}  {msg.get('args_summary') or ''}")
               elif kind == "hitl_approval_required":
                   payload = msg["payload"]
                   print(f"  [auto-approve] {payload['action_name']}: {payload['description']}")
                   await ws.send(json.dumps(
                       {"type": "hitl_response", "payload": {"request_id": payload["request_id"], "approved": True}}
                   ))
               elif kind == "assistant_message":
                   print(f"\n[dana] {msg['content']}\n")
                   break

           await ws.send(json.dumps({"type": "export_python_script", "filename": "bracket_demo"}))
           while True:
               msg = json.loads(await ws.recv())
               if msg.get("type") == "python_script_exported":
                   print(f"[demo] macro written to {msg['path']}")
                   return
               if msg.get("type") == "assistant_message":  # nothing to export
                   print(f"[demo] {msg['content']}")
                   return


   asyncio.run(main())
   ```

4. **Dry run once, all the way through.** Delete `exports/bracket_demo.py` afterwards so the take starts clean. Note how long the agent takes on your hardware. If it runs past ~60 s, speed up that stretch in editing rather than shortening the prompt.

5. **Screen layout.** Put the terminal on the left half (font ≥ 16 pt, dark theme) and leave the right half for the FreeCAD window. Close notifications. Record at 1920×1080.

---

## Shot list

| Time | On screen | Narration (suggested) |
|---|---|---|
| **0:00–0:15** | Repo root in the terminal. Run `cat routing_config.yaml \| head -40`, then scroll to the `fleet:` entries. | "Dānā is an agent that designs parts in FreeCAD. Model routing is deterministic: a config file sets which model handles which kind of turn, cloud DeepSeek for geometry and local Qwen as the fallback." |
| **0:15–0:30** | Start the backend: `python scripts/launchers/launch_api_server.py`. Wait for Uvicorn's `Application startup complete`. | "Everything runs locally: a FastAPI server, Ollama on localhost, and FreeCAD's own kernel." |
| **0:30–0:40** | Second terminal tab. Run `python demo_client.py`. The prompt echoes. | "One plain-English request: an L-bracket with a base, an upright and two bolt holes." |
| **0:40–1:15** | Tool calls stream past (`create_plan`, `create_freecad_box`, `perform_freecad_boolean`, `get_freecad_bounding_box`, …) while the FreeCAD window updates on the right. | "It plans first, then builds step by step, and measures the real geometry before placing the next feature. Every boolean cut and fuse is a typed tool call, not free-form code." |
| **1:15–1:25** | The final `[dana]` summary, then `[demo] macro written to …/exports/bracket_demo.py`. | "When it's done, the whole session exports as one standalone FreeCAD macro." |
| **1:25–1:45** | `code exports/bracket_demo.py` (or `less`). Scroll through the `Part::Cut` / `Part::MultiFuse` blocks. | "This is generated from a shared intermediate representation: the same IR drives live execution, this replay, and reusable skills." |
| **1:45–2:00** | In a **fresh** FreeCAD, *Macro → Macros… → bracket_demo.py → Execute*. The bracket rebuilds with no agent running. | "No agent, no model: stock FreeCAD rebuilds the exact part. That's the handoff to a real engineer." |

### Optional 15-second insert: image to blueprint

If there is time, cut in after 0:40. Attach a front/top drawing, and the agent calls `analyze_reference_design`. Show the JSON blueprint it returns (OCR-grounded bounding box plus primitives). Say "Printed dimensions are read by OCR, so the numbers come from the drawing, not the vision model's guess." On a machine without a GPU with `DANA_VISION_MOCK_FALLBACK=1`, the result says `"mocked": true`. **Either** leave this insert out **or** say on camera that it's a placeholder.

---

## If something goes wrong mid-take

- **Client hangs with no tool calls.** The model isn't reachable. Check `curl http://localhost:11434/api/tags` and the backend log in `logs/`.
- **`Nothing to export yet`.** The turn ended before any FreeCAD tool ran, usually because the model answered in prose. Rerun; if it repeats, add "use the FreeCAD tools" to the prompt.
- **FreeCAD window never appears.** Check that `DANA_HEADLESS=false`, and set `DANA_FREECADCMD_PATH` if FreeCAD is installed somewhere non-standard.
- **Approval prompts show up anyway.** The client auto-approves them and prints `[auto-approve]`. That's fine on camera, and it shows the human-in-the-loop gate.

After recording, set `DANA_AUTO_APPROVE` back to `0`.
