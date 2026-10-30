# Demo script: prompt → FreeCAD bracket → free rebuild → standalone macro (2:00)

A shot list for a two-minute screen recording of the Dānā desktop app. It shows one bracket-generation prompt, the agent driving FreeCAD, a dimension changed and rebuilt for 0 tokens in the Parameters tab, and the one-click macro from the Deliverables bar rebuilding the part in stock FreeCAD.

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
   DANA_VISION_MOCK_FALLBACK=1   # only if this machine has no GPU/vision model
   ```

   To get a clean routing story, copy `routing_config.yaml.example` to `routing_config.yaml`.

3. **Start the app.** `./launch_dana.sh` (Windows: `powershell -File launch_dana.ps1`) starts the API server and the desktop UI. Open the **CAD** plugin so the 3D viewport, the Inspector dock and the Deliverables bar are showing.

4. **Dry run once, all the way through,** with this prompt:

   > Design an L-shaped mounting bracket: a 60x40x5 mm base plate and a 40x5x50 mm upright fused along the back edge, with two 6 mm through-holes in the base. Then make a 2D blueprint of it.

   Start a new chat afterwards (and delete the dry run's chat from the sidebar, which also removes its files) so the take starts clean. Then turn on **Settings → Auto-approve tool actions** for the take: it applies to the open chat only and resets with every new chat. Note how long the agent takes on your hardware; if it runs past ~60 s, speed up that stretch in editing rather than shortening the prompt.

5. **Screen layout.** The app on the left two-thirds, a FreeCAD window on the right third. Close notifications. Record at 1920×1080.

---

## Shot list

| Time | On screen | Narration (suggested) |
|---|---|---|
| **0:00–0:10** | The app with the CAD plugin open: chat, empty viewport, Inspector dock (Plan · Topology · Parameters · Terminal), Deliverables bar. | "Dānā is an agent that designs parts in FreeCAD. Model routing is deterministic: a config file sets which model handles which kind of turn." |
| **0:10–0:20** | Paste the prompt into the chat and send. | "One plain-English request: an L-bracket with a base, an upright, two bolt holes, and a drawing." |
| **0:20–0:55** | The reply streams; the **Plan** tab fills and ticks off tasks; the viewport and the FreeCAD window update as `create_freecad_box`, `perform_freecad_boolean`, `get_freecad_bounding_box`, … run. | "It plans first, then builds step by step, and measures the real geometry before placing the next feature. Every boolean is a typed tool call, not free-form code." |
| **0:55–1:05** | The final answer, then the cost bar: model, tokens, cost. | "The whole build cost this many tokens." |
| **1:05–1:25** | **Parameters** tab: the plate's length is 60 mm. Change it to 80 and press **Apply & Rebuild (0 tokens)**. The part regrows; the status says "Rebuilt"; the cost bar shows **+1 free rebuild**. | "Every dimension is editable. Changing one replays the recorded steps without asking the model again: zero tokens." |
| **1:25–1:40** | Deliverables bar: **Blueprint** opens the 2D drawing; **STEP** and **STL** download the part. Then **Macro .py**: open the downloaded file and show its header (the editable parameters, LLM calls vs. free rebuilds). | "Drawings, CAD files and a replayable macro, one click each, none of them costing tokens." |
| **1:40–2:00** | In a **fresh** FreeCAD, *Macro → Macros… → the .py → Execute*. The 80 mm bracket rebuilds with no agent running. | "No agent, no model: stock FreeCAD rebuilds the exact part, including the change. That's the handoff to a real engineer." |

### Optional 15-second insert: an assembly's bill of materials

Cut in after 1:40 with a second chat: "A stand: a 40x40x10 mm base and a 10 mm diameter, 30 mm tall post on top, then cost it." When it finishes, **BOM ▾** in the Deliverables bar opens the bill of materials: per-part mass and cost, the total, and whether the stand tips over. Say "The bill of materials comes straight from the geometry."

### Optional 15-second insert: image to blueprint

If there is time, cut in after 0:20. Attach a front/top drawing, and the agent calls `analyze_reference_design`. Show the JSON blueprint it returns (OCR-grounded bounding box plus primitives). Say "Printed dimensions are read by OCR, so the numbers come from the drawing, not the vision model's guess." On a machine without a GPU with `DANA_VISION_MOCK_FALLBACK=1`, the result says `"mocked": true`. **Either** leave this insert out **or** say on camera that it's a placeholder.

---

## If something goes wrong mid-take

- **No tool calls, only text.** The model isn't reachable or answered in prose. Check `curl http://localhost:11434/api/tags` and the Terminal tab; if it repeats, add "use the FreeCAD tools" to the prompt.
- **Macro .py says there is nothing to export.** The turn ended before any FreeCAD tool ran. Rerun.
- **Apply & Rebuild is disabled or says rebuild is unavailable.** The session used a step that can't be replayed (a raw script or an assembly tool); the Parameters tab names it. Use a fresh chat with the bracket prompt.
- **FreeCAD window never appears.** Check that `DANA_HEADLESS=false`, and set `DANA_FREECADCMD_PATH` if FreeCAD is installed somewhere non-standard.
- **Approval prompts show up anyway.** Auto-approve is off (it resets with every new chat). Approve them on camera; that shows the human-in-the-loop gate.

Auto-approve is off again in the next chat you open.
