# Dānā: An Agentic CAD & Manufacturing Engineer

**Repository:** `dana` · **Status:** 2026-10-30 · **Companion:** [`../ARCHITECTURE.md`](../ARCHITECTURE.md) (the system reference this paper summarizes)

---

## Abstract

Dānā turns a plain-language request ("a 40 × 30 × 10 mm mounting plate with a centered 4 mm through-hole") into a verified FreeCAD model and the files needed to make it: STEP/STL, a dimensioned 2D drawing, a costed bill of materials, a printability report, G-code and a replayable macro. An LLM proposes one tool call at a time inside a phase-gated ReAct loop; deterministic code owns everything numeric — geometry, measurement, collision and stability checks, rebuilds. The design target is a single 8 GB GPU (an RTX 2080 running `qwen2.5-coder:14b` with an 8,192-token context) or free/cheap cloud models, which constrains how many tool schemas each turn may carry. An offline evaluation suite of five canonical CAD tasks, run through the real loop, passes 5/5 with no turn's tool schema over 2,000 tokens.

---

## 1. Problem and thesis

Small and free-tier language models fail at 3D CAD in predictable ways:

- **They invent numbers.** Asked to place a hole, a model guesses coordinates instead of measuring the part, or assumes the wrong placement convention (a box's corner vs. its center).
- **They drown in tools.** Dānā's catalog of 95 tools is ~18,000 tokens of JSON schema even with descriptions minified. Offered at once it overflows a local model's context and multiplies cloud cost; offered too narrowly, the right tool is missing.
- **They declare success without checking.** "The hole goes through" is a claim; whether it does is a property of the solid.
- **They are expensive to iterate with.** Changing one dimension by re-prompting re-runs the whole build through the model.

**Thesis.** Keep the model in the role it is good at — decomposing intent into a plan and choosing the next operation — and take every other responsibility away from it:

1. a **phase-gated state machine** decides which tools exist on each turn, within a fixed token budget;
2. a **typed intermediate representation** turns each tool call into geometry the same way every time, so a session can be replayed, re-parameterized or compiled without the model;
3. **deterministic verification** (B-Rep topology, boolean intersection, mass properties, mesh checks) decides whether a part is right.

---

## 2. System architecture

### 2.1 Turn pipeline

A FastAPI server exposes one WebSocket per chat (`/ws/chat`). Each user turn runs as a tracked `asyncio` task chained per session, so turns never overlap and the socket keeps reading (Stop interrupts the model's HTTP stream itself). Every turn sends only the system prompt, the user message and a distilled working-memory summary (≤ 1,200 characters) — never a replay of the whole chat. The model proposes at most one tool call per step; `dispatch_tool_call` runs it in a worker thread, so a 60 s FreeCAD call never blocks other sessions, and FreeCAD itself runs as **one `FreeCADCmd` subprocess per call** against a per-session document, with a backup restored on any failure. The loop ends when the model answers in text (or after 30 steps).

### 2.2 Phase-restricted tool retrieval

A CAD session moves through `planning → executing ⇄ validating → done`. Geometry tools do not exist until `create_plan` succeeds, each task declares the tools it expects, and a task advances when they have run. Each phase gets a bounded tool set:

| Phase | Tools offered |
|---|---|
| Planning | core tools minus those that need an existing plan |
| Executing a task with declared tools | exactly those tools + discovery and plan management (other core tools stay callable without their schemas) |
| Executing without declared tools, validating, wrong-tool escape hatch, plan done | discovery and plan management always; then the phase's candidates, ranked (tools this turn used first, then by a 384-d hash-embedding retriever against the task), until `_TOOL_TOKEN_BUDGET` (2,000 tokens) is spent |

Schema descriptions are minified to their first sentence (`schema_minify.py`), and any tool outside the offered set is one `search_tool_catalog` / `load_specific_tool` call away. The evaluation in §5 found and closed the paths where this used to fail: a finished plan's wrap-up turn offered the entire CAD domain (10,336 tokens — more than the local context on its own), and four other phases could fall back to it.

### 2.3 Fleet routing

Which model serves a turn is a pure function of configuration and turn state (`routing_config.yaml`, `dana/core/llm_router.py`): entries list provider, model, priority, context window and cost; an entry is eligible only if its context window (with a safety margin) fits the turn; eligible entries are tried in priority order with local Qwen as the terminal fallback. Planning turns are pinned to a cloud model, geometry-precision turns exclude local models (which were seen to hallucinate argument names there), bookkeeping turns prefer the local model, and `allow_cloud: false` makes the whole system local. An explicit `DANA_CLOUD_PROVIDER` outranks the configured priorities, and `DANA_FREE_ONLY=true` removes every priced entry, so a budget set in `.env` is never bypassed for a paid model. A failed entry falls through to the next.

### 2.4 Resource constraints

`ollama_num_ctx()` is capped at 8,192 (a larger KV cache ran out of VRAM on the reference GPU); local requests serialize on one lock; local models unload when the last connection closes; OCR, speech and tool retrieval run on CPU. Without FreeCAD (the hosted demo, Linux CI), a trimesh + manifold3d mock engine provides the same tool results from meshes, placed with FreeCAD's conventions.

---

## 3. Universal IR and zero-token parametric rebuilds

Every geometry tool is defined once in `dana/plugins/freecad/ir.py` as an IR kind: `from_args` builds a step dict from a live call, `from_record` rebuilds it from the session's call log, and one Jinja2 template (`universal_ir.py.jinja2`) renders steps into FreeCAD Python. Composite tools (a hole on a named face) unroll into atomic steps (a face-relative profile, then a boolean). Four consumers share this path:

- **live execution** — the tool call itself;
- **macro export** — the whole session as one standalone `.py` that rebuilds the part in stock FreeCAD, with a provenance header (editable parameters, LLM calls vs. free rebuilds);
- **skill compilation** — `compile_plan_as_skill` freezes a finished plan into a new parameterized tool;
- **parametric rebuild** — the Parameters tab.

**Zero-token rebuilds.** The Parameters tab lists every editable dimension recovered from the call log, with units and a warning where a downstream fillet selects edges by position. Changing a value patches the IR steps and renders them once into a *scratch* document, which replaces the live one only if the whole script succeeds — a bad value never touches the model. No LLM is called. Each parameter keeps the value the agent originally chose, so any field, or the whole model, can be reset to the agent's design, also for zero tokens. A live FreeCAD round-trip test checks that the live session, the exported macro and the compiled skill produce identical volumes.

---

## 4. Closed-loop manufacturing verification

- **Topology.** `query_topology` reports every face's surface type, area, centroid, normal and parametric extent from the B-Rep, so the agent (and the user) can verify a hole is a bore rather than trust a bounding box.
- **Collisions.** `validate_assembly_collisions` intersects every member pair's solids (`Shape.common`) after an axis-aligned bounding-box broadphase, and reports each pair's true minimum clearance; a pair that cannot be computed fails the check instead of passing as collision-free.
- **Stability.** The BOM computes each part's mass from its volume and material and checks whether the assembly's center of mass, dropped onto the ground plane, lies inside the convex hull of its ground-contact points, with a margin.
- **Printability.** The part is meshed and checked for closed, manifold, consistently wound geometry, build-volume fit (a PrusaSlicer profile, `DANA_PRINTER_BED_MM`, or a Prusa MK4) and overhangs beyond 45°; the six principal orientations are scored and the best one recommended.
- **Drawings and robotics.** TechDraw produces dimensioned orthographic and isometric sheets with a title block filled from the BOM; `export_assembly_to_urdf` writes one link per part with its mesh, mass and inertia tensor, joined by the kinematic joints the agent declared.
- **Pipeline.** `run_full_manufacturing_pipeline` chains BOM → collisions → URDF → drawing → per-part printability → slicing, and reports `complete: false` with a reason (for example, colliding parts) rather than a partial success. It stops at G-code: sending a job to a printer is a separate call that always asks the user.

---

## 5. Empirical evaluation

### 5.1 Offline suite

`tests/evals/` runs five canonical tasks through the real `/ws/chat` loop on the mock engine. The model is scripted, so runs are deterministic and free, but it must pass the real gates: load the CAD capability, create a plan the validator accepts, load tools it isn't offered, and call tools whose arguments are validated. Subprocesses and network access are blocked during a run. Geometry is checked against hand-computed values.

| Task | Result | Turns | Max tool-schema tokens | Retries | Geometry checks |
|---|---|---:|---:|---:|---:|
| Single primitive + topology check | PASS | 9 | 1,998 | 0 | 6/6 |
| Multi-feature part (holes + chamfer) | PASS | 11 | 1,954 | 1 | 5/5 |
| Boolean cut + tool-token budget | PASS | 6 | 1,998 | 0 | 5/5 |
| Assembly + manufacturing verdict | PASS | 17 | 1,958 | 1 | 11/11 |
| Zero-token parametric rebuild | PASS | 11 | 1,954 | 1 | 5/5 |

Per-phase maxima across the five tasks, in cl100k tokens of tool schema per call:

| Phase | First measurement | Now | Target |
|---|---:|---:|---:|
| Planning | 2,826 | 1,503 | ≤ 2,000 |
| Executing | 3,382 | 1,979 | ≤ 2,000 |
| Validating | (uncapped) | 1,993 | ≤ 2,000 |
| Plan done (wrap-up) | 10,336 | 1,998 | ≤ 2,000 |

Turns over 2,000 tokens: 0 of 54. Executing turns with declared tools are 756–1,312 tokens; the turns near 2,000 are the capped sets, which add optional tools by relevance until the budget is spent.

Retries are genuine self-corrections: an invalid face name rejected by the tool, and a part move blocked until the reference object was measured. The assembly task checks both verdicts: a colliding stand makes the pipeline report `complete: false` for that reason; once the post is lifted clear, collisions, URDF export, BOM masses and printability all pass. The rebuild task changes the plate from 80 to 100 mm with zero LLM calls and checks the new volume. The ceilings are enforced: a turn that grows past them fails the suite (`pytest tests/evals/`, ~5 s; `python scripts/run_offline_evals.py` prints the scorecard).

### 5.2 Live-model check

The mounting-plate prompt from the abstract was run through the real router, the real loop and the mock engine.

**DeepSeek (paid).** The model planned three tasks with valid tool ids, called every tool correctly on the first attempt, stayed at or under 2,645 tool-schema tokens per turn, and produced a solid of 11,500.57 mm³ — exactly a 40 × 30 × 10 mm plate minus a 4 mm-radius through-hole (as a 32-segment mesh) — in six LLM calls, ~27,000 prompt tokens and 15 s. Earlier runs found two defects, since fixed: the mock engine centered primitives on their placement while FreeCAD places a box by its corner, and `query_topology` was missing from the only tool list the planner sees.

**OpenRouter free tier, $0.00.** With `DANA_CLOUD_PROVIDER=openrouter` and `DANA_FREE_ONLY=true`, every turn went to `nemotron-3-super-120b-a12b:free`. (Before that, the router ignored `DANA_CLOUD_PROVIDER` after planning and billed the paid priority-1 entry.) The weaker model exposed four more defects, each fixed and re-run:

- the placement parameters said "Global X-axis offset" without saying which point is placed, so it centered the hole's cylinder at mid-depth (11,750 mm³);
- `load_specific_tool` reported "available now" for a tool the active task would block, and the run gave up with no hole;
- the sketch-geometry error described a flat `[x, y, ...]` format, which the model then sent five times;
- arrays sent as JSON strings were rejected although only the encoding was wrong.

After the fixes, three consecutive runs built the correct 11,500.57 mm³ solid in 6, 9 and 17 LLM calls (19–47 s), with no turn over 2,000 tool-schema tokens. The cleanest used 1,503 tokens to plan, 876–1,100 per executing turn and 1,957 to wrap up. Earlier free-tier runs, before the fixes, also picked a sketch-and-pocket route the model could not finish; free models remain the least reliable lane, which is why `routing_config.yaml` ranks them below DeepSeek by default.

---

## 6. Limitations

- The offline suite uses a scripted model: it measures Dānā's loop, gates and geometry, not a model's planning ability. Live checks so far use one prompt (one paid run, a handful of free-tier runs).
- The mock engine approximates B-Rep topology with mesh regions and cannot export STEP; real FreeCAD is required for production output.
- Printability is a geometric heuristic, not a slicer simulation; slicing and printing have been tested against mocks, not real hardware.
- A task's declared tools are never dropped to meet the budget, so a task declaring four or more of the heaviest tools (feature-on-face, assembly constraints, drawings) could still exceed 2,000 tokens; plan-management tools give way first.

---

## Code map

| Concern | Where |
|---|---|
| Turn loop, FSM, tool budget | `dana/api/server.py`, `dana/core/react_dispatch.py`, `dana/tools/schema_minify.py` |
| Routing | `routing_config.yaml`, `dana/core/llm_router.py`, `dana/core/model_provider.py` |
| CAD engine, IR, templates | `dana/plugins/freecad/engine.py`, `ir.py`, `templates/` |
| Macro export, skills, rebuild | `py_export.py`, `skill_compiler.py`, `rebuild.py` |
| Manufacturing | `dana/plugins/manufacturing/`, `dana/plugins/freecad/printability.py`, `techdraw_export.py`, `dana/tools/urdf_builder.py` |
| Mock engine | `dana/platform/mock.py` |
| Evaluation | `tests/evals/eval_harness.py`, `scripts/run_offline_evals.py` |
