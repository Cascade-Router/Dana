# Dānā: A Production-Hardened Cybernetic Control Plane for Autonomous Desktop Actuation, Vision Grounding, and Self-Healing Execution

**Package:** `dana` · **Product:** Dānā / Dana · **Evaluation:** offline CAD & manufacturing agent evals, `tests/evals/` (scorecard 2026-10-30)

---

## Abstract & Design Philosophy

Dānā is a **production-hardened**, local-by-design cybernetic control plane for Windows desktops. Cognition, tool binding, hybrid vision grounding, transactional file mutation, and episodic memory remain on-device (Ollama + local Florence-2 / YOLO). The LangGraph ReAct corridor is the deterministic skeleton; LLMs propose content and tool calls, while Python owns routing, validation, interrupts, shadow commits, fatal classification, and fail-closed exits.

> **Note (2026-10-30).** This paper predates the current CAD product; several modules it names (`dana/graph/`, the LangGraph corridor, the OSWorld harness) no longer exist. The current architecture is in [`../ARCHITECTURE.md`](../ARCHITECTURE.md). Section 3 now describes the evaluation suite that does exist.

The agent loop is evaluated offline by five canonical CAD and manufacturing tasks (`tests/evals/`, Section 3), run through the real ReAct loop and its gates with a scripted model on the trimesh mock CAD engine.

Three invariants structure the system:

| Principle | Operational meaning |
|-----------|---------------------|
| **Local-by-design** | Voice critical path, ReAct bind-tools loops, Florence OCR, hybrid UIA grounding, and the episodic SQLite ledger do not require a cloud round-trip. Optional cloud surfaces (e.g. Gradio Space, GitHub issue escalation) are off the hot path. |
| **Zero Cloud Latency (control plane)** | Plan-Then-Execute, Mailroom RapidFuzz short-circuits, heuristic critic / consolidate fallbacks, and autonomous sub-graph retries (N=2) keep decision edges offline-capable. Heavy models load JIT; the graph itself does not wait on remote APIs. |
| **Fail-Closed HITL Safety** | Destructive / ledger-mutating intents route through `draft_cursor_prompt` → `ticket_validate` → `jason_ticket_review` → `ticket_approval` (interrupt). `FATAL_EXCEPTIONS` bypass Critic retries and draft HITL tickets. Deny / exhausted validation / REPL self-heal exhaustion halt without consolidating bad preferences into memory. |

The product brand is **Dānā** / **Dana**; the Python package is `dana`. Supporting packages: `dana_jason_loop/`, `dana_security/`. This paper describes the live ReAct architecture as implemented in `dana/agentic_react_graph.py`, `dana/schema.py`, `dana/graph/`, `dana/exec/shadow_workspace.py`, `dana/vision/hybrid_grounding.py`, `dana/memory/`, and `dana/ui/watchdog.py`.

---

## 1. Architecture & Topology

### 1.1 Live ReAct corridor (production topology)

`compile_dana_react_graph` wires the following StateGraph over `ReactGraphState`. Entry is **Memory Hydration → Supervisor Router** (`hydrate_memory` → `planner` Plan-Then-Execute), then executor → agent:

```text
START
  │
  ▼
hydrate_memory ──► planner (supervisor router) ──► executor ──► agent
                                                                  │
              ┌───────────────────────────────────────────────────┼───────────────────────────────┐
              │ draft_cursor_prompt                               │ other tool_calls              │ halt / final
              ▼                                                   ▼                               ▼
      ticket_validate                                          tools                    consolidate_memory
              │                                                   │                               │
     ┌────────┼────────┐          ┌───────────────────────────────┼───────────────┐               ▼
     │ valid  │ invalid│          │ REPL err      │ fatal_block   │ continue      │              END
     ▼        │ (<3)   │          ▼               ▼               ▼               │
jason_ticket  │→ agent │       critic        fail_closed       agent /            │
  _review     │        │          │          (+ HITL ticket)   END path           │
     │        │ max→END│          ▼               │                               │
     ▼        └────────┘       tools ◄── (retry)  ▼                               │
ticket_approval                   │              END                              │
     │ approve → tools            │ retries exhausted                             │
     │ deny    → END              ▼                                               │
                                  fail_closed → END                               │
                                                                                  │
Actuation plane (from tools):                                                     │
  · Hybrid Win32 UIA + 2-Stage Crop & Zoom Florence-2                             │
  · Transactional Shadow Workspace (.dana_scratch/<session_id>/)                  │
  · Zero-Copy raw_state_buffer + Sub-Graph Retry (N=2)                            │
Memory plane:                                                                     │
  · CompactionEngine — exponential decay (λ=0.05/hr) + spatial TTL (900s)         │
```

### 1.2 Mermaid — Hydration · Supervisor · Shadow · Vision · Buffer · Memory

```mermaid
flowchart TB
  subgraph Ambient["Ambient plane"]
    WD[ShellWatchdog<br/>stdout/stderr text feed]
    TOAST[Windows toast tickets<br/>dana.ui.notifications]
    WD --> TOAST
    TOAST -.->|build_structured_plan| PL
  end

  subgraph Corridor["ReAct corridor — compile_dana_react_graph"]
    HM[hydrate_memory]
    PL[planner / supervisor router]
    EX[executor]
    AG[agent / MoA bind_tools]
    TV[ticket_validate]
    JR[jason_ticket_review]
    TA[ticket_approval HITL]
    TL[tools]
    CR[critic]
    FC[fail_closed + FATAL ticket]
    CM[consolidate_memory]

    START([START]) --> HM --> PL --> EX --> AG
    AG -->|draft_cursor_prompt| TV
    TV -->|valid| JR --> TA
    TV -->|invalid retry| AG
    TA -->|Approve| TL
    TA -->|Deny| END1([END])
    AG -->|tool_calls| TL
    TL -->|python_repl error<br/>non-fatal, retry_count < max| CR --> TL
    TL -->|FATAL_EXCEPTIONS / fatal_block| FC --> END2([END])
    TL -->|retries exhausted| FC
    TL -->|continue| AG
    AG -->|halt| CM --> END3([END])
    TL -->|halt| CM
  end

  subgraph SubGraph["Injectable sub-graph retry — compile_subgraph_retry_graph"]
    SG[subgraph]
    BUMP[bump_subgraph_retry]
    ESC[escalate_subgraph]
    SUP[supervisor]
    SG -->|failure, count < N=2| BUMP --> SG
    SG -->|exhausted / fatal| ESC --> SUP
    SG -->|success| ESC --> END4([END])
  end

  subgraph Actuation["Vision · Shadow · Buffer · Memory"]
    HY[HybridVisionGrounding<br/>UIA → Florence coarse → crop/zoom]
    SW[ShadowWorkspace<br/>.dana_scratch/]
    RB[(raw_state_buffer<br/>zero-copy traces)]
    EM[(EpisodicMemoryStore<br/>+ CompactionEngine)]
  end

  TL --> HY
  TL --> SW
  SG -.-> RB
  HM <--> EM
  CM <--> EM
```

### 1.3 Control-plane state (minimal)

Durable pointers live in graph state; chat history / CoT live on the Blackboard keyed by `session_id` (`dana/schema.py:ReactGraphState`):

| Field group | Examples | Lifetime |
|-------------|----------|----------|
| Bureaucratic | `session_id`, `current_agent`, `active_intent` | Cross-node |
| Ephemeral turn | `messages`, `iterations`, `always_include`, `execution_plan` | Single invoke |
| REPL self-heal | `execution_error`, `critique_history`, `retry_count`, `max_retries`, `last_code_snippet` | Until heal or fail-closed |
| Fatal / HITL | `fatal_block`, `drafted_ticket`, `ticket_validated`, `jason_critique`, `consecutive_denials` | Until approve/deny or halt |
| Zero-copy diagnostics | `raw_state_buffer`, `subgraph_retry_count`, `max_subgraph_retries` | Until escalate / clear |
| Memory hydrate | `memory_context` | Per turn |

---

## 2. Detailed Subsystem Specs

### 2.1 Transactional Shadow Workspaces & Fatal Error Classification

**Shadow workspaces** (`dana/exec/shadow_workspace.py`). File mutations from REPL / `file_editor` stage under `.dana_scratch/<session_id>/` first:

| Outcome | Behavior |
|---------|----------|
| `exit_code == 0` | `commit()` copies staged files to destinations, then clears scratch |
| Nonzero exit / exception | `rollback()` discards scratch; destination paths untouched |
| Context binding | `bind_shadow_workspace` / `get_active_shadow` via `ContextVar` for tool hooks |

`run_shadow_transaction(session_id, runner)` is the canonical API: commit on success, rollback otherwise.

**Fatal error classification** (`dana/graph/nodes/critic.py`). `FATAL_EXCEPTIONS` are never Critic-healed:

```text
PermissionError | FileNotFoundError | ModuleNotFoundError
ConnectionRefusedError | TimeoutError | OSError
```

When `fatal_block` or `is_fatal_execution_error(...)` is true, `route_after_execution` sends **tools → fail_closed** (bypass Critic retries). `fail_closed_node` / fatal Critic short-circuit draft a structured HITL ticket (`drafted_ticket` with objective `Fatal OS Block: Missing dependency or permission denied`) on the existing ticket corridor fields—without consolidating memory.

Fixable code faults (`SyntaxError`, `NameError`, `TypeError`, …) remain on the bounded Critic loop (`max_retries` default **3**).

### 2.2 Spatial Coordinate TTL (900s) & Exponential Decay

**Spatial TTL** (`dana/graph/nodes/memory.py`). On consolidate, spatial / UI location facts receive `ttl_seconds = SPATIAL_FACT_TTL_SECONDS` (**900** = 15 minutes) when not explicitly set. Hydrate calls `prune_expired_entries()` before search so stale click targets and element locations do not poison grounding.

**Exponential decay** (`dana/memory/compaction.py` · `CompactionEngine`). After consolidate, `compact_memory` applies:

\[
W = W_0 \cdot \exp(-\lambda \cdot \Delta\mathrm{hours}), \quad \lambda = 0.05\ \mathrm{(procedural)}
\]

| Category | \(\lambda\) | Policy |
|----------|-------------|--------|
| Procedural / non-preference facts | **0.05 / hour** | Pruned when decayed weight &lt; **0.15** |
| `user_preference` | **0.0** | Never decay |

### 2.3 Hybrid Win32 UIA + Crop-and-Zoom Florence-2

**Module:** `dana/vision/hybrid_grounding.py` · `HybridVisionGrounding` + `dana/vision/uia_provider.py` · `Win32UIAProvider`.

Pipeline (all side-effects injectable for offline evals):

1. **UIA first** — `Win32UIAProvider.find_element_bounds(label)` → return `[0, 1000]^4` immediately on hit.
2. **Coarse Florence** — phrase ground on the full frame via `run_ocr_with_region`.
3. **Crop & zoom if small** — if coarse width **or** height &lt; **30** (1000-scale): crop ROI with **15%** padding (`ROI_PADDING = 0.15`), upscale **2×** (Lanczos/bicubic), fine Florence on the zoomed crop, project back to global `[0, 1000]^4`.

Stage tags: `uia` | `coarse` | `zoom` | `miss`. Coordinate contract remains Florence **0–1000** normalized (with already-pixel acceptance in `norm_box_to_screen` / conversion helpers).

### 2.4 Zero-Copy State Buffer & Autonomous Sub-Graph Retries (N=2)

**Zero-copy buffer** (`dana/graph/buffer.py`). `store_raw_trace` writes the **full** traceback + exception metadata into `raw_state_buffer.last_error` without LLM truncation. Supervisors and critics may summarize separately; diagnostics stay complete.

**Sub-graph retries** (`dana/graph/subgraph_router.py`). Default `max_subgraph_retries = 2` (`DEFAULT_MAX_SUBGRAPH_RETRIES`):

```text
START → subgraph ─(non-fatal, count < 2)→ bump_subgraph_retry → subgraph
               ─(exhausted / fatal)→ escalate_subgraph → supervisor → END
               ─(success)→ escalate_subgraph → END
```

Fatal failures (`fatal_block` / `FATAL_EXCEPTIONS`) skip local retries and escalate immediately with `raw_state_buffer` intact. This corridor is injectable via `compile_subgraph_retry_graph` and does not rewrite ToolForge gates or `dana/paths.py`.

### 2.5 Legacy corridor nodes (still live)

| Subsystem | Modules | Role |
|-----------|---------|------|
| Self-healing REPL Critic | `dana/graph/nodes/critic.py` | Bounded patch loop for fixable errors; fatal → HITL |
| Macro engine | `dana/macros/engine.py` | Phrase-grounded record/replay |
| Shell Watchdog | `dana/ui/watchdog.py` | Opt-in ambient Traceback → toast + plan handoff |
| Episodic store | `dana/memory/store.py` | SQLite `episodic_facts` + TTL prune |

---

## 3. Empirical Evaluation: offline CAD & manufacturing evals

### 3.1 Method

**Harness:** `tests/evals/eval_harness.py` · **pytest entry:** `tests/evals/test_offline_evals.py` · **Scorecard:** `python scripts/run_offline_evals.py [--json results.json]`

Each task runs one chat session through the real `/ws/chat` server (in-process, FastAPI TestClient) against the trimesh mock CAD engine. The model is scripted, so runs are deterministic, free and offline, but it must pass the same gates a real model does: load the CAD capability, create a plan the validator accepts (one task per operation), load tools it isn't offered, and call tools whose arguments the dispatcher validates. Per task the harness records:

- **turns**: LLM calls the loop made;
- **tool-schema tokens**: the `tools=` payload of every call (cl100k) and its FSM phase, held to per-phase ceilings: planning 1,476, executing 2,559, post-plan wrap-up 2,000 (`_TOOL_TOKEN_BUDGET`);
- **retries**: calls the gates rejected that the model then retried successfully;
- **geometry checks**: assertions against hand-computed values (volumes, bounding boxes, faces, clearances, BOM masses, URDF links).

Subprocesses (FreeCADCmd, a slicer) and non-loopback network access are blocked and counted while a task runs; a passing run has none. The script also fails if the real `freecad_output/`, `exports/` or `agent_workspace/` changed.

Tasks: (1) a block and a rod measured with `query_topology`; (2) a plate with four holes cut through its top face and a chamfer, including a wrong face name the tool rejects; (3) a boolean through-bore; (4) a stand whose post collides — the manufacturing pipeline reports `complete: False`, blamed on the collision — then moved clear (after the bounding-box gate makes it measure first) and passing the collision check, URDF export, BOM and printability; (5) the plate rebuilt from 80 to 100 mm with zero LLM calls.

### 3.2 Scorecard (2026-10-30)

| Task | Result | Turns | Max tool-schema tokens | Retries | Geometry checks |
|---|---|---:|---:|---:|---:|
| Single primitive + topology check | PASS | 7 | 2,325 | 0 | 6/6 |
| Multi-feature part (holes + chamfer) | PASS | 11 | 2,559 | 1 | 5/5 |
| Boolean cut + tool-token budget | PASS | 6 | 2,325 | 0 | 5/5 |
| Assembly + manufacturing verdict | PASS | 17 | 2,401 | 1 | 11/11 |
| Zero-token parametric rebuild | PASS | 11 | 2,559 | 1 | 5/5 |

The suite's first run found the wrap-up turn after a finished plan offering the whole CAD tool domain: 8,586–10,336 tokens of schema, more than a local model's 8,192-token context. With the post-plan set capped and the core schemas tightened, the per-phase maxima went from planning 2,826 / executing 3,382 / wrap-up 10,336 to 1,476 / 2,559 / at most 2,000.

Reproduce:

```bash
pytest tests/evals/                      # ~5 s, part of the normal suite
python scripts/run_offline_evals.py      # prints the scorecard
```

---

## 4. Modular Extension Contract

Dānā’s corridor is intentionally **injectable**: production defaults bind real nodes; evals and plugins substitute callables without rewriting edges. All imports use the `dana` package.

### 4.1 Register custom graph nodes

```python
from typing import Any

from dana.agentic_react_graph import compile_dana_react_graph
from dana.graph.nodes.critic import critic_node, fail_closed_node
from dana.graph.nodes.memory import (
    consolidate_memory_node,
    hydrate_memory_node,
)
from dana.schema import ReactGraphState


def my_agent_node(state: ReactGraphState) -> dict[str, Any]:
    ...


def my_tools_node(state: ReactGraphState) -> dict[str, Any]:
    ...


def my_planner(state: ReactGraphState) -> dict[str, Any]:
    """Optional override for Plan-Then-Execute / supervisor router."""
    return {"execution_plan": {"steps": []}, "current_agent": "Planner"}


graph = compile_dana_react_graph(
    my_agent_node,
    my_tools_node,
    planner_node_fn=my_planner,
    critic_node_fn=critic_node,
    fail_closed_node_fn=fail_closed_node,
    hydrate_memory_node_fn=hydrate_memory_node,
    consolidate_memory_node_fn=consolidate_memory_node,
)
```

### 4.2 Sub-graph retries + zero-copy buffer

```python
from dana.graph.buffer import store_raw_trace
from dana.graph.subgraph_router import (
    apply_subgraph_failure,
    compile_subgraph_retry_graph,
)


def my_subgraph(state: dict) -> dict:
    try:
        ...
    except Exception as exc:
        return apply_subgraph_failure(state, exc)


def my_supervisor(state: dict) -> dict:
    return {"current_agent": "Supervisor"}


retry_graph = compile_subgraph_retry_graph(my_subgraph, my_supervisor)
```

### 4.3 Custom tools

```python
# dana/tools/my_actuator.py
def ping_workspace(path: str = ".") -> str:
    """Return a grounded observation string for the ReAct loop."""
    return f"[ping_workspace] ok path={path!r}"


# Inside your tools_node / execute_fn dispatcher:
from dana.tools.my_actuator import ping_workspace

DISPATCH = {
    "ping_workspace": lambda args: ping_workspace(**args),
}
```

Register the tool id with the Intent Broker / MoA bind list the same way existing tools are foresight-cascaded—without modifying ToolForge security gates or `dana/paths.py`.

### 4.4 Shadow workspace + hybrid vision injection

```python
from dana.exec.shadow_workspace import ShadowWorkspace, run_shadow_transaction
from dana.vision.hybrid_grounding import HybridVisionGrounding
from dana.vision.uia_provider import Win32UIAProvider

ws, code, obs = run_shadow_transaction("eval-session", lambda w: (0, "ok"))

grounder = HybridVisionGrounding(
    uia_provider=Win32UIAProvider(control_tree=[]),
    florence_ground_fn=my_florence_fn,  # injectable
)
box = grounder.locate_ui_element(image, "Save")
```

### 4.5 Episodic store + compaction

```python
from dana.graph.nodes.memory import make_hydrate_memory_node, make_consolidate_memory_node
from dana.memory.compaction import CompactionEngine
from dana.memory.store import EpisodicMemoryStore

store = EpisodicMemoryStore(db_path=":memory:")
hydrate = make_hydrate_memory_node(store)
consolidate = make_consolidate_memory_node(store)
CompactionEngine().compact_memory(store)
```

---

## 5. References (code map)

| Concern | Primary paths |
|---------|----------------|
| Graph compile & routing | `dana/agentic_react_graph.py` |
| Shared state / Handoff | `dana/schema.py` |
| Critic / FATAL / fail-closed | `dana/graph/nodes/critic.py` |
| Shadow workspaces | `dana/exec/shadow_workspace.py` |
| Zero-copy buffer | `dana/graph/buffer.py` |
| Sub-graph retries (N=2) | `dana/graph/subgraph_router.py` |
| Hybrid UIA + crop/zoom | `dana/vision/hybrid_grounding.py`, `dana/vision/uia_provider.py` |
| Memory hydrate/consolidate | `dana/graph/nodes/memory.py` |
| Exponential compaction | `dana/memory/compaction.py` |
| Episodic SQLite | `dana/memory/store.py` |
| Jason loop package | `dana_jason_loop/` |
| Security / patch ledger | `dana_security/` |
| Offline agent evals | `tests/evals/eval_harness.py`, `tests/evals/test_offline_evals.py` |
| Eval scorecard | `scripts/run_offline_evals.py` |

---

*End of white paper.*
