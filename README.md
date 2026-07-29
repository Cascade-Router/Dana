---
title: Dānā
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: agpl-3.0
---

# Dānā: Local-First Agentic Voice OS

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![OS: Windows 10/11](https://img.shields.io/badge/OS-Windows%2010%2F11-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-AMIXXM%2FDonna-yellow)](https://huggingface.co/spaces/AMIXXM/Donna)
[![White Paper](https://img.shields.io/badge/docs-White%20Paper-0B7285)](docs/WHITE_PAPER.md)

**Bridge local LLMs (Llama 3.2), hybrid Win32 UIA + Florence-2 vision, and Mixture-of-Agents reasoning (DeepSeek-R1) into a deterministic, low-latency voice operating system — with a CustomTkinter Live Trace UI that makes every graph transition observable.**

Dānā is a **production-hardened**, offline-first agentic control plane for the desktop: wake-word perception, strict mode-isolated cognition, transactional shadow workspaces, filesystem-jailed tool execution, and thread-safe telemetry. It is engineered as infrastructure — not a chatbot shell. Evaluated under adversarial OSWorld-style conditions (`pytest tests/evals/test_osworld_bench.py`).

**Deep dive:** [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) — 5-phase hardening specs, architecture topology, and OSWorld benchmarks.

Try the Gradio simulator on Hugging Face: [AMIXXM/Donna](https://huggingface.co/spaces/AMIXXM/Donna).

---

## System Requirements

| Resource | Requirement |
|----------|-------------|
| **OS** | Windows 10 / 11 (recommended for tray, Startup, and OS automation) |
| **Python** | 3.10+ (3.11+ recommended) |
| **GPU** | NVIDIA RTX with **8GB+ VRAM** recommended (Whisper / YOLO / vision) |
| **CUDA** | **12.6** wheels via `requirements-cuda.txt` (`torch==2.13.0+cu126`); HF ZeroGPU Spaces omit CUDA pins and use the preinstalled runtime |
| **CPU fallback** | Supported; vision / Whisper are slower (`run.py` warns and continues) |
| **Storage** | ~15GB+ free for venv, PyTorch CUDA wheels, and local model weights |
| **Runtime** | [Ollama](https://ollama.com/) with local models (e.g. `llama3.2`, `deepseek-r1:8b`) |

---

## Quickstart / Local Installation

Copy-paste on Windows (PowerShell):

```powershell
git clone https://github.com/Cascade-Router/Donna.git
cd Donna
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt
ollama pull llama3.2
ollama pull deepseek-r1:8b
python run.py
```

macOS / Linux (voice tray features may differ):

```bash
git clone https://github.com/Cascade-Router/Donna.git
cd Donna
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt
ollama pull llama3.2
ollama pull deepseek-r1:8b
python run.py
```

`run.py` is the local entry point for Dānā. First launch configures mic/speaker into `settings.json` (gitignored). Optional Windows logon autostart:

```bash
python -m dana.tools.setup_startup install
```

That writes `start_dana.bat`, Desktop `Dana.lnk`, and the `DanaAssistant` HKCU Run key (migrating any legacy `Donna*` names). Silent teardown (no console flash): run `stop_dana.vbs` (or `stop_dana.bat`).

Open the Live Trace window from the system tray (**Open Settings**).

Dev / unit tests:

```bash
pip install -r requirements-dev.txt
pytest tests/test_router.py tests/test_environment.py -q
```

OSWorld adversarial bench (offline, seeded):

```bash
pytest tests/evals/test_osworld_bench.py -q
```

Scores land in [`tests/evals/osworld_bench_summary.json`](tests/evals/osworld_bench_summary.json). Full architectural write-up: [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md).

---

## Key Features

| Capability | Engineering win |
|---|---|
| **Instant Wake & JIT ML Pipeline** | OpenWakeWord on the critical path; Whisper STT and YOLOv8 load deferred in background / on Vision demand so cold start stays sub-second where it matters. |
| **LangGraph Orchestration** | FSM hybrid: RapidFuzz **Mailroom** (≥80% ASR match) short-circuits LLM routing; Memory Hydration → Supervisor Router (`hydrate_memory` → `planner`); minimized state with SQLite **Blackboard** off-graph memory. |
| **Transactional Shadow Workspaces** | File mutations stage under `.dana_scratch/<session_id>/` and `commit` / `rollback` atomically (`dana/exec/shadow_workspace.py`). |
| **Fatal Error → HITL Tickets** | `FATAL_EXCEPTIONS` bypass Critic retries and draft HITL tickets on the existing corridor (`dana/graph/nodes/critic.py`). |
| **Hybrid Win32 UIA + Crop & Zoom Florence-2** | UIA-first grounding; coarse Florence fallback; 15% pad + 2× upscale when edge &lt; 30 px in 1000-space (`dana/vision/hybrid_grounding.py`). |
| **Zero-Copy Buffer & Sub-Graph Retries (N=2)** | Full traces in `raw_state_buffer`; autonomous local retries before supervisor escalate (`dana/graph/buffer.py`, `dana/graph/subgraph_router.py`). |
| **Memory Compaction** | Spatial coordinate TTL **900s** + exponential decay \(W = W_0 e^{-\lambda\Delta h}\), \(\lambda=0.05\) (`dana/memory/compaction.py`). |
| **Live Trace UI** | Thread-safe CustomTkinter telemetry; structured JSONL forensics: `logs/donna_telemetry.jsonl`. |
| **Execution Jail & Single-Instance Lock** | Socket-bound process lock (`127.0.0.1:47473`) plus a filesystem execution jail so concurrent headless E2E runs cannot corrupt `task_queue.json` or `patch_ledger.md`. |

---

## Architecture at a Glance

```text
Mic / .trigger_ask / input.txt
        │
        ▼
┌───────────────────┐     ┌────────────────────┐
│  MicIngest (16k)  │────▶│  Conversation FSM  │
│  InputIngest      │     │  Mailroom ≥80%     │
└───────────────────┘     └─────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
         Chat Mode            Developer Mode         Vision / Research
      (local llama3.2)     (MoA + ReAct tools)      (Hybrid UIA / Florence)
              │               Pydantic guards              │
              │               Handoff schema               │
              └──────────┬────────────────────────────────┘
                         ▼
         Blackboard (SQLite) ← session_id
         JSONL telemetry + gui_telemetry_queue → Live Trace UI

Packages:
  dana/                 core agent, graph, vision, memory, tools
  dana_jason_loop/      Jason supervisor / critic loop
  dana_security/        AST/subprocess gates + patch_ledger.md
```

Deep dive: [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) · [`docs/architecture.md`](docs/architecture.md) · OSWorld: [`pytest tests/evals/test_osworld_bench.py`](tests/evals/test_osworld_bench.py) · Telemetry: [`docs/telemetry_and_ui.md`](docs/telemetry_and_ui.md) · Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Security: [`SECURITY.md`](SECURITY.md)

---

## Documentation / Operator Guide

- **[User Handbook](User%20Handbook.md)** — operating modes, wake/voice commands, and automated system behaviors for day-to-day use of Dānā.
- **White Paper:** [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) — production-hardened cybernetic control plane, 5-phase hardening specs, OSWorld benchmarks
- **OSWorld bench:** `pytest tests/evals/test_osworld_bench.py` → [`tests/evals/osworld_bench_summary.json`](tests/evals/osworld_bench_summary.json)
- Architecture: [`docs/architecture.md`](docs/architecture.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Telemetry & UI contract: [`docs/telemetry_and_ui.md`](docs/telemetry_and_ui.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)

---

## Runtime Boundaries (Local State)

Donna treats the repo root as the active workspace. The following are **machine-local** and excluded from Git:

| Path | Role |
|------|------|
| `execution_jail/` | Task queue + filesystem sandbox |
| `logs/` | Runtime / conversation logs |
| `vault/` / `donna_memory.enc` | Encrypted profile (legacy filename) |
| `.dana_scratch/` | Transactional shadow workspace staging |
| `.env`, `settings.json` | Secrets and device IDs |
| `*.onnx`, `*.pt`, `*.bin` | Model weights |

Do not commit these. Contributors: see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Design Principles

1. **Local-first** — cognition stays on-device via Ollama; no cloud dependency on the voice critical path.
2. **Strict state isolation** — Chat memory never pollutes ReAct/MoA context; Chat mode refuses the tool jail.
3. **Observable orchestration** — every meaningful stage can emit a Live Trace event without touching Tk from workers.
4. **Fail-closed concurrency** — a second `run.py` aborts rather than racing the jail.

---

## License & Status

Open-source under **AGPL-3.0**. Architecture notes and UI telemetry contracts in `docs/` are the source of truth for external integrators.
