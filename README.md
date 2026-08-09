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

# Dānā: Open-Source Cybernetic Multi-Agent Control Plane

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![OS: Windows 10/11](https://img.shields.io/badge/OS-Windows%2010%2F11-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-AMIXXM%2FDana-yellow)](https://huggingface.co/spaces/AMIXXM/Dana)
[![White Paper](https://img.shields.io/badge/docs-White%20Paper-0B7285)](docs/WHITE_PAPER.md)

**Bridge local LLMs (Ollama / `qwen2.5-coder:7b`), hybrid Win32 UIA + Florence-2 vision, and Mixture-of-Agents reasoning into a deterministic, low-latency voice operating system — with a CustomTkinter Live Trace UI locally and a Gradio headless bridge on Hugging Face Spaces.**

Dānā is a **production-hardened**, offline-first agentic control plane for the desktop: wake-word perception, strict mode-isolated cognition, transactional shadow workspaces, filesystem-jailed tool execution, and thread-safe telemetry. It is engineered as infrastructure — not a chatbot shell. Evaluated under adversarial OSWorld-style conditions (`pytest tests/evals/test_osworld_bench.py`).

**Deep dive:** [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) — 5-phase hardening specs, architecture topology, and OSWorld benchmarks.  
**Current architecture notes:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`docs/architecture.md`](docs/architecture.md)

Try the Gradio headless Meta-Broker dashboard on Hugging Face: [AMIXXM/Dana](https://huggingface.co/spaces/AMIXXM/Dana) (`app.py` → `dana/web/headless_bridge.py`).

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
| **Runtime** | [Ollama](https://ollama.com/) with local models (e.g. `qwen2.5-coder:7b`, `llama3.2`) |

---

## Quickstart / Local Installation

Copy-paste on Windows (PowerShell):

```powershell
git clone https://github.com/Cascade-Router/Dana.git
cd Dana
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt
ollama pull qwen2.5-coder:7b
python run.py
# Headless (no Tkinter):
python run.py --no-gui
```

macOS / Linux (voice tray features may differ):

```bash
git clone https://github.com/Cascade-Router/Dana.git
cd Dana
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt
ollama pull qwen2.5-coder:7b
python run.py --no-gui
```

`run.py` is the local entry point for Dānā. First launch configures mic/speaker into `settings.json` (gitignored). Optional Windows logon autostart:

```bash
python -m dana.tools.setup_startup install
```

That writes `scripts/launchers/start_dana.bat` (plus a thin root wrapper), Desktop `Dana.lnk`, and the `DanaAssistant` HKCU Run key (migrating any legacy `Dana*` names). Silent teardown (no console flash): run `scripts/launchers/stop_dana.vbs` (or the root `stop_dana.vbs` / `stop_dana.bat` wrappers).

Open the Live Trace window from the system tray (**Open Settings**). The Settings tab's **Integrations Setup** button walks through getting Pushover and Telegram credentials into `.env` for remote notifications/2-way chat.

Dev / unit tests:

```bash
pip install -r requirements-dev.txt
pytest tests/test_router.py tests/test_environment.py tests/web/test_headless_bridge.py -q
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
| **Isolated Meta-Broker** | Multi-epic plans run in a `multiprocessing.Process` with non-blocking Queue IPC (`dana/graph/meta_broker_process.py`); parent UI/headless drainers never deadlock on a full pipe. |
| **Inter-epic `manifest.json`** | AST export contract under `.dana_scratch/manifest.json` (`ClassDef` + `FunctionDef`) prepended to the next epic prompt (`dana/graph/artifact_manifest.py`). |
| **Zero keep-alive + GC** | `DANA_OLLAMA_KEEP_ALIVE=0` unloads models between hops; `gc.collect()` between Meta-Broker epics to reclaim RAM under AST load. |
| **TTSManager** | Single thread-safe `speech_queue` + daemon Piper consumer (`dana/audio/tts_manager.py`); system notifications never overlap. |
| **Gradio HF bridge** | Tkinter-free Space UI (`app.py`) submits prompts via `dana/web/headless_bridge.py` and streams telemetry into Status / Task Tracker panels. |
| **Instant Wake & JIT ML Pipeline** | OpenWakeWord on the critical path; Whisper STT and YOLOv8 load deferred in background / on Vision demand so cold start stays sub-second where it matters. |
| **LangGraph Orchestration** | FSM hybrid: RapidFuzz **Mailroom** (≥80% ASR match) short-circuits LLM routing; Memory Hydration → Supervisor Router (`hydrate_memory` → `planner`); minimized state with SQLite **Blackboard** off-graph memory. |
| **Transactional Shadow Workspaces** | File mutations stage under `.dana_scratch/<session_id>/` and `commit` / `rollback` atomically (`dana/exec/shadow_workspace.py`). |
| **Fatal Error → HITL Tickets** | `FATAL_EXCEPTIONS` bypass Critic retries and draft HITL tickets on the existing corridor (`dana/graph/nodes/critic.py`). |
| **Hybrid Win32 UIA + Crop & Zoom Florence-2** | UIA-first grounding; coarse Florence fallback; 15% pad + 2× upscale when edge &lt; 30 px in 1000-space (`dana/vision/hybrid_grounding.py`). Set `DANA_DEBUG_VISION=1` to enable ROI debug windows. |
| **Zero-Copy Buffer & Sub-Graph Retries (N=2)** | Full traces in `raw_state_buffer`; autonomous local retries before supervisor escalate (`dana/graph/buffer.py`, `dana/graph/subgraph_router.py`). |
| **Memory Compaction** | Spatial coordinate TTL **900s** + exponential decay \(W = W_0 e^{-\lambda\Delta h}\), \(\lambda=0.05\) (`dana/memory/compaction.py`). |
| **Live Trace UI** | Thread-safe CustomTkinter telemetry; structured JSONL forensics: `logs/dana_telemetry.jsonl`. |
| **Execution Jail & Single-Instance Lock** | Socket-bound process lock (`127.0.0.1:47473`) plus a filesystem execution jail so concurrent headless E2E runs cannot corrupt `task_queue.json` or `patch_ledger.md`. |

---

## Capabilities

| Pillar | What it means |
|---|---|
| **Vision Grounding** | Hybrid Win32 UI-Automation + Florence-2 grounding locates on-screen elements by plain-language description (`dana/vision/hybrid_grounding.py`, `dana/graph/nodes/vision.py`) — UIA hit-test first, coarse Florence-2 phrase-grounding fallback, crop + 2× zoom re-ground when the matched box is small. Grounding results are normalized ``[0, 1000]`` boxes, rescaled to real screen pixels before any actuation. |
| **Win32 Actuation** | Every physical action — mouse, keyboard, scroll, drag-and-drop — is raw `ctypes` `SendInput`; no `pyautogui`/`pynput` anywhere on the input path (`dana/tools/os_control.py`, `mouse_actuator.py`, `keyboard_actuator.py`, `scroll_actuator.py`, `drag_actuator.py`). Every actuator shares one safety pipeline: `DANA_OS_DRY_RUN` dry-run mode, module-wide rate limiting, failsafe bounds checks, and a kill-switch check immediately before physical input. |
| **Workspace Orchestration** | Window management (`list_active_windows` / `focus_window`, `dana/tools/window_actuator.py`) confirms the right application is in the foreground before Dana acts on it. Clipboard I/O (`read_clipboard` / `write_clipboard`, `dana/tools/clipboard_actuator.py`) plus `press_keyboard_shortcut` (`dana/tools/keyboard_actuator.py`) extract or inject exact text via the OS clipboard — precise where vision OCR would be lossy. |
| **Remote Telegram / Pushover Access** | `send_notification` (`dana/tools/notifications.py`) pushes phone alerts via Pushover. A Telegram bot (`dana/middleware/telegram_poller.py`) long-polls for messages from one allowlisted chat id and routes them into Dana's normal input pipeline exactly like typed UI input, replying back over the same chat — a 2-way remote channel for when Dana is running unattended in the background. Setup steps for both live in the desktop app's Settings tab (`dana/ui/settings.py`). |

---

## Architecture at a Glance

```text
Mic / .trigger_ask / input.txt / Gradio (HF Space)
        │
        ▼
┌───────────────────┐     ┌────────────────────┐
│  MicIngest (16k)  │────▶│  Conversation FSM  │
│  InputIngest      │     │  Mailroom ≥80%     │
└───────────────────┘     └─────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
         Chat Mode            Developer Mode         Meta-Broker
      (local Ollama)       (MoA + ReAct tools)   (isolated Process)
              │               Pydantic guards         │
              │               Handoff schema          ▼
              │                              Queue IPC telemetry
              │                              → Task Tracker / Gradio
              └──────────┬────────────────────────────────┘
                         ▼
         Blackboard (SQLite) ← session_id
         TTSManager.speech_queue → Piper (sequential)
         JSONL telemetry + gui_telemetry_queue → Live Trace UI

Packages:
  dana/                 core agent, graph, vision, memory, tools, web
  dana/web/             Gradio / HF headless bridge (no Tkinter)
  dana_jason_loop/      Jason supervisor / critic loop
  dana_security/        AST/subprocess gates + patch_ledger.md
  legacy/               Archived scratch (not on the critical path)
```

Deep dive: [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) · [`docs/architecture.md`](docs/architecture.md) · Legal/IP: [`docs/LEGAL_AND_IP.md`](docs/LEGAL_AND_IP.md) · License audit: [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md) · OSWorld: [`pytest tests/evals/test_osworld_bench.py`](tests/evals/test_osworld_bench.py) · Telemetry: [`docs/telemetry_and_ui.md`](docs/telemetry_and_ui.md) · Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Security: [`SECURITY.md`](SECURITY.md)

---

## Documentation / Operator Guide

- **[User Handbook](docs/user_guide/User%20Handbook.md)** — operating modes, wake/voice commands, and automated system behaviors for day-to-day use of Dānā.
- **Legal & IP:** [`docs/LEGAL_AND_IP.md`](docs/LEGAL_AND_IP.md) — product branding, Class 009/042 scope, bundled-model posture, and license audit linkage
- **License audit:** [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md) — third-party package inventory (includes GPL/AGPL flags)
- **White Paper:** [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) — production-hardened cybernetic control plane, 5-phase hardening specs, OSWorld benchmarks
- **OSWorld bench:** `pytest tests/evals/test_osworld_bench.py` → [`tests/evals/osworld_bench_summary.json`](tests/evals/osworld_bench_summary.json)
- Architecture: [`docs/architecture.md`](docs/architecture.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Telemetry & UI contract: [`docs/telemetry_and_ui.md`](docs/telemetry_and_ui.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)

---

## Runtime Boundaries (Local State)

Dana treats the repo root as the active workspace. The following are **machine-local** and excluded from Git:

| Path | Role |
|------|------|
| `execution_jail/` | Task queue + filesystem sandbox |
| `logs/` | Runtime / conversation logs |
| `vault/` / `dana_memory.enc` | Encrypted profile (legacy filename) |
| `.dana_scratch/` | Transactional shadow workspace + `manifest.json` contracts |
| `.dana/` | Local vault / runtime mirrors |
| `.env`, `settings.json` | Secrets and device IDs |
| `*.onnx`, `*.pt`, `*.bin` | Model weights |
| `*.db` / Chroma dirs | Local SQLite + vector stores |

Do not commit these. Contributors: see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Design Principles

1. **Local-first** — cognition stays on-device via Ollama; no cloud dependency on the voice critical path.
2. **Strict state isolation** — Chat memory never pollutes ReAct/MoA context; Chat mode refuses the tool jail.
3. **Observable orchestration** — every meaningful stage can emit a Live Trace / Gradio telemetry event without touching Tk from workers.
4. **Fail-closed concurrency** — a second `run.py` aborts rather than racing the jail.
5. **Stdlib-first epics** — Meta-Broker codegen prefers the Python standard library unless the prompt explicitly requests third-party packages.

---

## License & Status

Open-source under **AGPL-3.0**. Architecture notes and UI telemetry contracts in `docs/` are the source of truth for external integrators.
