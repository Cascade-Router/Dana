# Documentation index

Start with [`../ARCHITECTURE.md`](../ARCHITECTURE.md): the current, maintained
description of Dānā — an agentic CAD and manufacturing engineer (FastAPI +
WebSocket ReAct loop, FreeCAD via the Universal CAD IR, zero-token parametric
rebuilds, per-session persistence, Tauri/React UI).

Much of what is in this folder predates the move to CAD and describes the
earlier desktop voice-agent ("cybernetic control plane") design. Those files
are kept for history; read them as such.

## Current

| File | About |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System architecture and hard constraints (also returned to the agent by `read_system_architecture`) |
| [`../SYSTEM_ARCHITECTURE_MAP.md`](../SYSTEM_ARCHITECTURE_MAP.md) | Detailed module map from the 2026-09 audit |
| [`plugins.md`](plugins.md) | Writing a plugin (`manifest.json` + handlers) |
| [`safety_and_hitl.md`](safety_and_hitl.md) | Dry-run, HITL approval and the panic switch |
| [`setup.md`](setup.md) | Developer setup |
| [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) | Dependency licences (2026-07 snapshot) |

## Historical (pre-CAD design)

| File | Note |
|---|---|
| [`architecture.md`](architecture.md) | The "local-first agentic voice OS" architecture |
| [`WHITE_PAPER.md`](WHITE_PAPER.md) | "Cybernetic control plane" white paper; references removed packages |
| [`SYSTEM_ARCHITECTURE_AUDIT.md`](SYSTEM_ARCHITECTURE_AUDIT.md) | Marked stale in place (2026-08-25) |
| [`architecture/`](architecture/) | LangGraph-era maps and the Phase 7 decomposition plan |
| [`telemetry_and_ui.md`](telemetry_and_ui.md) | The removed CustomTkinter UI |
| [`LEGAL_AND_IP.md`](LEGAL_AND_IP.md) | Positioning written for the control-plane product |
| [`patch_ledger.md`](patch_ledger.md), [`patch_ledger_archive.md`](patch_ledger_archive.md) | Tickets from the old self-healing ledger |
