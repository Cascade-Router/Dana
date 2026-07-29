# Dānā — Legal Positioning & IP Declaration

**Product branding:** Dānā AI Control Plane  
**Short description:** Multi-Agent AI Orchestrator & Local REPL Control Plane  
**Package positioning:** *Dānā: Open-Source Cybernetic Multi-Agent Control Plane*

This document states product scope, trademark-style branding, and license posture for bundled runtime models versus third-party Python packages. It is **not** legal advice. Operators redistributing binaries or SaaS builds must review each upstream LICENSE / MODEL_CARD themselves.

---

## Nice classification alignment (product scope)

| Class | Scope definition |
|---|---|
| **Class 009** | Downloadable multi-agent artificial intelligence orchestrator software and local REPL execution sandboxes. |
| **Class 042** | Non-downloadable cloud platform software for AI agent orchestration and tool synthesis. |

Dānā’s open-source tree primarily implements the Class 009 local control-plane / REPL sandbox story. Any Class 042 cloud offering is a separate distribution surface and must carry its own dependency and model-license review.

---

## Distribution intent

1. **Default TTS voice switched away from non-commercial dataset licenses.**  
   The previous default Piper voice `en_US-hfc_female-medium` is licensed **CC BY-NC-SA 4.0** (non-commercial). The runtime default is now **`en_US-ljspeech-high`** (LJ Speech dataset — **public domain** per the Piper MODEL_CARD).
2. **`en_US-lessac-medium` is intentionally not the commercial default.**  
   Piper’s Lessac MODEL_CARD points at the Blizzard 2013 / Lessac research licence, which is limited to **research purposes** and excludes commercial voice-synthesis product use. Operators may still override via `DONNA_PIPER_VOICE=en_US-lessac-medium` for local research, but that is not the product default.
3. **Legacy migration.**  
   If the preferred voice cannot be downloaded (offline / mocked tests) and `en_US-hfc_female-medium` is already on disk, the spooler may fall back to that file and log a warning. Legacy NC weights must not be redistributed in commercial packages.
4. **Third-party Python packages are not uniformly MIT/Apache/PD.**  
   See [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) for the full inventory and restrictive flags.

---

## Bundled / runtime models (commercially oriented defaults)

These are the models Dānā intends to load by default for local inference. Licenses below are as declared by upstream at audit time; re-verify before shipping a commercial build.

| Asset | Role | Upstream license posture (summary) |
|---|---|---|
| **Piper `en_US-ljspeech-high`** | Default offline TTS voice weights | Dataset: **public domain** (LJ Speech). Piper runtime package is separate (see audit). |
| **Distil-Whisper** (`distil-whisper/distil-small.en`) | Speech-to-text | **MIT** (Hugging Face Distil-Whisper) |
| **Silero VAD** | Voice activity detection | **MIT** / BSD-style as published by snakers4/silero-vad |
| **Florence-2** (Microsoft) | Vision grounding fallback | **MIT** (microsoft/Florence-2 family cards) |

**Not the commercial default:**

| Asset | Why |
|---|---|
| Piper `en_US-hfc_female-medium` | **CC BY-NC-SA 4.0** — non-commercial share-alike |
| Piper `en_US-lessac-medium` | Blizzard 2013 **research** licence — excludes commercial TTS products |

Model weight files under `tts_models/`, `*.onnx`, `*.pt`, and similar paths remain **gitignored** machine-local artifacts.

---

## Third-party package license flags

A venv scan (`pip-licenses`) is recorded in [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md). **Do not claim that every dependency is MIT / Apache / Public Domain.**

Restrictive packages flagged at last audit (2026-07-29) include:

| Package | License flag |
|---|---|
| `piper-tts` | **GPL-3.0-or-later** |
| `MouseInfo` | **GPL-3.0-or-later** |
| `PyMsgBox` | **GPL-3.0-or-later** |
| `pylint` | **GPL-2.0-or-later** (dev tooling) |
| `ultralytics` / `ultralytics-thop` | **AGPL-3.0-or-later** |

Project source is published under **AGPL-3.0** (see repository `LICENSE` / README badge). Combining AGPL application code with other GPL/AGPL transitive deps requires a coherent redistribution strategy (source offer, network-copyleft awareness for AGPL, etc.).

---

## Operator checklist

- [ ] Confirm default Piper files are `en_US-ljspeech-high.onnx` + `.onnx.json` (or a deliberately chosen override).
- [ ] Do not ship `hfc_female` or Lessac weights in a commercial redistribution without separate counsel review.
- [ ] Re-run / refresh [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) when locking a release venv.
- [ ] Review YOLO / Ultralytics AGPL obligations if those weights remain in the shipped graph.

---

## Related docs

- [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) — full package inventory + GPL/AGPL flags  
- [`../README.md`](../README.md) — product overview  
- [`architecture.md`](architecture.md) — technical control paths  
- [`WHITE_PAPER.md`](WHITE_PAPER.md) — hardening narrative  
