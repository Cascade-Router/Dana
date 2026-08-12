# Developer Setup

This is the contributor-focused companion to the top-level
[`README.md`](../README.md) Quickstart — it covers environment keys, feature
toggles, linting, and the test execution protocol in more depth. For a first
install, follow the README's Quickstart first; come back here once you're
changing code.

## 1. System requirements

See [`README.md` § System Requirements](../README.md#system-requirements)
for the full table (Windows 10/11 recommended, Python 3.10+, optional CUDA
GPU, [Ollama](https://ollama.com/) for local models). This repo targets
Python 3.12 for tooling (`pyproject.toml`'s `[tool.mypy]`), but supports
3.10+ at runtime.

## 2. Install

```powershell
git clone https://github.com/Cascade-Router/Dana.git
cd Dana
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt   # optional, local NVIDIA CUDA builds only
pip install -r requirements-dev.txt    # pytest + dev-only tooling
ollama pull qwen2.5-coder:7b
```

`requirements.txt` intentionally omits `torch` (Hugging Face ZeroGPU Spaces
supply their own runtime build) — install `requirements-cuda.txt` for a local
GPU box.

## 3. Environment keys

Dānā is configured almost entirely through environment variables (checked at
call sites, not a single central config object) plus two persisted JSON
files. Secrets and per-machine values (Pushover/Telegram credentials, cloud
API keys) live in a gitignored `.env` at the repo root — the Settings tab's
**Integrations Setup** button walks through populating it. The tables below
list the toggles most relevant to day-to-day development; grep
`os.environ.get("DANA_` across `dana/` for the complete, current list before
relying on this table for anything security-sensitive.

**Safety (see [`safety_and_hitl.md`](safety_and_hitl.md) for full detail):**

| Key | Effect |
|-----|--------|
| `DANA_OS_DRY_RUN` | `1`/`true`/`yes`/`on` → all Win32 actuation becomes a logged no-op |
| `DANA_HITL_TICKET` | Set `0`/`false`/`off`/`no` to disable the HITL approval gate |
| `DANA_HITL_REQUIRE_GUI` | Force tickets to block on a real GUI decision even headless |
| `DANA_HITL_AUTO_APPROVE` / `DANA_HITL_AUTO_DENY` | Deterministic ticket resolution (use in CI) |
| `DANA_KILL_HOTKEY` | Override the panic hotkey (default `f12`) |
| `DANA_DISABLE_KILL_SWITCH` | Skip arming the global hotkey listener |
| `DANA_DISABLE_TOAST` | Disable the silent-toast fallback notifications |

**Runtime / mode:**

| Key | Effect |
|-----|--------|
| `DANA_NO_GUI` / `DANA_HEADLESS` | Run without the Tkinter dashboard (`run.py --no-gui` sets this) |
| `DANA_DEBUG` / `DANA_DEBUG_VISION` / `DANA_AUDIO_PIPELINE_DEBUG` | Verbose logging for their respective subsystems |
| `DANA_CASCADE_ROUTER` | Toggle the RapidFuzz mailroom fast-path router |
| `DANA_FORCE_LOCAL` / `DANA_ALLOW_CLOUD_FALLBACK` | Control local-vs-cloud model fallback policy |

**Model selection** (each defaults to a sane local Ollama model if unset):
`DANA_OLLAMA_MODEL`, `DANA_LOCAL_MODEL`, `DANA_LIGHTWEIGHT_MODEL`,
`DANA_REASONER_MODEL`, `DANA_CASCADE_MODEL`, `DANA_VISION_MODEL`,
`DANA_DRAFT_MODEL`, plus cloud-provider overrides
(`DANA_OPENAI_MODEL`, `DANA_ANTHROPIC_MODEL`, `DANA_GEMINI_MODEL`,
`DANA_GROQ_MODEL`, `DANA_CLOUD_PROVIDER`).

**Feature flags (persisted, not env-only):** `feature_flags.json` at the
project root — see [`architecture.md` §8](architecture.md#8-feature-manager--env-key-toggles).
Delete it (or use the Settings UI toggle) to fall back to
auto-detected defaults per feature.

## 4. Linting

```bash
ruff check dana/
```

Config lives in `pyproject.toml`'s `[tool.ruff]` / `[tool.ruff.lint]`
(`select = ["E", "F", "I", "UP", "B"]`, line length 100, `E501` and `B008`
ignored intentionally). Run `ruff check dana/ --fix` to auto-apply safe
fixes (import sorting, unused-import removal) before hand-fixing anything
`ruff` flags as unsafe to auto-fix (undefined names, unused locals,
redefinitions).

## 5. Running tests

```bash
pytest tests/
```

`pyproject.toml`'s `[tool.pytest.ini_options]` sets `testpaths = ["tests"]`
and adds `scripts` / `scripts/diagnostics` to `pythonpath`, so diagnostic
helper modules import cleanly from tests without a separate install step.

Useful narrower runs while iterating:

```bash
pytest tests/test_router.py tests/test_environment.py tests/web/test_headless_bridge.py
pytest tests/evals/test_osworld_bench.py   # offline, seeded adversarial bench
pytest -m e2e                              # live OS/vision/browser tests (see below)
```

The `e2e` marker (declared in `pyproject.toml`) is for tests that perform
live end-to-end OS, vision, or browser manipulation — they are **not** run
by a bare `pytest tests/` unless you explicitly select the marker, since they
assume a real desktop session (a display, an unlocked vault, sometimes a
running Ollama instance) rather than a headless CI sandbox.

Before opening a PR: `ruff check dana/` and `pytest tests/` should both be
clean. If you touched actuator/plugin code, also set `DANA_OS_DRY_RUN=1` and
manually exercise the affected tool once — the safety layers in
[`safety_and_hitl.md`](safety_and_hitl.md) are tested in isolation, but a
dry run of your actual change is still the fastest way to catch a bad
argument mapping before it moves a real window.
