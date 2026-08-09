# Contributing to Dana

Thank you for helping harden a local-first agentic voice OS. Dana prioritizes **low-latency infrastructure**, **strict state isolation**, and **observable graph orchestration**. PRs that bypass verification or mutate local-state contracts will be rejected.

---

## Ground Rules

1. **Do not commit local state** — `vault/`, `execution_jail/`, `logs/`, `.env`, `settings.json`, `*.onnx`, `*.pt`, `*.bin`, ledgers, and device configs are gitignored for a reason.
2. **Respect mode isolation** — Chat mode must not gain tool-jail side effects; Developer paths must not leak into the chat memory buffer.
3. **Tk is main-thread only** — Background workers update the Live Trace exclusively through `emit_trace()` → `gui_telemetry_queue` (see [`docs/telemetry_and_ui.md`](docs/telemetry_and_ui.md)).
4. **Single instance** — Never run two `run.py` processes against the same workspace; the socket lock exists to protect the jail.
5. **Do not modify** ToolForge security gates, offline routing structure, or `dana/paths.py` unless a maintainer explicitly assigns that work.

---

## Code formatting

- Prefer **minimal diffs**: change only what the ticket or PR requires; avoid drive-by renames and cleanups.
- Match surrounding style (imports, typing, logging prefixes).
- Keep production paths fail-closed; do not weaken HITL ticket gates.
- No secrets in source, tests, or docs.

---

## Local setup & unit tests

```bash
git clone https://github.com/Cascade-Router/Dana.git
cd Dana
python -m venv .venv

# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-cuda.txt
pip install -r requirements-dev.txt

# Focused corridor / env guards (fast)
pytest tests/test_router.py tests/test_environment.py -q

# Broader suite (may need full local deps / GPU stack)
pytest -q
```

Entry point for a normal desktop run:

```bash
python run.py
```

---

## Pull request process

1. Branch from `main`; keep commits focused.
2. Run the relevant unit tests locally (`pytest` as above).
3. Open a PR against `main` with a short summary and test notes.
4. For tool / LangGraph / jail wiring changes, also complete the **Headless E2E** gate below.

### Required Gate: Headless E2E State Reachability Analysis

Any PR that adds or changes tool calls, Intent Broker / Cascade routes, LangGraph or ReAct node branches, mode fast-paths, or ingest → jail wiring **must** demonstrate a passing **Headless E2E State Reachability Analysis** before review.

#### Method (InputIngest)

1. Ensure a **single** Dana instance is running (`python run.py`).
2. For Chat / mode switches, inject via `.trigger_ask` (Main file trigger).
3. For Developer tool / jail paths: switch to developer mode, write the command into `execution_jail/input.txt`, and wake the session so `task_queue.json` drains.
4. Verify in `logs/dana_runtime.log` (and Live Trace where applicable): expected mode transitions, tool execution, and no dual-writer corruption.

PRs without evidence of this headless reachability pass (log excerpts, harness output, or equivalent CI) will not be merged.

---

## Pull Request Checklist

- [ ] No secrets or local state files staged
- [ ] Unit tests added/updated when changing routing, HITL, or startup guards
- [ ] `pytest` relevant modules pass locally
- [ ] Headless E2E State Reachability Analysis passed for new tools / node branches (`input.txt` injection path)
- [ ] Live Trace emissions are queue-based (no cross-thread Tk)
- [ ] Mode / jail invariants preserved
- [ ] Docs updated if behavior or contracts changed

---

## Security

See [`SECURITY.md`](SECURITY.md) for local privacy guarantees and private vulnerability reporting.

---

## Code of Conduct (engineering)

Prefer minimal diffs, explicit failure modes, and telemetry over silent fallbacks. Dana is infrastructure: correctness under concurrency beats feature surface area.
