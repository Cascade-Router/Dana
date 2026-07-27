# Dānā (دانا) — Hugging Face Space Deployment (Stage 9.2)

Gradio wrapper that exposes the LangGraph text corridor as::

    POST /api/predict
    Content-Type: application/json
    {"data": ["your prompt here"]}

This matches `website/src/utils/hf_api.ts` (Astro global chat).

## Files

| Path | Role |
|------|------|
| `deploy/hf_app.py` | Gradio `Interface` (text → text) |
| `deploy/cloud_bridge.py` | Cloud env + mocked actuators + `run_react_loop` |
| `deploy/requirements.txt` | Lean HF deps (no CustomTkinter / Piper / YOLO) |
| `app.py` | Repo-root entry for Spaces that expect `app.py` |

## Create the Space

1. New Hugging Face Space → SDK **Gradio** → Hardware CPU (or GPU if you later add local models).
2. Set **App file** to `app.py` (repo root) or `deploy/hf_app.py`.
3. Set **Requirements** to `deploy/requirements.txt`.
4. Add **Secrets** (Settings → Variables and secrets) — see below.
5. Point the Astro site at the Space root (no `/api/predict` suffix)::

       PUBLIC_DANA_HF_API=https://YOUR_USER-YOUR_SPACE.hf.space

## Secrets / environment

| Secret | Required | Purpose |
|--------|----------|---------|
| `OPENAI_API_KEY` | Recommended | OpenAI / compatible Chat completions for MoA ReAct |
| `GROQ_API_KEY` | Alt | Auto-wires Groq OpenAI-compatible base URL |
| `DEEPSEEK_API_KEY` | Alt | Auto-wires `https://api.deepseek.com/v1` |
| `ANTHROPIC_API_KEY` | Optional | Reserved for future Anthropic cascade |
| `DONNA_CLOUD_MODEL` | Optional | Model id (default `gpt-4o-mini` or provider default) |
| `OPENAI_BASE_URL` / `DONNA_OPENAI_BASE_URL` | Optional | Custom OpenAI-compatible endpoint |
| `OLLAMA_URL` | Optional | Remote Ollama instead of cloud LLM |
| `DONNA_VAULT_KEY` | Optional | Avoids interactive vault unlock (default ephemeral) |
| `DONNA_HITL_AUTO_APPROVE` | Auto-set | `1` — Spaces have no HITL GUI |
| `DONNA_OS_DRY_RUN` | Auto-set | `1` — no desktop keystrokes |
| `DONNA_CLOUD` | Auto-set | `1` — enables mocks |

At least one LLM path must work: **OpenAI / Groq / DeepSeek key** **or** a reachable **`OLLAMA_URL`**.

## Cloud mocks

Vision (`analyze_visual_context`, `ocr_with_region`) and desktop actuators
(`navigate_and_click`, `press_key`, `type_stealth_text`, …) return
`[CLOUD] … mocked` instead of touching a webcam / Win32 stack.

`draft_cursor_prompt` still writes to the Space filesystem under
`donna_security/patch_ledger.md` when validation passes.

## Local smoke

```bash
pip install -r deploy/requirements.txt
# plus your preferred venv with the Donna package importable
set DONNA_CLOUD=1
set OPENAI_API_KEY=sk-...
python deploy/hf_app.py
# open http://127.0.0.1:7860 — or curl:
curl -X POST http://127.0.0.1:7860/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"data\":[\"Hello Dānā\"]}"
```

## Astro wiring

```bash
# website/.env
PUBLIC_DANA_HF_API=https://YOUR_USER-YOUR_SPACE.hf.space
```

Cold boots may return 502/503; the website client already surfaces
“Dānā is warming up…”.
