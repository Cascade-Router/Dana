import { useCallback, useEffect, useState } from "react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import { maskSecret } from "../secrets/types";
import "./EnvViewerWidget.css";

type EnvSnapshot = Record<string, string>;

const PROVIDERS: { key: string; label: string }[] = [
  { key: "GROQ_API_KEY", label: "Groq" },
  { key: "GEMINI_API_KEY", label: "Gemini" },
  { key: "OPENAI_API_KEY", label: "OpenAI" },
  { key: "ANTHROPIC_API_KEY", label: "Anthropic" },
];

// Gradio mode (app.py's bare gr.Blocks "chat" endpoint) never mounts the
// FastAPI app dana/api/system.py's GET/POST /api/system/env lives on — see
// apiBase.ts's IS_GRADIO_MODE, which rejects every /api/* call outright
// there. There's also no wire path today for the Gradio "chat"/"artifacts"
// api endpoints to accept extra per-request keys (its public predict()
// signature is just `{ message }` — see gradioChatClient.ts), so a key
// saved here can't reach the HF Space's own model calls yet; it's stored
// so THIS browser has it on hand for anything client-side that can use
// it, and so this modal never has to show a REST error to get there.
const CLOUD_KEYS_STORAGE_KEY = "dana:cloud_env_keys";

function readCloudKeys(): EnvSnapshot {
  try {
    const raw = localStorage.getItem(CLOUD_KEYS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeCloudKeys(keys: EnvSnapshot): void {
  try {
    localStorage.setItem(CLOUD_KEYS_STORAGE_KEY, JSON.stringify(keys));
  } catch {
    // Best-effort — a private-browsing tab or storage-disabled browser
    // just won't persist across reloads; saving still updates local state
    // for the rest of this session.
  }
}

// Read/write view of GET+POST /api/system/env (dana/api/system.py) — every
// sensitive value is already masked server-side against a fixed allowlist
// before it's ever serialized, so this component never receives, stores, or
// could possibly render a raw secret; saving a NEW value is a one-way POST,
// never round-tripped back in full either. In Gradio mode (IS_GRADIO_MODE)
// that REST endpoint doesn't exist at all — see readCloudKeys/writeCloudKeys
// above — so this falls back to a browser-localStorage-backed copy instead
// of ever surfacing a REST error.
export function EnvViewerWidget({ onClose }: { onClose: () => void }) {
  const [env, setEnv] = useState<EnvSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [saveKey, setSaveKey] = useState(PROVIDERS[0].key);
  const [saveValue, setSaveValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<{ valid: boolean; detail: string } | null>(null);

  const refresh = useCallback(() => {
    if (IS_GRADIO_MODE) {
      // No REST endpoint reachable at all in this build (see apiBase.ts) —
      // read straight from this browser's own localStorage instead of ever
      // attempting the doomed fetch.
      setEnv(readCloudKeys());
      return;
    }
    apiFetch("/api/system/env")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setEnv(data.env ?? {}))
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const rows = env ? Object.entries(env).sort(([a], [b]) => a.localeCompare(b)) : [];

  const handleSave = useCallback(() => {
    const value = saveValue.trim();
    if (!value) return;
    if (IS_GRADIO_MODE) {
      // No server round-trip to "validate" against here — persist locally
      // and confirm the save itself, rather than a REST call that would
      // always fail.
      const next = { ...readCloudKeys(), [saveKey]: value };
      writeCloudKeys(next);
      setEnv(next);
      setSaveResult({ valid: true, detail: "Saved locally in this browser." });
      setSaveValue("");
      return;
    }
    setSaving(true);
    setSaveResult(null);
    apiFetch("/api/system/env", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: saveKey, value }),
    })
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.detail || "save failed");
        setSaveResult({ valid: Boolean(data.valid), detail: String(data.detail || "") });
        setSaveValue("");
        setEnv(data.env ?? null);
      })
      .catch((err) => setSaveResult({ valid: false, detail: String(err instanceof Error ? err.message : err) }))
      .finally(() => setSaving(false));
  }, [saveKey, saveValue]);

  return (
    <div className="env-viewer">
      <div className="env-viewer__backdrop" onClick={onClose} />
      <div className="env-viewer__panel" role="dialog" aria-label="Environment Variables">
        <div className="env-viewer__header">
          <h2>Environment</h2>
          <button type="button" className="env-viewer__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="env-viewer__badges">
          {PROVIDERS.map((p) => {
            const configured = Boolean(env?.[p.key]);
            return (
              <div key={p.key} className={`env-viewer__badge ${configured ? "env-viewer__badge--on" : ""}`}>
                <span className="env-viewer__badge-dot" />
                {p.label}
              </div>
            );
          })}
        </div>

        <div className="env-viewer__save">
          <select
            className="env-viewer__save-select"
            value={saveKey}
            onChange={(e) => setSaveKey(e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}
              </option>
            ))}
          </select>
          <input
            className="env-viewer__save-input"
            type="password"
            placeholder="Paste API key…"
            value={saveValue}
            onChange={(e) => setSaveValue(e.target.value)}
          />
          <button
            type="button"
            className="env-viewer__save-btn"
            onClick={handleSave}
            disabled={saving || !saveValue.trim()}
          >
            {saving ? "Saving…" : IS_GRADIO_MODE ? "Save" : "Save & Validate"}
          </button>
        </div>
        {saveResult && (
          <div className={`env-viewer__save-result ${saveResult.valid ? "" : "env-viewer__save-result--bad"}`}>
            {saveResult.detail}
          </div>
        )}

        <div className="env-viewer__list">
          {error && <div className="env-viewer__error">Failed to load: {error}</div>}
          {!env && !error && <div className="env-viewer__empty">Loading…</div>}
          {env && rows.length === 0 && <div className="env-viewer__empty">No variables set.</div>}
          {rows.map(([key, value]) => (
            <div key={key} className="env-viewer__row">
              <span className="env-viewer__key">{key}</span>
              {/* REST mode already receives pre-masked values server-side
                  (see this component's header comment) — Gradio mode holds
                  the real value locally, so mask it here instead, the same
                  way it's never shown in full anywhere else in this app. */}
              <span className="env-viewer__value">{IS_GRADIO_MODE ? maskSecret(value) : value}</span>
            </div>
          ))}
        </div>

        <div className="env-viewer__footer">
          {IS_GRADIO_MODE
            ? "In Cloud mode, API keys are stored in your browser session or configured via Hugging Face Space Secrets."
            : "Values are masked server-side before they ever leave the backend."}
        </div>
      </div>
    </div>
  );
}
