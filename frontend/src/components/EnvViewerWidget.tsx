import { useCallback, useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import { maskSecret } from "../secrets/types";
import { ToggleSwitch } from "./ToggleSwitch";
import "./EnvViewerWidget.css";

type EnvSnapshot = Record<string, string>;
type ValidationState = { valid: boolean; detail: string; checking?: boolean };

const PROVIDERS: { key: string; label: string }[] = [
  { key: "GROQ_API_KEY", label: "Groq" },
  { key: "GEMINI_API_KEY", label: "Gemini" },
  { key: "OPENAI_API_KEY", label: "OpenAI" },
  { key: "OPENROUTER_API_KEY", label: "OpenRouter" },
  { key: "ANTHROPIC_API_KEY", label: "Anthropic" },
];

// Model Priority Manager: the Cloud Provider Manager's dropdown, mapping
// straight onto dana.core.model_provider.cloud_provider_name()'s
// recognized values (DANA_CLOUD_PROVIDER) — the same three cloud options
// the task's own credential list above covers via OPENAI_API_KEY/
// OPENROUTER_API_KEY/GEMINI_API_KEY.
const CLOUD_PROVIDERS: { key: string; label: string }[] = [
  { key: "openai", label: "OpenAI" },
  { key: "openrouter", label: "OpenRouter" },
  { key: "gemini", label: "Gemini" },
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
export function EnvViewerWidget({
  onClose,
  onSessionsCleared,
  autoApprove,
  onToggleAutoApprove,
}: {
  onClose: () => void;
  /** Called after "Clear All Sessions" successfully deletes every persisted
   * chat — lets App.tsx reset its own chat state (new chat, sidebar refresh)
   * instantly rather than leaving stale session data on screen. */
  onSessionsCleared?: () => void;
  /** useChat's own auto_approve toggle state (dana/api/server.py's
   * session["auto_approve"], mirrored locally — see useChatSocket.ts). */
  autoApprove: boolean;
  onToggleAutoApprove: (enabled: boolean) => void;
}) {
  const [env, setEnv] = useState<EnvSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validation, setValidation] = useState<Record<string, ValidationState>>({});

  const [saveKey, setSaveKey] = useState(PROVIDERS[0].key);
  const [saveValue, setSaveValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<{ valid: boolean; detail: string } | null>(null);

  // Model Priority Manager: Cloud Provider + Local Model name, both plain
  // routing config (DANA_CLOUD_PROVIDER/DANA_LOCAL_MODEL — already in
  // dana/api/system.py's _NON_SENSITIVE_VARS, so they ride the same GET/POST
  // /api/system/env endpoints the credential badges above use) rather than
  // requiring a manual .env edit + backend restart.
  const [cloudProvider, setCloudProvider] = useState("");
  const [savingProvider, setSavingProvider] = useState(false);
  const [providerResult, setProviderResult] = useState<string | null>(null);

  const [localModel, setLocalModel] = useState("");
  const [savingModel, setSavingModel] = useState(false);
  const [modelResult, setModelResult] = useState<string | null>(null);

  // Whether the ReAct loop's tool-calling hot path routes to a cloud
  // provider at all (DANA_CLOUD_PRIMARY — dana.core.model_provider.
  // cloud_primary_enabled/tool_calling_provider). This has to be OFF for
  // DANA_LOCAL_MODEL to actually be used — leaving it on would make every
  // other Model Priority Manager field above cosmetic, since
  // tool_calling_provider() skips local Ollama entirely while it's set.
  const [cloudPrimary, setCloudPrimaryState] = useState(false);
  const [savingCloudPrimary, setSavingCloudPrimary] = useState(false);

  const [clearingSessions, setClearingSessions] = useState(false);
  const [clearResult, setClearResult] = useState<string | null>(null);

  // Re-pings the provider for whichever key is CURRENTLY live in
  // os.environ server-side (dana/api/system.py's POST /api/system/env/validate)
  // without ever resending the secret itself — the badges' Green/Red state,
  // separate from "configured" (has some value) vs "unconfigured" (none).
  // Gradio mode has no server to ask, so badges there only ever reflect
  // "present in localStorage," never a real validity check.
  const validateProvider = useCallback((key: string) => {
    if (IS_GRADIO_MODE) return;
    setValidation((prev) => ({ ...prev, [key]: { valid: false, detail: "", checking: true } }));
    apiFetch("/api/system/env/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (!data.configured) {
          setValidation((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });
          return;
        }
        setValidation((prev) => ({ ...prev, [key]: { valid: Boolean(data.valid), detail: String(data.detail || "") } }));
      })
      .catch((err) => {
        setValidation((prev) => ({
          ...prev,
          [key]: { valid: false, detail: String(err instanceof Error ? err.message : err) },
        }));
      });
  }, []);

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
      .then((data) => {
        const nextEnv: EnvSnapshot = data.env ?? {};
        setEnv(nextEnv);
        // Live-validate every already-configured provider on open — this is
        // what makes the badges reflect actual current validity rather than
        // just "a value is saved," which could be a stale/revoked key.
        PROVIDERS.filter((p) => Boolean(nextEnv[p.key])).forEach((p) => validateProvider(p.key));
        // Model Priority Manager: seed the Cloud Provider select / Local
        // Model input from whatever is CURRENTLY configured — once, on
        // load, not on every refresh, so this never clobbers text the user
        // is mid-typing (refresh() only ever runs on mount here; nothing
        // else calls it).
        if (nextEnv.DANA_CLOUD_PROVIDER) setCloudProvider(nextEnv.DANA_CLOUD_PROVIDER);
        if (nextEnv.DANA_LOCAL_MODEL) setLocalModel(nextEnv.DANA_LOCAL_MODEL);
        setCloudPrimaryState((nextEnv.DANA_CLOUD_PRIMARY || "").trim().toLowerCase() === "true");
      })
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, [validateProvider]);

  // Shared write path for both Model Priority Manager fields below — same
  // POST /api/system/env the credential Save button uses, just for a
  // non-sensitive routing key instead of a masked secret.
  const saveSetting = useCallback((key: string, value: string): Promise<string> => {
    return apiFetch("/api/system/env", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, value }),
    })
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.detail || "save failed");
        setEnv(data.env ?? null);
        return String(data.detail || "Saved.");
      });
  }, []);

  const handleSaveProvider = useCallback(() => {
    if (!cloudProvider) return;
    setSavingProvider(true);
    setProviderResult(null);
    saveSetting("DANA_CLOUD_PROVIDER", cloudProvider)
      .then(setProviderResult)
      .catch((err) => setProviderResult(String(err instanceof Error ? err.message : err)))
      .finally(() => setSavingProvider(false));
  }, [cloudProvider, saveSetting]);

  const handleSaveModel = useCallback(() => {
    const value = localModel.trim();
    if (!value) return;
    setSavingModel(true);
    setModelResult(null);
    saveSetting("DANA_LOCAL_MODEL", value)
      .then(setModelResult)
      .catch((err) => setModelResult(String(err instanceof Error ? err.message : err)))
      .finally(() => setSavingModel(false));
  }, [localModel, saveSetting]);

  const handleToggleCloudPrimary = useCallback(
    (checked: boolean) => {
      setCloudPrimaryState(checked); // optimistic — reverted below on failure
      setSavingCloudPrimary(true);
      saveSetting("DANA_CLOUD_PRIMARY", checked ? "true" : "false")
        .catch(() => setCloudPrimaryState(!checked))
        .finally(() => setSavingCloudPrimary(false));
    },
    [saveSetting]
  );

  // "Clear All Sessions" — hard-to-reverse (deletes every saved chat with no
  // undo), so this confirms before sending the DELETE, same as any other
  // destructive action in this app.
  const handleClearAllSessions = useCallback(() => {
    if (!window.confirm("Delete ALL saved chat sessions? This cannot be undone.")) return;
    setClearingSessions(true);
    setClearResult(null);
    apiFetch("/api/sessions", { method: "DELETE" })
      .then((res) => res.json())
      .then((data) => {
        setClearResult(`Deleted ${data.deleted ?? 0} session(s).`);
        onSessionsCleared?.();
      })
      .catch((err) => setClearResult(String(err instanceof Error ? err.message : err)))
      .finally(() => setClearingSessions(false));
  }, [onSessionsCleared]);

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
        const result = { valid: Boolean(data.valid), detail: String(data.detail || "") };
        setSaveResult(result);
        // The save response already ran a live probe — reuse it for the
        // badge instead of an extra round trip to /validate right after.
        setValidation((prev) => ({ ...prev, [saveKey]: result }));
        setSaveValue("");
        setEnv(data.env ?? null);
      })
      .catch((err) => setSaveResult({ valid: false, detail: String(err instanceof Error ? err.message : err) }))
      .finally(() => setSaving(false));
  }, [saveKey, saveValue]);

  return (
    <div className="env-viewer">
      <div className="env-viewer__backdrop" onClick={onClose} />
      <div className="env-viewer__panel" role="dialog" aria-label="Settings">
        <div className="env-viewer__header">
          <h2>Settings</h2>
          <button type="button" className="env-viewer__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {!IS_GRADIO_MODE && (
          <div className="env-viewer__section">
            <div className="env-viewer__section-title">Model Priority Manager</div>
            <div className="env-viewer__save">
              <select
                className="env-viewer__save-select"
                value={cloudProvider}
                onChange={(e) => setCloudProvider(e.target.value)}
              >
                <option value="" disabled>
                  Cloud Provider…
                </option>
                {CLOUD_PROVIDERS.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="env-viewer__save-btn"
                onClick={handleSaveProvider}
                disabled={savingProvider || !cloudProvider}
              >
                {savingProvider ? "Saving…" : "Save Provider"}
              </button>
            </div>
            {providerResult && <div className="env-viewer__save-result">{providerResult}</div>}

            <div className="env-viewer__save">
              <input
                className="env-viewer__save-input"
                type="text"
                placeholder="Local model name (e.g. qwen2.5-coder:14b)"
                value={localModel}
                onChange={(e) => setLocalModel(e.target.value)}
              />
              <button
                type="button"
                className="env-viewer__save-btn"
                onClick={handleSaveModel}
                disabled={savingModel || !localModel.trim()}
              >
                {savingModel ? "Saving…" : "Save Model"}
              </button>
            </div>
            {modelResult && <div className="env-viewer__save-result">{modelResult}</div>}

            <div className="env-viewer__toggle-row">
              <ToggleSwitch
                checked={cloudPrimary}
                disabled={savingCloudPrimary}
                onChange={handleToggleCloudPrimary}
                label="Route to a Cloud Provider (turn off to use the Local Model above)"
              />
            </div>
          </div>
        )}

        <div className="env-viewer__badges">
          {PROVIDERS.map((p) => {
            const configured = Boolean(env?.[p.key]);
            const state = validation[p.key];
            // Desktop/REST mode: Green = live-validated, Red = the provider
            // rejected the key or the last check failed, amber "…" = a
            // check is in flight, neutral = configured but not yet checked
            // (or nothing saved). Gradio mode never has a real check to run
            // (see validateProvider) — "configured" is as specific as its
            // badge ever gets, same as before.
            const status = !configured
              ? "unconfigured"
              : IS_GRADIO_MODE
                ? "configured"
                : state?.checking
                  ? "checking"
                  : state
                    ? state.valid
                      ? "valid"
                      : "invalid"
                    : "configured";
            const title = IS_GRADIO_MODE
              ? configured
                ? "Saved in this browser"
                : "Not configured"
              : status === "checking"
                ? "Checking…"
                : status === "valid"
                  ? "Valid — provider accepted the key"
                  : status === "invalid"
                    ? state?.detail || "Invalid or rate-limited"
                    : configured
                      ? "Click to check validity"
                      : "Not configured";
            return (
              <button
                key={p.key}
                type="button"
                className={`env-viewer__badge env-viewer__badge--${status}`}
                onClick={() => configured && validateProvider(p.key)}
                disabled={!configured || IS_GRADIO_MODE}
                title={title}
              >
                <span className="env-viewer__badge-dot" />
                {p.label}
              </button>
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

        <div className="env-viewer__section env-viewer__section--danger">
          <div className="env-viewer__section-title">Danger Zone</div>

          <div className="env-viewer__toggle-row">
            <ToggleSwitch
              checked={autoApprove}
              disabled={IS_GRADIO_MODE}
              onChange={onToggleAutoApprove}
              label="Auto-approve tool actions (skip Human-in-the-Loop review)"
              tone="danger"
            />
          </div>

          <div className="env-viewer__danger-warning">
            <AlertTriangle size={15} strokeWidth={2.25} aria-hidden="true" />
            <span>
              {IS_GRADIO_MODE
                ? "Always on in Cloud mode — every mutating action there is already a simulated no-op."
                : autoApprove
                  ? "On: DANA will run terminal commands, execute scripts, and modify files immediately — with no approval prompt — for the rest of this chat session."
                  : "Off (default). While off, any terminal command, script, or file modification always pauses for your Approve/Modify/Cancel review first."}
            </span>
          </div>

          {!IS_GRADIO_MODE && (
            <div className="env-viewer__save">
              <button
                type="button"
                className="env-viewer__save-btn env-viewer__save-btn--danger"
                onClick={handleClearAllSessions}
                disabled={clearingSessions}
              >
                {clearingSessions ? "Clearing…" : "Clear All Sessions"}
              </button>
            </div>
          )}
          {clearResult && <div className="env-viewer__save-result">{clearResult}</div>}
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
