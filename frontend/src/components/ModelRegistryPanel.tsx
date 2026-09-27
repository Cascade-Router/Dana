import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw, ChevronDown, ChevronRight, GripVertical, ShieldAlert } from "lucide-react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import "./ModelRegistryPanel.css";

type Pricing = { prompt_per_1m: number; completion_per_1m: number };
type RateLimits = { tpm: number | null; rpm: number | null; rpd: number | null };

type ModelSpec = {
  id: string;
  name: string;
  context_window: number;
  max_output_tokens: number;
  pricing: Pricing;
  rate_limits: RateLimits;
  modalities: string[];
  supports_tool_calling: boolean;
  supports_thought_signature: boolean;
  is_deprecated: boolean;
  is_active: boolean;
  status_reason: string;
  has_runtime_error: boolean;
};

type ProviderDomain = {
  provider: string;
  requires_api_key: boolean;
  api_key_env: string | null;
  is_available: boolean;
  status_reason: string;
  models: ModelSpec[];
};

type Matrix = {
  providers: ProviderDomain[];
  preferences: { order: string[]; disabled: string[] };
};

type Props = { onClose: () => void };

function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 === 0 ? 0 : 1)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}k`;
  return String(tokens);
}

function formatPrice(usdPer1m: number): string {
  return usdPer1m === 0 ? "free" : `$${usdPer1m.toFixed(usdPer1m < 1 ? 3 : 2)}`;
}

function formatLimit(n: number | null): string {
  if (n === null) return "—";
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

function badgeClass(model: ModelSpec): string {
  if (model.has_runtime_error) return "model-registry__badge model-registry__badge--critical";
  if (model.is_deprecated) return "model-registry__badge model-registry__badge--red";
  if (model.is_active) return "model-registry__badge model-registry__badge--green";
  return "model-registry__badge model-registry__badge--amber";
}

function badgeLabel(model: ModelSpec): string {
  if (model.has_runtime_error) return "CRITICAL";
  if (model.is_deprecated) return "Deprecated";
  if (model.is_active) return "Active";
  return "Requires key";
}

export function ModelRegistryPanel({ onClose }: Props) {
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [order, setOrder] = useState<string[]>([]);
  const [disabled, setDisabled] = useState<Set<string>>(new Set());
  const [dragId, setDragId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveNote, setSaveNote] = useState<string | null>(null);
  const [clearingIds, setClearingIds] = useState<Set<string>>(new Set());

  const applyMatrix = useCallback((data: Matrix) => {
    setMatrix(data);
    setDisabled(new Set(data.preferences.disabled));
    const activeIds = data.providers.flatMap((p) => p.models.filter((m) => m.is_active).map((m) => m.id));
    const savedOrder = data.preferences.order.filter((id) => activeIds.includes(id));
    const missing = activeIds.filter((id) => !savedOrder.includes(id));
    setOrder([...savedOrder, ...missing]);
  }, []);

  const load = useCallback(() => {
    setSyncing(true);
    setError(null);
    return apiFetch("/api/models/matrix")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => applyMatrix(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSyncing(false));
  }, [applyMatrix]);

  useEffect(() => {
    if (IS_GRADIO_MODE) {
      setError("Not available in the hosted web demo.");
      return;
    }
    load();
  }, [load]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const modelsById = useMemo(() => {
    const map = new Map<string, ModelSpec>();
    for (const p of matrix?.providers ?? []) for (const m of p.models) map.set(m.id, m);
    return map;
  }, [matrix]);

  const toggleCollapsed = (provider: string) =>
    setCollapsed((prev) => ({ ...prev, [provider]: !prev[provider] }));

  const toggleDisabled = (modelId: string) =>
    setDisabled((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });

  const reorder = (draggedId: string, targetId: string) => {
    if (draggedId === targetId) return;
    setOrder((prev) => {
      const next = prev.filter((id) => id !== draggedId);
      const targetIdx = next.indexOf(targetId);
      next.splice(targetIdx, 0, draggedId);
      return next;
    });
  };

  const savePreferences = () => {
    setSaving(true);
    setSaveNote(null);
    apiFetch("/api/models/preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order, disabled: Array.from(disabled) }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        applyMatrix(data);
        setSaveNote("Saved.");
      })
      .catch((err) => setSaveNote(`Failed to save: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => setSaving(false));
  };

  const clearRuntimeError = (modelId: string) => {
    setClearingIds((prev) => new Set(prev).add(modelId));
    // modelId is interpolated raw (not encodeURIComponent'd) — model ids like
    // "openai/gpt-oss-120b" contain "/" on purpose, matching the backend's
    // {model_id:path} route, so percent-encoding the slash would break it.
    apiFetch(`/api/models/${modelId}/clear-error`, { method: "POST" })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => applyMatrix(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() =>
        setClearingIds((prev) => {
          const next = new Set(prev);
          next.delete(modelId);
          return next;
        }),
      );
  };

  return (
    <div className="model-registry">
      <div className="model-registry__backdrop" onClick={onClose} />
      <div className="model-registry__panel" role="dialog" aria-label="Model Registry & Preferences">
        <div className="model-registry__header">
          <h2>Model Registry &amp; Preferences</h2>
          <div className="model-registry__header-actions">
            <button
              type="button"
              className="model-registry__sync-btn"
              onClick={load}
              disabled={syncing}
              title="Re-check API keys and query Ollama for downloaded models"
            >
              <RefreshCw size={14} strokeWidth={2.25} className={syncing ? "model-registry__spin" : ""} />
              Sync with Ollama &amp; Keys
            </button>
            <button type="button" className="model-registry__close" onClick={onClose} aria-label="Close">
              ×
            </button>
          </div>
        </div>

        <div className="model-registry__body">
          {error && <div className="model-registry__error">Failed to load: {error}</div>}
          {!matrix && !error && <div className="model-registry__empty">Loading…</div>}

          {matrix && (
            <>
              <section className="model-registry__section">
                <h3>Fallback priority order</h3>
                <p className="model-registry__hint">
                  Drag active models to reorder the fallback chain. Uncheck a model to disable it.
                </p>
                <ul className="model-registry__order-list">
                  {order.map((id, idx) => {
                    const model = modelsById.get(id);
                    if (!model) return null;
                    return (
                      <li
                        key={id}
                        className="model-registry__order-item"
                        draggable
                        onDragStart={() => setDragId(id)}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault();
                          if (dragId) reorder(dragId, id);
                          setDragId(null);
                        }}
                      >
                        <GripVertical size={14} className="model-registry__grip" aria-hidden="true" />
                        <span className="model-registry__order-rank">{idx + 1}</span>
                        <span className="model-registry__order-name">{model.name}</span>
                        <label className="model-registry__order-toggle">
                          <input
                            type="checkbox"
                            checked={!disabled.has(id)}
                            onChange={() => toggleDisabled(id)}
                          />
                          enabled
                        </label>
                      </li>
                    );
                  })}
                  {order.length === 0 && (
                    <li className="model-registry__order-empty">No active models yet — configure an API key or pull an Ollama model below.</li>
                  )}
                </ul>
                <div className="model-registry__save-row">
                  <button type="button" className="model-registry__save-btn" onClick={savePreferences} disabled={saving}>
                    {saving ? "Saving…" : "Save priority"}
                  </button>
                  {saveNote && <span className="model-registry__save-note">{saveNote}</span>}
                </div>
              </section>

              <section className="model-registry__section">
                <h3>Providers</h3>
                {matrix.providers.map((domain) => {
                  const isCollapsed = collapsed[domain.provider] ?? false;
                  return (
                    <div key={domain.provider} className="model-registry__card">
                      <button
                        type="button"
                        className="model-registry__card-header"
                        onClick={() => toggleCollapsed(domain.provider)}
                      >
                        {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                        <span className="model-registry__card-title">{domain.provider}</span>
                        <span
                          className={
                            "model-registry__badge " +
                            (domain.is_available ? "model-registry__badge--green" : "model-registry__badge--amber")
                          }
                        >
                          {domain.is_available ? "Ready" : domain.status_reason}
                        </span>
                        <span className="model-registry__card-count">{domain.models.length} models</span>
                      </button>
                      {!isCollapsed && (
                        <div className="model-registry__table-wrap">
                          <table className="model-registry__table">
                            <thead>
                              <tr>
                                <th>Model</th>
                                <th>Status</th>
                                <th>Context</th>
                                <th>Price / 1M (in / out)</th>
                                <th>TPM</th>
                                <th>Tools</th>
                              </tr>
                            </thead>
                            <tbody>
                              {domain.models.map((model) => (
                                <tr key={model.id} className={model.has_runtime_error ? "model-registry__row--critical" : undefined}>
                                  <td>
                                    <div className="model-registry__model-name">{model.name}</div>
                                    <div className="model-registry__model-id">{model.id}</div>
                                    {model.has_runtime_error && (
                                      <div className="model-registry__error-detail">
                                        <ShieldAlert size={12} aria-hidden="true" />
                                        {model.status_reason}
                                      </div>
                                    )}
                                  </td>
                                  <td>
                                    <div className="model-registry__status-cell">
                                      <span className={badgeClass(model)} title={model.status_reason}>
                                        {badgeLabel(model)}
                                      </span>
                                      {model.has_runtime_error && (
                                        <button
                                          type="button"
                                          className="model-registry__reset-btn"
                                          onClick={() => clearRuntimeError(model.id)}
                                          disabled={clearingIds.has(model.id)}
                                        >
                                          {clearingIds.has(model.id) ? "Resetting…" : "Acknowledge & Reset"}
                                        </button>
                                      )}
                                    </div>
                                  </td>
                                  <td>{formatContext(model.context_window)}</td>
                                  <td>
                                    {formatPrice(model.pricing.prompt_per_1m)} / {formatPrice(model.pricing.completion_per_1m)}
                                  </td>
                                  <td>{formatLimit(model.rate_limits.tpm)}</td>
                                  <td>{model.supports_tool_calling ? "✓" : "—"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  );
                })}
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
