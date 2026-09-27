import { useEffect, useState } from "react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import "./ConfigViewer.css";

// Mirrors dana/api/server.py's GET /api/config response shape — a
// deliberately narrow, read-only view of dana.config's two tuning knobs
// (never API keys/DSNs, see that route's own docstring).
type ConfigSnapshot = {
  max_react_iterations: number;
  llm_max_output_tokens: number;
};

type Row = { label: string; value: string; hint: string };

function rowsFromConfig(config: ConfigSnapshot): Row[] {
  return [
    {
      label: "MAX_STEPS",
      value: String(config.max_react_iterations),
      hint: "Max reasoning/tool-call iterations per turn before Dana stops itself (dana.config.MAX_REACT_ITERATIONS).",
    },
    {
      label: "LLM_MAX_OUTPUT_TOKENS",
      value: String(config.llm_max_output_tokens),
      hint: "Max completion tokens per model call — raised from 1024 to prevent a verbose plan/response being cut off mid-generation (dana.config.LLM_MAX_OUTPUT_TOKENS).",
    },
  ];
}

type Props = {
  onClose: () => void;
};

// Small read-only viewer for dana/config.py's runtime constants — opened
// from App.tsx's "⚙️ Config" header button. Same overlay/panel convention
// as EnvViewerWidget (backdrop click / Escape / a header ✕ button all
// close it), kept as its own component rather than folded into
// EnvViewerWidget since that one is specifically about secrets/API keys —
// a different concern from "what tuning constants is this backend
// actually running with."
export function ConfigViewer({ onClose }: Props) {
  const [config, setConfig] = useState<ConfigSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // No /api/config on the pure-Gradio HF backend (app.py never mounts
    // the FastAPI app at all — see apiBase.ts's IS_GRADIO_MODE) — same
    // "nothing to show" convention EnvViewerWidget/App.tsx's health poll
    // already use there, rather than surfacing a confusing fetch error.
    if (IS_GRADIO_MODE) {
      setError("Not available in the hosted web demo.");
      return;
    }
    let cancelled = false;
    apiFetch("/api/config")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        if (!cancelled) setConfig(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const rows = config ? rowsFromConfig(config) : [];

  return (
    <div className="config-viewer">
      <div className="config-viewer__backdrop" onClick={onClose} />
      <div className="config-viewer__panel" role="dialog" aria-label="Active Configuration">
        <div className="config-viewer__header">
          <h2>⚙️ Active Configuration</h2>
          <button type="button" className="config-viewer__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="config-viewer__body">
          {error && <div className="config-viewer__error">Failed to load: {error}</div>}
          {!config && !error && <div className="config-viewer__empty">Loading…</div>}
          {rows.map((row) => (
            <div key={row.label} className="config-viewer__row" title={row.hint}>
              <span className="config-viewer__label">{row.label}</span>
              <span className="config-viewer__value">{row.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
