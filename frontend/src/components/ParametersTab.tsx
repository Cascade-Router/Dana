import { useEffect, useMemo, useState } from "react";
import type { ModelParameter, ParameterTable, RebuildStatus } from "../lib/useChatSocket";
import "./ParametersTab.css";

type Props = {
  table: ParameterTable;
  status: RebuildStatus;
  /** Absent where there is no live socket — the tab is then read-only. */
  onRebuild?: (overrides: Record<string, number>) => void;
};

// InspectorDock's "Parameters" tab: every editable dimension of the model,
// recovered from the session's recorded CAD calls. "Apply & Rebuild" re-runs
// those calls with the new values in FreeCAD directly — no LLM involved.

function validate(param: ModelParameter, raw: string): string | null {
  if (raw.trim() === "") return "Enter a value";
  const value = Number(raw);
  if (!Number.isFinite(value)) return "Not a number";
  if (param.unit === "count") {
    if (!Number.isInteger(value) || value < (param.min ?? 1)) return `Whole number ≥ ${param.min ?? 1}`;
  } else if (param.min !== null && value <= param.min) {
    return `Must be > ${param.min}`;
  }
  return null;
}

export function ParametersTab({ table, status, onRebuild }: Props) {
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  // A refreshed table (after a rebuild or any new geometry) is the new truth.
  useEffect(() => {
    setDrafts({});
  }, [table]);

  const errors = useMemo(() => {
    const out: Record<string, string> = {};
    for (const param of table.params) {
      const raw = drafts[param.id];
      if (raw === undefined) continue;
      const error = validate(param, raw);
      if (error) out[param.id] = error;
    }
    return out;
  }, [drafts, table.params]);

  const overrides = useMemo(() => {
    const out: Record<string, number> = {};
    for (const param of table.params) {
      const raw = drafts[param.id];
      if (raw === undefined || errors[param.id]) continue;
      const value = Number(raw);
      if (value !== param.value) out[param.id] = value;
    }
    return out;
  }, [drafts, errors, table.params]);

  const changed = Object.keys(overrides).length;
  const running = status.state === "running";
  const canRebuild = Boolean(onRebuild) && !table.reason && changed > 0 && !Object.keys(errors).length && !running;

  if (table.params.length === 0) {
    return (
      <div className="parameters-tab__empty">
        {table.reason ?? "No editable dimensions yet — they appear here as soon as Dana builds geometry."}
      </div>
    );
  }

  return (
    <form
      className="parameters-tab"
      onSubmit={(e) => {
        e.preventDefault();
        if (canRebuild) onRebuild?.(overrides);
      }}
    >
      {table.reason && <p className="parameters-tab__notice">{table.reason}</p>}
      <ul className="parameters-tab__list">
        {table.params.map((param) => {
          const inputId = `param-${param.id}`;
          const error = errors[param.id];
          const dirty = param.id in overrides;
          return (
            <li key={param.id} className="parameters-tab__row">
              <label htmlFor={inputId} className="parameters-tab__label" title={param.label}>
                {param.label}
                {param.warning && (
                  <span className="parameters-tab__warning" title={param.warning} aria-label={param.warning}>
                    ⚠
                  </span>
                )}
              </label>
              <div className="parameters-tab__field">
                <input
                  id={inputId}
                  className={`parameters-tab__input${dirty ? " parameters-tab__input--dirty" : ""}${error ? " parameters-tab__input--error" : ""}`}
                  type="number"
                  inputMode="decimal"
                  step={param.unit === "count" ? 1 : "any"}
                  value={drafts[param.id] ?? String(param.value)}
                  disabled={!onRebuild || Boolean(table.reason) || running}
                  aria-invalid={Boolean(error)}
                  aria-describedby={error ? `${inputId}-error` : undefined}
                  onChange={(e) => setDrafts((prev) => ({ ...prev, [param.id]: e.target.value }))}
                />
                <span className="parameters-tab__unit">{param.unit}</span>
              </div>
              {error && (
                <span id={`${inputId}-error`} className="parameters-tab__error">
                  {error}
                </span>
              )}
            </li>
          );
        })}
      </ul>
      <div className="parameters-tab__actions">
        <button type="submit" className="parameters-tab__apply" disabled={!canRebuild}>
          {running ? "Rebuilding…" : "Apply & Rebuild (0 tokens)"}
        </button>
        {changed > 0 && !running && (
          <button type="button" className="parameters-tab__reset" onClick={() => setDrafts({})}>
            Reset
          </button>
        )}
      </div>
      {status.state === "done" && (
        <p
          className={status.ok ? "parameters-tab__status" : "parameters-tab__status parameters-tab__status--error"}
          role="status"
        >
          {status.ok
            ? `Rebuilt${status.durationMs !== undefined ? ` in ${(status.durationMs / 1000).toFixed(1)} s` : ""}.`
            : status.error}
        </p>
      )}
    </form>
  );
}
