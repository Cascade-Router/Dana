import { useCallback, useEffect, useState } from "react";
import { resolveApiUrl } from "../lib/apiBase";
import type { PluginComponentProps } from "./types";
import "./ServicesPlugin.css";

type Service = { alias: string; pid: number; running: boolean };

// Simple interval polling for the selected service's log (the "Refresh
// Logs" requirement) — only while a service is actually selected (see the
// effect below), so this never fires for a plugin the user isn't even
// looking at.
const LOG_POLL_INTERVAL_MS = 3000;

// Background Process Management's user-facing counterpart
// (dana/api/services.py -> dana.plugins.os.background_services) — direct
// visibility into (and control over) whatever the agent has started via
// its own start_background_service ReAct tool, without needing to ask the
// LLM to check on or kill one. Ignores the CAD-shaped PluginComponentProps
// every plugin currently receives (same pattern as WorkspacePlugin/
// SkillsPlugin) — this plugin manages its own state via REST calls
// instead. Killing a service here goes through the EXACT SAME
// DELETE /api/services/{alias} -> stop_background_service the agent's own
// tool call would, so a service killed from this UI behaves identically
// to one the agent stopped itself — same cross-platform process-tree kill,
// no separate logic here.
export default function ServicesPlugin(_props: PluginComponentProps) {
  const [services, setServices] = useState<Service[] | null>(null);
  const [servicesError, setServicesError] = useState<string | null>(null);
  const [killingAlias, setKillingAlias] = useState<string | null>(null);

  const [selectedAlias, setSelectedAlias] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[] | null>(null);
  const [logExists, setLogExists] = useState(true);
  const [logError, setLogError] = useState<string | null>(null);

  const loadServices = useCallback(() => {
    setServicesError(null);
    fetch(resolveApiUrl("/api/services"))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setServices(data.services ?? []))
      .catch((err) => setServicesError(String(err instanceof Error ? err.message : err)));
  }, []);

  useEffect(() => {
    loadServices();
  }, [loadServices]);

  const loadLogs = useCallback((alias: string) => {
    setLogError(null);
    fetch(resolveApiUrl(`/api/services/${encodeURIComponent(alias)}/logs`))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setLogLines(Array.isArray(data.lines) ? data.lines : []);
        setLogExists(Boolean(data.exists));
      })
      .catch((err) => setLogError(String(err instanceof Error ? err.message : err)));
  }, []);

  const selectService = useCallback(
    (alias: string) => {
      setSelectedAlias(alias);
      setLogLines(null);
      setLogExists(true);
      setLogError(null);
      loadLogs(alias);
    },
    [loadLogs]
  );

  useEffect(() => {
    if (!selectedAlias) return;
    const id = window.setInterval(() => loadLogs(selectedAlias), LOG_POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [selectedAlias, loadLogs]);

  const killService = useCallback(
    (alias: string) => {
      setKillingAlias(alias);
      fetch(resolveApiUrl(`/api/services/${encodeURIComponent(alias)}`), { method: "DELETE" })
        .then(async (res) => {
          if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then(() => {
          loadServices();
          if (selectedAlias === alias) loadLogs(alias);
        })
        .catch((err) => setServicesError(String(err instanceof Error ? err.message : err)))
        .finally(() => setKillingAlias(null));
    },
    [loadServices, loadLogs, selectedAlias]
  );

  return (
    <div className="services-plugin">
      <div className="services-plugin__sidebar">
        <div className="services-plugin__sidebar-header">
          <span>Background Services</span>
          <button type="button" className="services-plugin__icon-btn" onClick={loadServices} title="Refresh">
            ↻
          </button>
        </div>

        <div className="services-plugin__list">
          {servicesError && <div className="services-plugin__error">{servicesError}</div>}
          {!servicesError && !services && <div className="services-plugin__placeholder">Loading…</div>}
          {services && services.length === 0 && (
            <div className="services-plugin__placeholder">No background services running.</div>
          )}
          {services?.map((s) => (
            <div
              key={s.alias}
              className={`services-plugin__row ${
                selectedAlias === s.alias ? "services-plugin__row--selected" : ""
              }`}
            >
              <button type="button" className="services-plugin__row-main" onClick={() => selectService(s.alias)}>
                <span
                  className={`services-plugin__status services-plugin__status--${
                    s.running ? "running" : "stopped"
                  }`}
                  aria-hidden="true"
                />
                <span className="services-plugin__alias">{s.alias}</span>
                <span className="services-plugin__pid">pid {s.pid}</span>
              </button>
              <button
                type="button"
                className="services-plugin__kill"
                disabled={!s.running || killingAlias === s.alias}
                onClick={() => killService(s.alias)}
                title={s.running ? "Kill this service" : "Already stopped"}
              >
                {killingAlias === s.alias ? "Killing…" : "Kill"}
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="services-plugin__viewer">
        {!selectedAlias && (
          <div className="services-plugin__placeholder services-plugin__placeholder--centered">
            Select a service to view its log.
          </div>
        )}
        {selectedAlias && (
          <>
            <div className="services-plugin__viewer-header">
              <span className="services-plugin__viewer-title">{selectedAlias}</span>
              <button
                type="button"
                className="services-plugin__refresh-logs"
                onClick={() => loadLogs(selectedAlias)}
              >
                ↻ Refresh Logs
              </button>
            </div>
            {logError && <div className="services-plugin__error">{logError}</div>}
            {!logError && logLines === null && <div className="services-plugin__placeholder">Loading…</div>}
            {!logError && logLines !== null && !logExists && (
              <div className="services-plugin__placeholder">No log output yet.</div>
            )}
            {!logError && logLines !== null && logExists && (
              <pre className="services-plugin__log">{logLines.length ? logLines.join("\n") : "(empty log)"}</pre>
            )}
          </>
        )}
      </div>
    </div>
  );
}
