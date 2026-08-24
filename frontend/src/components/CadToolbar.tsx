import { useCallback, useEffect, useState } from "react";
import { apiFetch, resolveApiUrl } from "../lib/apiBase";
import "./CadToolbar.css";

type Artifact = {
  filename: string;
  format: string;
  size_bytes: number;
  modified_at: number;
  source: "generated" | "exported";
};

// CAD tab toolbar: launches the real FreeCAD desktop GUI on the most
// recently generated document (dana.api.cad's open-desktop endpoint reuses
// dana.plugins.freecad.engine.show_in_freecad_gui — the SAME never-steal-
// focus/never-duplicate logic already used automatically after every
// create_freecad_*/perform_freecad_boolean tool call), and a dropdown to
// download the latest generated .FCStd/.stl or exported .step/.stl.
export function CadToolbar() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [exportOpen, setExportOpen] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const refreshArtifacts = useCallback(() => {
    apiFetch("/api/cad/artifacts")
      .then((res) => (res.ok ? res.json() : { artifacts: [] }))
      .then((data) => setArtifacts(data.artifacts ?? []))
      .catch(() => setArtifacts([]));
  }, []);

  useEffect(() => {
    refreshArtifacts();
  }, [refreshArtifacts]);

  const launchDesktop = useCallback(() => {
    setLaunching(true);
    setLaunchError(null);
    apiFetch("/api/cad/open-desktop", { method: "POST" })
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.detail || "failed to open FreeCAD");
      })
      .catch((err) => setLaunchError(String(err instanceof Error ? err.message : err)))
      .finally(() => setLaunching(false));
  }, []);

  const download = useCallback((filename: string) => {
    // window.open (not an <a download>) — dev mode serves the frontend and
    // API from different origins (see apiBase.ts), and a cross-origin
    // "download" attribute is silently ignored by most browsers; the
    // backend's Content-Disposition: attachment header is what actually
    // triggers the save either way.
    window.open(resolveApiUrl(`/api/cad/artifacts/${encodeURIComponent(filename)}/download`), "_blank");
    setExportOpen(false);
  }, []);

  return (
    <div className="cad-toolbar">
      <button type="button" className="cad-toolbar__btn" onClick={launchDesktop} disabled={launching}>
        {launching ? "Launching…" : "🖥 Launch FreeCAD GUI"}
      </button>

      <div className="cad-toolbar__export">
        <button
          type="button"
          className="cad-toolbar__btn"
          onClick={() => {
            refreshArtifacts();
            setExportOpen((v) => !v);
          }}
        >
          ⬇ Export ▾
        </button>
        {exportOpen && (
          <div className="cad-toolbar__export-menu">
            {artifacts.length === 0 && (
              <div className="cad-toolbar__export-empty">No artifacts generated yet.</div>
            )}
            {artifacts.map((a) => (
              <button
                key={a.filename}
                type="button"
                className="cad-toolbar__export-item"
                onClick={() => download(a.filename)}
              >
                <span className="cad-toolbar__export-format">{a.format.toUpperCase()}</span>
                <span className="cad-toolbar__export-name">{a.filename}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {launchError && <div className="cad-toolbar__error">{launchError}</div>}
    </div>
  );
}
