import { useCallback, useEffect, useState } from "react";
import { apiFetch, IS_GRADIO_MODE, resolveApiUrl } from "../lib/apiBase";
import { fetchGradioArtifacts } from "../lib/gradioChatClient";
import "./CadToolbar.css";

type Artifact = {
  filename: string;
  format: string;
  size_bytes: number;
  modified_at: number;
  source: "generated" | "exported";
  /** Only set in Gradio mode — a real fetchable URL to the file itself
   * (Gradio's own FileData resolution), since there's no REST download
   * endpoint (dana/api/cad.py) to build one from there. */
  url?: string;
};

const _SPACE_URL = import.meta.env.VITE_HF_SPACE_URL as string;

// CAD tab toolbar: launches the real FreeCAD desktop GUI on the most
// recently generated document (dana.api.cad's open-desktop endpoint reuses
// dana.plugins.freecad.engine.show_in_freecad_gui — the SAME never-steal-
// focus/never-duplicate logic already used automatically after every
// create_freecad_*/perform_freecad_boolean tool call), and a dropdown to
// download the latest generated .FCStd/.stl or exported .step/.stl.
type Props = {
  /** The mesh currently shown in Viewer3D — a plain, already-resolved URL in
   * both modes (resolveMeshUrl() for WS/Desktop, the raw Gradio FileData URL
   * for Gradio — see useChatSocket.ts/useGradioChat.ts), so one fetch+blob
   * download path covers both without a mode branch. */
  meshUrl: string | null;
};

export function CadToolbar({ meshUrl }: Props) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [exportOpen, setExportOpen] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [exportingMesh, setExportingMesh] = useState(false);

  const refreshArtifacts = useCallback(() => {
    if (IS_GRADIO_MODE) {
      // No REST endpoint reachable at all in this build (see apiBase.ts) —
      // list via app.py's hidden Gradio "artifacts" api instead.
      fetchGradioArtifacts(_SPACE_URL)
        .then((files) =>
          setArtifacts(
            files.map((f) => ({
              filename: f.filename,
              format: f.format,
              size_bytes: f.size_bytes,
              modified_at: 0,
              source: "generated" as const,
              url: f.url,
            }))
          )
        )
        .catch((err) => {
          console.error("[CadToolbar] fetchGradioArtifacts failed:", err);
          setArtifacts([]);
        });
      return;
    }
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

  const download = useCallback((artifact: Artifact) => {
    // window.open (not an <a download>) — dev mode serves the frontend and
    // API from different origins (see apiBase.ts), and a cross-origin
    // "download" attribute is silently ignored by most browsers; the
    // backend's Content-Disposition: attachment header is what actually
    // triggers the save either way. In Gradio mode there's no REST download
    // route at all — `artifact.url` is already the real, fetchable Gradio
    // file URL (see refreshArtifacts/fetchGradioArtifacts).
    const url = artifact.url ?? resolveApiUrl(`/api/cad/artifacts/${encodeURIComponent(artifact.filename)}/download`);
    window.open(url, "_blank");
    setExportOpen(false);
  }, []);

  // The live viewport mesh (Viewer3D's meshUrl) isn't in the artifacts list
  // above — it's the in-progress geometry, not yet a saved file on either
  // backend. Fetching it as a blob (rather than window.open, which the
  // artifact download() above uses) works identically cross-origin in dev
  // and same-origin in prod, and lets us force a filename since neither
  // mode's mesh URL carries orig_name (see gradioChatClient.ts's
  // GradioFileData — that's only populated for the artifacts endpoint).
  const exportMesh = useCallback(async () => {
    if (!meshUrl) return;
    setExportingMesh(true);
    setLaunchError(null);
    try {
      const res = await fetch(meshUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status} fetching mesh`);
      const blob = await res.blob();
      const nameFromUrl = meshUrl.split(/[\\/]/).pop()?.split("?")[0];
      const filename = nameFromUrl && nameFromUrl.includes(".") ? nameFromUrl : "export.stl";
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setLaunchError(String(err instanceof Error ? err.message : err));
    } finally {
      setExportingMesh(false);
      setExportOpen(false);
    }
  }, [meshUrl]);

  return (
    <div className="cad-toolbar">
      {!IS_GRADIO_MODE && (
        <button type="button" className="cad-toolbar__btn" onClick={launchDesktop} disabled={launching}>
          {launching ? "Launching…" : "🖥 Launch FreeCAD GUI"}
        </button>
      )}

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
            <button
              type="button"
              className="cad-toolbar__export-item"
              onClick={exportMesh}
              disabled={!meshUrl || exportingMesh}
            >
              <span className="cad-toolbar__export-format">STL</span>
              <span className="cad-toolbar__export-name">
                {exportingMesh ? "Exporting…" : "Current mesh (viewport)"}
              </span>
            </button>
            {artifacts.length === 0 && (
              <div className="cad-toolbar__export-empty">No artifacts generated yet.</div>
            )}
            {artifacts.map((a) => (
              <button
                key={a.filename}
                type="button"
                className="cad-toolbar__export-item"
                onClick={() => download(a)}
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
