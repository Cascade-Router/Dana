import { useCallback, useState } from "react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import { resolveArtifactUrl, type CadArtifact } from "../lib/useCadArtifacts";
import "./CadToolbar.css";

type Artifact = CadArtifact;

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
  /** The active chat's session_id — Scoped Mini-Explorer: every REST call
   * below carries this so the artifact list/download/open-desktop only
   * ever touches THIS session's own freecad_output/sessions/<session_id>/
   * files, never another chat's (dana.api.cad enforces the same scoping
   * server-side; this is what actually tells it which session). `null`
   * before the first WS "ready" event of a brand-new chat — every REST
   * call below is skipped/no-ops until it's set, same as a real session_id
   * would eventually arrive; not a permanent no-session state. */
  sessionId: string | null;
  /** Lifted to CadPlugin (useCadArtifacts) rather than fetched here directly
   * — MeshHistoryPicker needs the exact same session-scoped list, and two
   * independent hook instances would mean two redundant polls/fetches of
   * the same data every time this tab re-renders. */
  artifacts: CadArtifact[];
  onRefreshArtifacts: () => void;
};

export function CadToolbar({ meshUrl, sessionId, artifacts, onRefreshArtifacts }: Props) {
  const [exportOpen, setExportOpen] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [exportingMesh, setExportingMesh] = useState(false);
  const [printingFile, setPrintingFile] = useState<string | null>(null);

  // The live viewport mesh's actual format — export_mesh_stl now writes
  // .glb by default (see that function's docstring), but an older session
  // could still be showing a plain .stl artifact, so this reads the real
  // extension off meshUrl rather than assuming either one.
  const meshFormatLabel =
    meshUrl?.split(/[\\/]/).pop()?.split("?")[0]?.split(".").pop()?.toUpperCase() || "MESH";

  const launchDesktop = useCallback(() => {
    if (!sessionId) return;
    setLaunching(true);
    setLaunchError(null);
    apiFetch(`/api/cad/open-desktop?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" })
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.detail || "failed to open FreeCAD");
      })
      .catch((err) => setLaunchError(String(err instanceof Error ? err.message : err)))
      .finally(() => setLaunching(false));
  }, [sessionId]);

  const download = useCallback(
    (artifact: Artifact) => {
      // window.open (not an <a download>) — dev mode serves the frontend and
      // API from different origins (see apiBase.ts), and a cross-origin
      // "download" attribute is silently ignored by most browsers; the
      // backend's Content-Disposition: attachment header is what actually
      // triggers the save either way. In Gradio mode there's no REST download
      // route at all — `artifact.url` is already the real, fetchable Gradio
      // file URL (see refreshArtifacts/fetchGradioArtifacts), and no
      // session_id to append either (single-session deployment).
      window.open(resolveArtifactUrl(artifact, sessionId), "_blank");
      setExportOpen(false);
    },
    [sessionId]
  );

  // Hands a generated .step/.stp file to the OS's own default 3D
  // slicer/viewer association (dana.api.cad's print-step endpoint reuses
  // the same "let the OS pick the app" pattern open-desktop already uses
  // for FreeCAD, just without hardcoding which app that is). Desktop-only,
  // same reason launchDesktop is gated on !IS_GRADIO_MODE below — there's
  // no local OS to hand a file off to in the hosted Gradio/HF Space mode.
  const printStep = useCallback(
    (artifact: Artifact) => {
      if (!sessionId) return;
      setPrintingFile(artifact.filename);
      setLaunchError(null);
      apiFetch(
        `/api/cad/artifacts/${encodeURIComponent(artifact.filename)}/print?session_id=${encodeURIComponent(sessionId)}`,
        { method: "POST" }
      )
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
          if (!ok) throw new Error(data.detail || "failed to open in the default app");
        })
        .catch((err) => setLaunchError(String(err instanceof Error ? err.message : err)))
        .finally(() => setPrintingFile(null));
    },
    [sessionId]
  );

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
      // Fallback only fires when meshUrl carries no extension at all (rare
      // — every real mesh_url does); matches the live preview's actual
      // current default format (export_mesh_stl writes .glb, not .stl —
      // see that function's own docstring) rather than a stale STL guess.
      const filename = nameFromUrl && nameFromUrl.includes(".") ? nameFromUrl : "export.glb";
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
            onRefreshArtifacts();
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
              <span className="cad-toolbar__export-format">{meshFormatLabel}</span>
              <span className="cad-toolbar__export-name">
                {exportingMesh ? "Exporting…" : "Current mesh (viewport)"}
              </span>
            </button>
            {artifacts.length === 0 && (
              <div className="cad-toolbar__export-empty">No artifacts generated yet.</div>
            )}
            {artifacts.map((a, index) => (
              <div key={`${a.filename}-${index}`} className="cad-toolbar__export-row">
                <button type="button" className="cad-toolbar__export-item" onClick={() => download(a)}>
                  <span className="cad-toolbar__export-format">{a.format.toUpperCase()}</span>
                  <span className="cad-toolbar__export-name">{a.filename}</span>
                </button>
                {!IS_GRADIO_MODE && (a.format === "step" || a.format === "stp") && (
                  <button
                    type="button"
                    className="cad-toolbar__print-btn"
                    title="Open in the OS's default 3D slicer/viewer"
                    onClick={() => printStep(a)}
                    disabled={printingFile === a.filename}
                  >
                    {printingFile === a.filename ? "…" : "🖨"}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {launchError && <div className="cad-toolbar__error">{launchError}</div>}
    </div>
  );
}
