import { useCallback, useEffect, useState } from "react";
import { apiFetch, IS_GRADIO_MODE, resolveApiUrl } from "./apiBase";
import { fetchGradioArtifacts } from "./gradioChatClient";

export type CadArtifact = {
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

/** The real, fetchable URL for one artifact in either transport — same
 * formula CadToolbar's own download() already used inline, extracted here
 * so a second consumer (MeshHistoryPicker) doesn't have to re-derive it. */
export function resolveArtifactUrl(artifact: CadArtifact, sessionId: string | null): string {
  return (
    artifact.url ??
    resolveApiUrl(
      `/api/cad/artifacts/${encodeURIComponent(artifact.filename)}/download` +
        (sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "")
    )
  );
}

// Shared between CadToolbar (Export dropdown) and MeshHistoryPicker (viewport
// history list) — both need the exact same session-scoped, transport-aware
// artifact list, so this is the one place that fetches it rather than two
// independent copies drifting apart over time. dana.api.artifacts_registry.
// list_artifacts (both the REST endpoint and app.py's Gradio "artifacts" api
// call it under the hood) already returns newest-first, so callers needing
// "most recent first" get that for free with no client-side re-sort.
export function useCadArtifacts(sessionId: string | null): {
  artifacts: CadArtifact[];
  refresh: () => void;
} {
  const [artifacts, setArtifacts] = useState<CadArtifact[]>([]);

  const refresh = useCallback(() => {
    if (IS_GRADIO_MODE) {
      // No REST endpoint reachable at all in this build (see apiBase.ts) —
      // list via app.py's hidden Gradio "artifacts" api instead. Gradio mode
      // is a single-session HF Space deployment (no session_id concept
      // there at all), so no scoping is needed or possible here.
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
          console.error("[useCadArtifacts] fetchGradioArtifacts failed:", err);
          setArtifacts([]);
        });
      return;
    }
    if (!sessionId) {
      setArtifacts([]);
      return;
    }
    apiFetch(`/api/cad/artifacts?session_id=${encodeURIComponent(sessionId)}`)
      .then((res) => (res.ok ? res.json() : { artifacts: [] }))
      .then((data) => setArtifacts(data.artifacts ?? []))
      .catch(() => setArtifacts([]));
  }, [sessionId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { artifacts, refresh };
}
