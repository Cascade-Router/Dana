import { useEffect, useMemo, useState } from "react";
import { BlueprintViewer } from "../components/BlueprintViewer";
import { CadToolbar } from "../components/CadToolbar";
import { TopologyTab } from "../components/DAGMonitor";
import { InspectorDock, type InspectorTab } from "../components/InspectorDock";
import { MeshHistoryPicker } from "../components/MeshHistoryPicker";
import { PlanTab } from "../components/PlanTab";
import { TerminalTab } from "../components/TerminalTab";
import { Viewer3D } from "../components/Viewer3D";
import { useCadArtifacts } from "../lib/useCadArtifacts";
import type { PluginComponentProps } from "./types";
import "./CadPlugin.css";

type ViewportMode = "3d" | "2d";

// Default export so this can be React.lazy()-imported — the R3F canvas,
// three.js and @xyflow/react bundles only load once the CAD plugin is
// actually activated, not on Dana's core chat-only startup path.
export default function CadPlugin({
  meshUrl,
  cameraTarget,
  onSelect,
  log,
  topologyGraph,
  plan,
  sessionId,
}: PluginComponentProps) {
  // Active Plan / Topology / Terminal used to be three independent floating
  // cards fighting for the same space over the viewport — now they're tabs
  // in one docked InspectorDock (see that component). Badges reuse the
  // exact same counts the old floating cards showed in their own headers.
  const tabs: InspectorTab[] = useMemo(
    () => [
      {
        id: "plan",
        label: "Active Plan",
        glyph: "▤",
        badge:
          plan.tasks.length > 0
            ? `${plan.tasks.filter((t) => t.status === "completed").length}/${plan.tasks.length}`
            : undefined,
        content: <PlanTab plan={plan} />,
      },
      {
        id: "topology",
        label: "Topology",
        glyph: "◈",
        badge: Object.keys(topologyGraph.nodes).length || undefined,
        content: <TopologyTab graph={topologyGraph} />,
      },
      {
        id: "terminal",
        label: "Terminal",
        glyph: "▥",
        badge: log.length || undefined,
        content: <TerminalTab log={log} />,
      },
    ],
    [plan, topologyGraph, log]
  );

  // Lifted here (not fetched separately inside CadToolbar/MeshHistoryPicker)
  // so both consumers share one session-scoped list/poll instead of two
  // redundant fetches of the same data.
  const { artifacts, refresh: refreshArtifacts } = useCadArtifacts(sessionId);

  // Mesh History: the live chat feed only ever carries the SINGLE
  // most-recently-touched object's mesh (dana.api.server scopes
  // export_mesh_stl to target_object=result_name, by design), so several
  // independently-generated objects in one session would otherwise only
  // ever show the newest one. `pinnedMeshUrl` overrides that when set;
  // reset to null (follow live again) the instant a NEW live mesh arrives,
  // so picking an old artifact never silently hides a freshly requested
  // change — the user has to explicitly go back to browsing history after
  // that, rather than a stale pin silently surviving new work.
  const [pinnedMeshUrl, setPinnedMeshUrl] = useState<string | null>(null);
  useEffect(() => {
    setPinnedMeshUrl(null);
  }, [meshUrl]);
  const displayedMeshUrl = pinnedMeshUrl ?? meshUrl;

  // 3D Assembly / 2D Blueprint toggle. Viewer3D itself is NEVER conditionally
  // mounted based on this (see the comment on it below) — only which one is
  // visible (via a modifier class in CadPlugin.css) and whether the
  // 3D-only MeshHistoryPicker overlay renders at all change.
  const [viewportMode, setViewportMode] = useState<ViewportMode>("3d");

  return (
    <div className="cad-plugin">
      <CadToolbar
        meshUrl={meshUrl}
        sessionId={sessionId}
        artifacts={artifacts}
        onRefreshArtifacts={refreshArtifacts}
      />
      <div className={`cad-plugin__viewport cad-plugin__viewport--${viewportMode}`}>
        <div className="cad-plugin__mode-toggle">
          <button
            type="button"
            className={viewportMode === "3d" ? "cad-plugin__mode-btn cad-plugin__mode-btn--active" : "cad-plugin__mode-btn"}
            onClick={() => setViewportMode("3d")}
          >
            3D Assembly
          </button>
          <button
            type="button"
            className={viewportMode === "2d" ? "cad-plugin__mode-btn cad-plugin__mode-btn--active" : "cad-plugin__mode-btn"}
            onClick={() => setViewportMode("2d")}
          >
            2D Blueprint
          </button>
        </div>
        {/* Viewer3D — and the <Canvas>/WebGLRenderer inside it — is always
            rendered here, never gated behind meshUrl, an artifact list's
            length, OR the 2D/3D toggle above. A conditional mount would tear
            down and recreate the renderer every time (see Viewer3D's own
            lifecycle notes) — CadPlugin.css hides it with plain CSS
            (cad-plugin__viewport--2d > .viewer3d) instead, so the WebGL
            context survives switching to the 2D tab and back. */}
        <Viewer3D meshUrl={displayedMeshUrl} cameraTarget={cameraTarget} onSelect={onSelect} />
        <BlueprintViewer artifacts={artifacts} sessionId={sessionId} />
        {viewportMode === "3d" && (
          <MeshHistoryPicker
            artifacts={artifacts}
            sessionId={sessionId}
            liveUrl={meshUrl}
            activeUrl={displayedMeshUrl}
            onSelectArtifact={setPinnedMeshUrl}
            onFollowLive={() => setPinnedMeshUrl(null)}
          />
        )}
        <InspectorDock tabs={tabs} defaultTabId="topology" />
      </div>
    </div>
  );
}
