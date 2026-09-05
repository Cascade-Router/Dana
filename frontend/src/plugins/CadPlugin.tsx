import { useEffect, useMemo, useState } from "react";
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

  return (
    <div className="cad-plugin">
      <CadToolbar
        meshUrl={meshUrl}
        sessionId={sessionId}
        artifacts={artifacts}
        onRefreshArtifacts={refreshArtifacts}
      />
      <div className="cad-plugin__viewport">
        {/* Viewer3D — and the <Canvas>/WebGLRenderer inside it — is always
            rendered here, never gated behind meshUrl or an artifact list's
            length. A conditional mount would tear down and recreate the
            renderer on every intermediate ReAct step where the mesh
            payload is transiently null/[], exhausting the browser's WebGL
            context budget (see Viewer3D's own lifecycle notes). */}
        <Viewer3D meshUrl={displayedMeshUrl} cameraTarget={cameraTarget} onSelect={onSelect} />
        <MeshHistoryPicker
          artifacts={artifacts}
          sessionId={sessionId}
          liveUrl={meshUrl}
          activeUrl={displayedMeshUrl}
          onSelectArtifact={setPinnedMeshUrl}
          onFollowLive={() => setPinnedMeshUrl(null)}
        />
        <InspectorDock tabs={tabs} defaultTabId="topology" />
      </div>
    </div>
  );
}
