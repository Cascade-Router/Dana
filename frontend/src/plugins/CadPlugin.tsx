import { useMemo } from "react";
import { CadToolbar } from "../components/CadToolbar";
import { TopologyTab } from "../components/DAGMonitor";
import { InspectorDock, type InspectorTab } from "../components/InspectorDock";
import { PlanTab } from "../components/PlanTab";
import { TerminalTab } from "../components/TerminalTab";
import { Viewer3D } from "../components/Viewer3D";
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

  return (
    <div className="cad-plugin">
      <CadToolbar meshUrl={meshUrl} sessionId={sessionId} />
      <div className="cad-plugin__viewport">
        {/* Viewer3D — and the <Canvas>/WebGLRenderer inside it — is always
            rendered here, never gated behind meshUrl or an artifact list's
            length. A conditional mount would tear down and recreate the
            renderer on every intermediate ReAct step where the mesh
            payload is transiently null/[], exhausting the browser's WebGL
            context budget (see Viewer3D's own lifecycle notes). */}
        <Viewer3D meshUrl={meshUrl} cameraTarget={cameraTarget} onSelect={onSelect} />
        <InspectorDock tabs={tabs} defaultTabId="topology" />
      </div>
    </div>
  );
}
