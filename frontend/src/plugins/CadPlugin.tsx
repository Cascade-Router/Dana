import { DAGMonitor } from "../components/DAGMonitor";
import { Viewer3D } from "../components/Viewer3D";
import type { PluginComponentProps } from "./types";
import "./CadPlugin.css";

// Default export so this can be React.lazy()-imported — the R3F canvas,
// three.js and @xyflow/react bundles only load once the CAD plugin is
// actually activated, not on Dana's core chat-only startup path.
export default function CadPlugin({ meshUrl, cameraTarget, onSelect, log }: PluginComponentProps) {
  return (
    <div className="cad-plugin">
      <Viewer3D meshUrl={meshUrl} cameraTarget={cameraTarget} onSelect={onSelect} />
      <DAGMonitor log={log} />
    </div>
  );
}
