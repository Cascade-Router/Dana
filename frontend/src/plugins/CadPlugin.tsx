import { CadToolbar } from "../components/CadToolbar";
import { DAGMonitor } from "../components/DAGMonitor";
import { Viewer3D } from "../components/Viewer3D";
import type { PluginComponentProps } from "./types";
import "./CadPlugin.css";

// Default export so this can be React.lazy()-imported — the R3F canvas,
// three.js and @xyflow/react bundles only load once the CAD plugin is
// actually activated, not on Dana's core chat-only startup path.
export default function CadPlugin({ meshUrl, cameraTarget, onSelect, log, sessionId }: PluginComponentProps) {
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
        <DAGMonitor log={log} />
      </div>
    </div>
  );
}
