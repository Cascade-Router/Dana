import { useMemo } from "react";
import { resolveArtifactUrl, type CadArtifact } from "../lib/useCadArtifacts";
import "./MeshHistoryPicker.css";

// Only formats Viewer3D can actually render survive into this list.
// StlMesh (Viewer3D.tsx) always parses through three.js's STLLoader
// regardless of extension — it has no GLTFLoader/OBJLoader branch at all —
// so a "step"/"glb"/"obj" artifact (export_freecad_model's STEP sibling,
// generate_3d_from_image's mesh) would silently fail to parse if picked
// here. ".urdf" is the one other extension Viewer3D actually branches on
// (isUrdfUrl -> URDFLoader). Fixing StlMesh to dispatch on format is a
// separate, larger change than this history picker; excluding the
// unrenderable formats here keeps every entry in this list guaranteed to
// actually show something when clicked.
const _VIEWABLE_FORMATS = new Set(["stl", "urdf"]);

type Props = {
  artifacts: CadArtifact[];
  sessionId: string | null;
  /** The chat feed's own current mesh — always available as "● Live",
   * regardless of whether it's also already in `artifacts` (a turn's very
   * latest mesh briefly isn't, until the next artifact-list refresh). */
  liveUrl: string | null;
  /** Whichever URL Viewer3D is actually showing right now — `liveUrl` when
   * nothing is pinned, or the pinned artifact's URL otherwise. */
  activeUrl: string | null;
  onSelectArtifact: (url: string) => void;
  onFollowLive: () => void;
};

// Floating history list over the CAD viewport (see Viewer3D.css's own
// joint-panel for the same absolute-over-.viewer3d convention) — lets a
// session with several independently-generated objects (not merged via
// perform_freecad_boolean) go back and view an earlier one, since the live
// feed only ever carries the SINGLE most-recently-touched object's mesh
// (dana.api.server scopes export_mesh_stl to target_object=result_name, by
// design — see that call site's own comment). Renders nothing for 0 or 1
// viewable entries: with at most the live mesh to show, there is nothing to
// pick between.
export function MeshHistoryPicker({ artifacts, sessionId, liveUrl, activeUrl, onSelectArtifact, onFollowLive }: Props) {
  const viewable = useMemo(() => artifacts.filter((a) => _VIEWABLE_FORMATS.has(a.format.toLowerCase())), [artifacts]);

  if (viewable.length === 0) return null;

  return (
    <div className="mesh-history-picker">
      <button
        type="button"
        className={`mesh-history-picker__item ${!!liveUrl && activeUrl === liveUrl ? "mesh-history-picker__item--active" : ""}`}
        onClick={onFollowLive}
        disabled={!liveUrl}
        title="Follow the live/most-recent mesh"
      >
        ● Live
      </button>
      {viewable.map((a, index) => {
        const url = resolveArtifactUrl(a, sessionId);
        return (
          <button
            key={`${a.filename}-${index}`}
            type="button"
            className={`mesh-history-picker__item ${activeUrl === url ? "mesh-history-picker__item--active" : ""}`}
            onClick={() => onSelectArtifact(url)}
            title={a.filename}
          >
            <span className="mesh-history-picker__format">{a.format.toUpperCase()}</span>
            <span className="mesh-history-picker__name">{a.filename}</span>
          </button>
        );
      })}
    </div>
  );
}
