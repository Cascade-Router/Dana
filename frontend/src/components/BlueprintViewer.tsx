import { useEffect, useMemo, useState } from "react";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";
import { resolveArtifactUrl, type CadArtifact } from "../lib/useCadArtifacts";
import "./BlueprintViewer.css";

type Props = {
  artifacts: CadArtifact[];
  sessionId: string | null;
};

// 2D orthographic/isometric blueprint viewer — the .svg sibling
// dana.plugins.freecad.techdraw_export.generate_2d_blueprint now produces
// alongside its .pdf. Rendered via a plain <img>, not an inline fetch+parse:
// browsers already render SVG files natively given the correct
// image/svg+xml Content-Type (dana/api/cad.py's artifact endpoint sets this
// — see _MEDIA_TYPES), so no client-side DXF/SVG parsing library is needed
// here at all, just pan/zoom chrome around a normal image.
//
// Shows the MOST RECENT .svg artifact only — useCadArtifacts already
// returns newest-first (matching Viewer3D's own "latest mesh" behavior).
// No history picker of its own yet, unlike Viewer3D's MeshHistoryPicker;
// add one later if browsing older blueprints turns out to matter.
export function BlueprintViewer({ artifacts, sessionId }: Props) {
  const svgArtifact = useMemo(() => artifacts.find((a) => a.format === "svg"), [artifacts]);
  const url = svgArtifact ? resolveArtifactUrl(svgArtifact, sessionId) : null;

  // The artifact being LISTED doesn't guarantee its download still works —
  // dana/api/cad.py's _resolve_artifact can legitimately 404 (the registry
  // still names a file whose temp copy was already cleaned up), and a plain
  // <img> with no onError handler just shows the browser's own broken-image
  // icon with zero explanation on that failure. Reset whenever the URL
  // changes (a new/different artifact deserves a fresh load attempt, same
  // as Viewer3D's own meshError reset on a new meshUrl), not left sticky
  // across artifacts.
  const [loadFailed, setLoadFailed] = useState(false);
  useEffect(() => {
    setLoadFailed(false);
  }, [url]);

  if (!svgArtifact || !url) {
    return (
      <div className="blueprint-viewer__placeholder">
        No 2D blueprint yet — ask Dana to generate one.
      </div>
    );
  }

  if (loadFailed) {
    return (
      <div className="blueprint-viewer__placeholder">
        Blueprint unavailable — the file may have been removed or failed to load.
      </div>
    );
  }

  return (
    <div className="blueprint-viewer">
      {/* key={url}: a newly generated blueprint should open fresh, centered
          and at 1x — not silently reuse whatever pan/zoom transform the
          PREVIOUS drawing's geometry happened to leave the view at. */}
      <TransformWrapper key={url} initialScale={1} minScale={0.2} maxScale={8} centerOnInit doubleClick={{ mode: "toggle" }}>
        <TransformComponent
          wrapperStyle={{ width: "100%", height: "100%" }}
          contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}
        >
          <img
            src={url}
            alt="2D CAD blueprint"
            className="blueprint-viewer__image"
            onError={() => setLoadFailed(true)}
          />
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
}
