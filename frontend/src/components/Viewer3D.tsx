import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import * as THREE from "three";
import type { CameraTarget, CanvasSelection } from "../lib/useChatSocket";
import "./Viewer3D.css";

function StlMesh({ url, onSelect }: { url: string; onSelect: (selection: CanvasSelection) => void }) {
  // Explicit rather than relying on Three.js's own default (which already
  // is 'anonymous' as of the installed three version — see Loader.js).
  // Note this specific property has no effect on STLLoader in particular:
  // it loads through FileLoader, which fetches via `fetch()` using its own
  // `withCredentials` flag (default 'same-origin' credentials), not the
  // `crossOrigin` property at all — that property only matters for
  // <img>-backed loaders (TextureLoader/ImageLoader). Whether a cross-
  // origin Gradio mesh URL actually loads is governed entirely by the
  // server's CORS response headers (see gradioChatClient.ts's own note on
  // Gradio's CustomCORSMiddleware), not by anything set here. Kept anyway,
  // set explicitly rather than left to Three.js's default, so the intent
  // isn't silently dependent on an unannounced upstream default.
  const geometry = useLoader(STLLoader, url, (loader) => {
    loader.setCrossOrigin("anonymous");
  });
  geometry.computeVertexNormals();
  geometry.center();

  // useLoader's own cache (keyed by [Loader, url]) lives OUTSIDE this
  // component's lifetime, so it isn't touched by react-three-fiber's usual
  // unmount disposal. That's harmless for reuse (every generated mesh here
  // gets a brand-new, never-repeated opaque URL — see dana.api.server's
  // _MESH_REGISTRY / app.py's mesh_url — so a cache hit never happens
  // anyway), but left alone the Map entry itself accumulates forever across
  // a long session, each one pinning a reference to a geometry object
  // r3f's reconciler has *already disposed the GPU buffers of* (`geometry`
  // is attached to <mesh> the same way an `attach="geometry"` JSX child
  // would be, so it's included in react-three-fiber's own recursive
  // dispose-on-unmount walk — no separate `geometry.dispose()` call is
  // needed here, and adding one would double-dispose against r3f's own).
  // This just drops the now-stale cache entry when a new/no mesh replaces
  // this one, or the viewer unmounts (tab switch away from CAD).
  useEffect(() => {
    return () => {
      useLoader.clear(STLLoader, url);
    };
  }, [url]);

  const meshRef = useRef<THREE.Mesh>(null);
  const [markerPosition, setMarkerPosition] = useState<THREE.Vector3 | null>(null);

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    const mesh = meshRef.current;
    const face = event.face;
    if (!mesh || !face) return;

    const position = geometry.attributes.position;
    const a = new THREE.Vector3().fromBufferAttribute(position, face.a);
    const b = new THREE.Vector3().fromBufferAttribute(position, face.b);
    const c = new THREE.Vector3().fromBufferAttribute(position, face.c);
    const localCentroid = a.add(b).add(c).divideScalar(3);
    const worldCentroid = mesh.localToWorld(localCentroid.clone());
    const worldNormal = face.normal.clone().transformDirection(mesh.matrixWorld).normalize();

    setMarkerPosition(worldCentroid);
    onSelect({
      meshId: "current_mesh",
      centroid: [worldCentroid.x, worldCentroid.y, worldCentroid.z],
      normal: [worldNormal.x, worldNormal.y, worldNormal.z],
    });
  };

  return (
    <>
      <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} onClick={handleClick}>
        <meshStandardMaterial color="#4f8ff7" metalness={0.15} roughness={0.55} />
      </mesh>
      {markerPosition && (
        <mesh position={markerPosition}>
          <sphereGeometry args={[1.6, 20, 20]} />
          <meshBasicMaterial color="#ffb020" />
        </mesh>
      )}
    </>
  );
}

function CameraRig({ cameraTarget }: { cameraTarget: CameraTarget | null }) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls);
  const goalRef = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>(null);

  useEffect(() => {
    if (!cameraTarget) return;
    goalRef.current = {
      position: new THREE.Vector3(...cameraTarget.position),
      target: new THREE.Vector3(...cameraTarget.target),
    };
  }, [cameraTarget]);

  useFrame(() => {
    const goal = goalRef.current;
    if (!goal) return;
    camera.position.lerp(goal.position, 0.08);
    const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
    if (orbit) {
      orbit.target.lerp(goal.target, 0.08);
      orbit.update();
    }
    if (camera.position.distanceTo(goal.position) < 0.05) {
      goalRef.current = null;
    }
  });

  return null;
}

type Props = {
  meshUrl: string | null;
  cameraTarget: CameraTarget | null;
  onSelect: (selection: CanvasSelection) => void;
};

// WebGL lifecycle notes (App.tsx unmounts this whole component on every
// CAD <-> Chat/Workspace/Coder tab switch — <activePlugin.Component> is
// swapped by React, not hidden):
//
// - No duplicate WebGLRenderer: <Canvas> creates its renderer once in a
//   mount-only effect (react-three-fiber internals), never on a re-render,
//   so switching tabs and back always yields exactly one live renderer.
// - Renderer/geometry/material disposal on unmount: react-three-fiber's
//   own unmountComponentAtNode (invoked automatically when <Canvas>
//   itself unmounts) already calls `gl.forceContextLoss()`,
//   `gl.renderLists.dispose()`, and recursively disposes every
//   attached geometry/material in the scene graph — verified directly
//   against the installed @react-three/fiber source
//   (events-*.cjs.dev.js's unmountComponentAtNode/removeChild). Adding
//   our own renderer.dispose()/geometry.dispose() calls on top of that
//   would double-dispose objects r3f already tore down.
// - Resize: <Canvas> sizes itself via react-use-measure, which is a thin
//   ResizeObserver wrapper — the scene is never re-initialized on resize,
//   only the renderer/camera dimensions are updated.
//
// The one thing r3f does NOT do for us is handle the browser's own
// `webglcontextlost`/`webglcontextrestored` events (a backgrounded tab or
// GPU pressure can silently drop the context outside of any React
// lifecycle) — that's handled explicitly below.
export function Viewer3D({ meshUrl, cameraTarget, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [contextLost, setContextLost] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // preventDefault() is what tells the browser "try to restore this
    // context" — without it, a lost context is treated as permanent and
    // webglcontextrestored never fires, leaving a blank/frozen canvas the
    // next time this tab becomes active. Three.js's own WebGLRenderer
    // re-uploads geometry/material GPU buffers lazily on the next draw
    // call once restored, since the underlying JS-side objects (and this
    // component's own scene graph) were never torn down — no manual scene
    // rebuild needed here.
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      console.warn("[Viewer3D] WebGL context lost — waiting for the browser to restore it.");
      setContextLost(true);
    };
    const handleContextRestored = () => {
      console.log("[Viewer3D] WebGL context restored.");
      setContextLost(false);
    };

    canvas.addEventListener("webglcontextlost", handleContextLost, false);
    canvas.addEventListener("webglcontextrestored", handleContextRestored, false);
    return () => {
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      canvas.removeEventListener("webglcontextrestored", handleContextRestored);
    };
  }, []);

  return (
    <div className="viewer3d">
      <Canvas ref={canvasRef} camera={{ position: [80, 80, 80], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={1.1} />
        <Grid args={[400, 400]} cellColor="#333" sectionColor="#555" fadeDistance={400} />
        <OrbitControls makeDefault />
        <CameraRig cameraTarget={cameraTarget} />
        <Suspense fallback={null}>
          {meshUrl && !contextLost && <StlMesh key={meshUrl} url={meshUrl} onSelect={onSelect} />}
        </Suspense>
      </Canvas>
      {contextLost && <div className="viewer3d__placeholder">Recovering 3D view…</div>}
      {!contextLost && !meshUrl && (
        <div className="viewer3d__placeholder">No geometry yet — ask Dana to build something.</div>
      )}
    </div>
  );
}
