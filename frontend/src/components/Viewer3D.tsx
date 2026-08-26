import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import * as THREE from "three";
import type { CameraTarget, CanvasSelection } from "../lib/useChatSocket";
import { apiFetch, IS_GRADIO_MODE, resolveApiUrl } from "../lib/apiBase";
import { fetchGradioArtifacts } from "../lib/gradioChatClient";
import "./Viewer3D.css";

function isUrdfUrl(url: string): boolean {
  return /\.urdf(?:[?#].*)?$/i.test(url);
}

const _SPACE_URL = import.meta.env.VITE_HF_SPACE_URL as string;

// The live, fetchable URL for one previously-generated artifact — same
// {filename, url} shape in both transports, just sourced differently below.
type MeshArtifact = { filename: string; url: string };

// A URDF's <mesh filename="..."> can arrive as a bare filename, a
// "package://robot/meshes/wheel.stl" ROS path, or any other relative form
// dana/tools/urdf_builder.py (or a future non-Dana source) chooses to write
// — never a real, directly-fetchable URL: this project's meshes are served
// from HF Spaces/Vercel-hosted artifact storage, whose URLs are opaque and
// unrelated to whatever path string ended up in the XML. Rather than trust
// URDFLoader's own resolvePath (which just concatenates its `workingPath`
// onto the raw string — meaningless here, there's no real directory of
// sibling files at the URDF's own URL), this strips to the bare basename
// and looks it up directly against the CURRENT workspace artifacts list —
// the same list CadToolbar's Export dropdown already renders — matching by
// filename (Gradio's own `orig_name`, normalized into `.filename` below by
// fetchMeshArtifacts) to find that artifact's real, live `.url`.
function loadUrdfMesh(
  path: string,
  manager: THREE.LoadingManager,
  onLoad: (obj: THREE.Object3D | null, err?: Error) => void,
  artifacts: MeshArtifact[]
) {
  const filename = path.split(/[\\/]/).pop() || path;
  const artifact = artifacts.find((a) => a.filename === filename);
  if (!artifact) {
    console.warn(`[Viewer3D] no matching workspace artifact found for URDF mesh reference: ${filename}`);
    onLoad(null);
    return;
  }
  new STLLoader(manager).load(
    artifact.url,
    (geometry) => {
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({ color: "#4f8ff7", metalness: 0.15, roughness: 0.55 });
      onLoad(new THREE.Mesh(geometry, material));
    },
    undefined,
    () => onLoad(null, new Error(`failed to load URDF mesh: ${filename} (${artifact.url})`))
  );
}

// Mirrors CadToolbar's own refreshArtifacts — same two data sources (no
// shared cache between the two components; each fetches its own copy, the
// existing convention this codebase already follows for
// CadToolbar/WorkspacePlugin's Gradio artifact lists), normalized to one
// {filename, url} shape so loadUrdfMesh above never has to branch on
// transport. Gradio mode has no REST API at all (see apiBase.ts) — its
// artifacts already carry a real, live, cross-origin-fetchable `.url`
// (app.py FileData-ifies every registered path). REST mode's artifacts have
// no `.url` of their own, so one is built from the same
// /api/cad/artifacts/{filename}/download route CadToolbar's download()
// falls back to.
async function fetchMeshArtifacts(): Promise<MeshArtifact[]> {
  if (IS_GRADIO_MODE) {
    try {
      const files = await fetchGradioArtifacts(_SPACE_URL);
      return files.map((f) => ({ filename: f.filename, url: f.url }));
    } catch (err) {
      console.warn("[Viewer3D] fetchGradioArtifacts failed while resolving URDF meshes:", err);
      return [];
    }
  }
  try {
    const res = await apiFetch("/api/cad/artifacts");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const list: { filename: string }[] = Array.isArray(data.artifacts) ? data.artifacts : [];
    return list.map((a) => ({
      filename: a.filename,
      url: resolveApiUrl(`/api/cad/artifacts/${encodeURIComponent(a.filename)}/download`),
    }));
  } catch (err) {
    console.warn("[Viewer3D] fetching /api/cad/artifacts failed while resolving URDF meshes:", err);
    return [];
  }
}

// URDFLoader has no shared r3f `useLoader` cache to clear (unlike StlMesh
// below) — robots are instantiated directly in a plain useEffect (see
// Viewer3D's urdfRobot state), so this traversal is the ONLY teardown path
// for their geometry/material GPU buffers. Needed because <primitive>
// swaps (a new meshUrl replacing an old URDF while the Canvas itself stays
// mounted) don't go through react-three-fiber's own unmount-disposal walk
// the same way a whole-Canvas teardown does — this is exactly the WebGL
// context leak class this project fixed once before (see the lifecycle
// notes on the Viewer3D component itself).
function disposeUrdfRobot(robot: THREE.Object3D) {
  robot.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (material) (Array.isArray(material) ? material : [material]).forEach((m) => m.dispose());
  });
}

type JointSliderDef = { name: string; type: "revolute" | "continuous"; lower: number; upper: number };

// Only revolute/continuous joints are actuatable (fixed/prismatic/planar/
// floating are either rigid or have no single-scalar slider representation
// worth building yet) — the same movable-joint set dana/tools/urdf_builder.py
// tallies as movable_joint_count.
function jointSlidersFor(robot: URDFRobot): JointSliderDef[] {
  return Object.entries(robot.joints)
    .filter(([, joint]) => joint.jointType === "revolute" || joint.jointType === "continuous")
    .map(([name, joint]) => {
      const hasRealLimit =
        joint.jointType === "revolute" &&
        Number.isFinite(joint.limit?.lower) &&
        Number.isFinite(joint.limit?.upper) &&
        joint.limit.upper > joint.limit.lower;
      return {
        name,
        type: joint.jointType as "revolute" | "continuous",
        // "continuous" has no URDF-mandated limit (it rotates freely) — a
        // slider still needs finite bounds, so it gets the same +/-pi
        // default a revolute joint with a malformed/missing <limit> falls
        // back to.
        lower: hasRealLimit ? joint.limit.lower : -Math.PI,
        upper: hasRealLimit ? joint.limit.upper : Math.PI,
      };
    });
}

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
  const isUrdf = !!meshUrl && isUrdfUrl(meshUrl);

  const [urdfRobot, setUrdfRobot] = useState<URDFRobot | null>(null);
  const [jointDefs, setJointDefs] = useState<JointSliderDef[]>([]);
  const [jointValues, setJointValues] = useState<Record<string, number>>({});

  // Parses a fresh URDF whenever meshUrl points at one, and always tears
  // down whatever robot this effect previously built — on a new URL, on
  // unmount, and even if meshUrl flips to null or to a plain .stl mid-load
  // (the `cancelled` flag stops a late load() callback from installing a
  // robot for a URL that's no longer current).
  useEffect(() => {
    if (!isUrdf || !meshUrl) {
      setUrdfRobot((current) => {
        if (current) disposeUrdfRobot(current);
        return null;
      });
      setJointDefs([]);
      setJointValues({});
      return;
    }

    let cancelled = false;
    // Resolved BEFORE loader.load() starts (not raced against it) — every
    // mesh reference inside the URDF needs this list already in hand the
    // moment URDFLoader's parser reaches it, since loadUrdfMesh's lookup is
    // synchronous from its own caller's perspective (it has no way to tell
    // URDFLoader "wait, let me go fetch something first").
    fetchMeshArtifacts().then((artifacts) => {
      if (cancelled) return;
      const loader = new URDFLoader();
      // urdf-loader's own MeshLoadDoneFunc type demands a non-null
      // Object3D; loadUrdfMesh's `null` (no matching artifact/load failure)
      // becomes a harmless empty placeholder here instead — same as what
      // reaches it on any other load error, and URDFLoader's own parser
      // already guards with `else if (obj)` before adding it to the scene.
      loader.loadMeshCb = (meshPath, manager, onComplete) =>
        loadUrdfMesh(meshPath, manager, (obj, err) => onComplete(obj ?? new THREE.Object3D(), err), artifacts);
      loader.load(
        meshUrl,
        (robot) => {
          if (cancelled) {
            disposeUrdfRobot(robot);
            return;
          }
          const sliders = jointSlidersFor(robot);
          setUrdfRobot((current) => {
            if (current) disposeUrdfRobot(current);
            return robot;
          });
          setJointDefs(sliders);
          setJointValues(Object.fromEntries(sliders.map((s) => [s.name, 0])));
        },
        undefined,
        (err) => console.error("[Viewer3D] failed to load URDF assembly:", err)
      );
    });

    return () => {
      cancelled = true;
      setUrdfRobot((current) => {
        if (current) disposeUrdfRobot(current);
        return null;
      });
      setJointDefs([]);
      setJointValues({});
    };
  }, [meshUrl, isUrdf]);

  const handleJointChange = useCallback(
    (name: string, value: number) => {
      urdfRobot?.setJointValue(name, value);
      setJointValues((prev) => ({ ...prev, [name]: value }));
    },
    [urdfRobot]
  );

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
          {meshUrl && !isUrdf && !contextLost && <StlMesh key={meshUrl} url={meshUrl} onSelect={onSelect} />}
        </Suspense>
        {urdfRobot && !contextLost && (
          <primitive key={urdfRobot.uuid} object={urdfRobot} rotation={[-Math.PI / 2, 0, 0]} />
        )}
      </Canvas>
      {contextLost && <div className="viewer3d__placeholder">Recovering 3D view…</div>}
      {!contextLost && !meshUrl && (
        <div className="viewer3d__placeholder">No geometry yet — ask Dana to build something.</div>
      )}
      {!contextLost && jointDefs.length > 0 && (
        <div className="viewer3d__joint-panel">
          <div className="viewer3d__joint-panel-title">Joints</div>
          {jointDefs.map((joint) => (
            <label key={joint.name} className="viewer3d__joint-row">
              <span className="viewer3d__joint-name" title={joint.name}>
                {joint.name}
              </span>
              <input
                type="range"
                min={joint.lower}
                max={joint.upper}
                step={(joint.upper - joint.lower) / 200 || 0.01}
                value={jointValues[joint.name] ?? 0}
                onChange={(event) => handleJointChange(joint.name, parseFloat(event.target.value))}
              />
              <span className="viewer3d__joint-value">{(jointValues[joint.name] ?? 0).toFixed(2)}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
