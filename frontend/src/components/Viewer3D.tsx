import { Component, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import * as THREE from "three";
import type { CameraTarget, CanvasSelection } from "../lib/useChatSocket";
import { apiFetch, IS_GRADIO_MODE, resolveApiUrl } from "../lib/apiBase";
import { fetchGradioArtifacts } from "../lib/gradioChatClient";
import "./Viewer3D.css";

// Extension-Not-At-The-End fix: both patterns used to anchor the extension
// to the very END of the string (`(?:[?#].*)?$`) — correct for the live
// `/api/mesh/{token}.glb` push, but WRONG for MeshHistoryPicker's artifact
// download URL, `/api/cad/artifacts/{filename}/download?session_id=...`
// (dana/api/cad.py) — there the extension sits in the MIDDLE of the path,
// as part of `{filename}`, followed by `/download` and a query string, not
// at the end. Every artifact selected from history therefore matched
// NEITHER isUrdfUrl NOR isGlbUrl, fell through to the StlMesh branch below,
// and had its real GLB bytes fed to STLLoader — reading FreeCAD's own
// binary GLB header as if it were a binary STL triangle count produces
// exactly the "Invalid typed array length: <garbage>" crash this was
// chasing. Now matches the extension followed by `/`, `?`, `#`, OR end of
// string, so it's recognized whether it's the last thing in the URL or
// just the last thing in ITS OWN path segment.
function isUrdfUrl(url: string): boolean {
  return /\.urdf(?:[/?#]|$)/i.test(url);
}

// GLTF Binary — the live-preview bandwidth format switch (dana.plugins.
// freecad.engine.export_mesh_stl / dana.platform.mock now write .glb, not
// .stl, for every CAD tool call's viewport preview — see that function's
// own docstring). Checked separately from isUrdfUrl above so an OLDER
// .stl mesh_url (an already-generated session/history artifact, or
// export_model's own unrelated "download as STL" file) still renders via
// StlMesh below exactly as before — nothing about that path changed.
function isGlbUrl(url: string): boolean {
  return /\.(?:glb|gltf)(?:[/?#]|$)/i.test(url);
}

// Default Shading Pass: an STL (and any GLB converted FROM one — see
// dana.plugins.freecad.engine's STL->GLB trimesh fallback) has NO shared
// vertex indices — every triangle owns 3 independent vertex copies. Calling
// computeVertexNormals() straight off the loader can therefore only ever
// average a vertex's normal with the other 2 corners of its OWN triangle,
// which is indistinguishable from flat per-face shading no matter how good
// the lighting/material already is (this viewport already uses a real
// MeshStandardMaterial + ambient/directional lights — the flatness isn't a
// lighting problem). mergeVertices() welds coincident positions into shared
// indices first, so an edge two adjacent real-world triangles actually
// share is shared in the buffer too, letting computeVertexNormals() blend
// across it into genuine smooth per-vertex normals — the actual fix for
// curved CAD surfaces (fillets, cylinders, swept profiles) looking faceted
// and hard to read as 3D. Unrelated to FreeCAD's own DisplayMode/draw-style
// GUI properties, which this viewport never reads at all — geometry reaches
// here as a flat mesh file, not a live FreeCAD document.
//
// Disposes the pre-merge geometry: mergeVertices returns a NEW
// BufferGeometry rather than mutating in place, and the original is never
// attached to a <mesh> (so react-three-fiber's own unmount-disposal walk
// never sees it) — same "no cache hit ever happens, unique URL per
// generated mesh" reasoning StlMesh's own useLoader.clear() cleanup below
// already relies on, just applied one step earlier here.
function smoothShaded(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  // mergeVertices only treats two vertex COPIES as duplicates if ALL of
  // their attributes match within tolerance, not position alone. STLLoader
  // already bakes in a flat per-triangle `normal` attribute straight from
  // the STL file's own facet normals — two positions that are genuinely
  // the same point in space, on the shared edge between two adjacent
  // facets, still carry two DIFFERENT normals (each triangle's own) at
  // that point, so mergeVertices refuses to merge them. Confirmed live: a
  // coarse 8-sided test prism reduced from 96 to only 50 vertices (the
  // flat top/bottom caps merged fine, since every triangle on one cap
  // already shares the SAME normal) while the curved/faceted side walls —
  // exactly what needed smoothing — silently stayed fully unmerged, and
  // the render was pixel-identical to no fix at all. Deleting the stale
  // normal attribute first makes the merge decision position-only, so a
  // genuinely shared edge is actually recognized as shared; the fresh
  // computeVertexNormals() call below then has real topology to average
  // across instead of recomputing the same flat values back.
  geometry.deleteAttribute("normal");
  const merged = mergeVertices(geometry);
  merged.computeVertexNormals();
  geometry.dispose();
  return merged;
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
      const material = new THREE.MeshStandardMaterial({ color: "#4f8ff7", metalness: 0.15, roughness: 0.55 });
      onLoad(new THREE.Mesh(smoothShaded(geometry), material));
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
  const rawGeometry = useLoader(STLLoader, url, (loader) => {
    loader.setCrossOrigin("anonymous");
  });
  // Memoized on rawGeometry (useLoader's own stable, cached-by-url result)
  // — smoothShaded disposes its input, so recomputing it on every re-render
  // (e.g. a click updating markerPosition below) would try to re-merge an
  // already-disposed geometry instead of doing real work once per mesh.
  const geometry = useMemo(() => {
    const shaded = smoothShaded(rawGeometry);
    shaded.center();
    return shaded;
  }, [rawGeometry]);

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

// One entry per named mesh node inside a loaded GLB's scene graph — a
// FreeCAD assembly export (multiple boxes/cylinders/booleans left as
// separate top-level objects, e.g. via export_model rather than a single
// fused solid) round-trips as one glTF node per source object, each
// keeping that object's own FreeCAD Name. `object` is a direct, live
// reference into gltf.scene (NOT a copy) — toggling `.visible` on it is
// exactly how three.js/r3f already decides whether to draw a node, so no
// extra render-time branching is needed in GlbMesh itself.
type GlbPart = { id: string; label: string; object: THREE.Object3D; visible: boolean };

// GLB counterpart to StlMesh above — same click-to-select/marker contract,
// different loader since a GLTFLoader result is a full scene graph (nodes,
// meshes, materials already attached) rather than STLLoader's bare
// BufferGeometry, so this renders it via <primitive>, three.js/R3F's own
// standard pattern for an already-built object graph (matches how
// Viewer3D's own urdfRobot is rendered below), instead of StlMesh's
// single manually-built <mesh>.
//
// Rotation: deliberately NONE here, unlike StlMesh's -90° X rotation.
// STL carries no coordinate-system convention (FreeCAD's native Z-up needs
// that correction for Three.js's Y-up), but glTF's own spec MANDATES Y-up
// — a conformant exporter (FreeCAD's importGLTF, or Mesh.export's own
// glTF path — see dana.plugins.freecad.engine's _EXPORT_MESH_PREVIEW_SCRIPT)
// should already emit correctly-oriented geometry, so reapplying STL's
// rotation here would double-rotate it. Unverified against a live FreeCAD
// glTF export in this environment — if a real model renders on its side,
// this is the line to revisit.
function GlbMesh({
  url,
  onSelect,
  onPartsChange,
}: {
  url: string;
  onSelect: (selection: CanvasSelection) => void;
  onPartsChange: (parts: GlbPart[]) => void;
}) {
  const gltf = useLoader(GLTFLoader, url);
  const [markerPosition, setMarkerPosition] = useState<THREE.Vector3 | null>(null);

  // Same cache-eviction reasoning as StlMesh's own useLoader.clear() below
  // — every generated mesh gets a brand-new, never-repeated opaque URL, so
  // a cache hit never happens anyway; this just drops the now-stale entry.
  useEffect(() => {
    return () => {
      useLoader.clear(GLTFLoader, url);
    };
  }, [url]);

  // A raw FreeCAD glTF export carries no material worth trusting for a
  // consistent in-app look — override every mesh in the scene to the same
  // blue StlMesh/loadUrdfMesh already use, so a GLB and an STL result look
  // identical in the viewport. Collects the same traversal's named mesh
  // nodes into the parts list the parent's visibility panel renders — one
  // pass, since both need to walk every mesh in the scene anyway. Cleared
  // on unmount/URL change so a stale part list never outlives the scene
  // it was toggling (the parent panel just disappears along with it, same
  // as jointDefs does for a URDF).
  useEffect(() => {
    const parts: GlbPart[] = [];
    const usedIds = new Set<string>();
    gltf.scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry = smoothShaded(mesh.geometry);
      mesh.material = new THREE.MeshStandardMaterial({ color: "#4f8ff7", metalness: 0.15, roughness: 0.55 });

      const label = child.name || `Part ${parts.length + 1}`;
      let id = child.name || `part-${parts.length}`;
      while (usedIds.has(id)) id = `${id}#`;
      usedIds.add(id);
      parts.push({ id, label, object: child, visible: child.visible });
    });
    onPartsChange(parts);
    return () => onPartsChange([]);
  }, [gltf.scene, onPartsChange]);

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    const mesh = event.object as THREE.Mesh;
    const face = event.face;
    const geometry = mesh.geometry as THREE.BufferGeometry | undefined;
    if (!geometry?.attributes.position || !face) return;

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
      <primitive object={gltf.scene} onClick={handleClick} />
      {markerPosition && (
        <mesh position={markerPosition}>
          <sphereGeometry args={[1.6, 20, 20]} />
          <meshBasicMaterial color="#ffb020" />
        </mesh>
      )}
    </>
  );
}

// Neither StlMesh nor GlbMesh had any error boundary — a malformed/
// truncated mesh (a failed glTF export producing a corrupt .glb, a
// STLLoader parse error) throws out of useLoader's suspended render and,
// uncaught, crashes past the <Canvas> entirely instead of just failing to
// show one mesh. Class component because React error boundaries require
// componentDidCatch/getDerivedStateFromError, which only exist on classes
// — this works the same inside r3f's custom reconciler as it does in plain
// DOM React, since error-boundary lifecycle is a React tree concept, not a
// renderer-specific one. Keyed by meshUrl at the call site (like StlMesh/
// GlbMesh's own key) so a new mesh always gets a fresh, non-tripped
// boundary rather than staying stuck on a previous URL's failure.
class MeshErrorBoundary extends Component<
  { onError: (error: Error) => void; children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("[Viewer3D] mesh failed to load/render:", error);
    this.props.onError(error);
  }

  render() {
    return this.state.hasError ? null : this.props.children;
  }
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
//   (events-*.cjs.dev.js's unmountComponentAtNode/removeChild). An
//   explicit `gl.forceContextLoss()`/`gl.dispose()` pair also runs from
//   this component's own unmount effect below, as a belt-and-suspenders
//   safety net — both calls are idempotent in Three.js, so this never
//   double-frees anything; it just guarantees the context is released
//   even if r3f's own internals ever change.
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
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const [contextLost, setContextLost] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);

  // A new meshUrl (or a cleared one) always gets a clean slate — otherwise
  // a failure on one mesh would keep showing its error message forever,
  // even after a later mesh loads successfully.
  useEffect(() => {
    setMeshError(null);
  }, [meshUrl]);

  // Explicit safety-net teardown on THIS component's own unmount, on top of
  // react-three-fiber's automatic one (see the lifecycle notes above) —
  // belt-and-suspenders, not a replacement: r3f's unmountComponentAtNode
  // already calls forceContextLoss()/dispose() and recursively disposes the
  // scene graph, and both calls below are idempotent in Three.js (a no-op
  // if the context is already gone), so this never double-frees anything.
  // It exists so a lost/leaked WebGL context is guaranteed to be released
  // even if r3f's own internals ever change, or some edge case (a fast
  // repeated mount/unmount, a hot-reload) skips that automatic path — the
  // exact class of bug "THREE.WebGLRenderer: Context Lost" reports.
  useEffect(() => {
    return () => {
      const gl = glRef.current;
      if (!gl) return;
      gl.forceContextLoss();
      gl.dispose();
      glRef.current = null;
    };
  }, []);

  const isUrdf = !!meshUrl && isUrdfUrl(meshUrl);
  const isGlb = !!meshUrl && isGlbUrl(meshUrl);

  // Assembly part-visibility toggle — GlbMesh reports its scene's named
  // mesh nodes here via onPartsChange; toggling one flips `.visible`
  // directly on the live THREE.Object3D it holds a reference to (the same
  // property three.js's own render walk already checks, no extra
  // conditional rendering needed) and mirrors that into state so the
  // checkbox reflects it. Cleared whenever meshUrl changes so a stale
  // part list from a previous GLB never lingers in the panel.
  const [glbParts, setGlbParts] = useState<GlbPart[]>([]);
  useEffect(() => {
    setGlbParts([]);
  }, [meshUrl]);
  const toggleGlbPart = useCallback((id: string) => {
    setGlbParts((prev) =>
      prev.map((part) => {
        if (part.id !== id) return part;
        part.object.visible = !part.object.visible;
        return { ...part, visible: part.object.visible };
      })
    );
  }, []);

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

  // `<Canvas>` itself is ALWAYS rendered below — never gated behind
  // `meshUrl`/`urdfRobot`/an artifact-array length, and Viewer3D is in turn
  // rendered unconditionally by CadPlugin (see its own comment). Only what
  // goes INSIDE the canvas (StlMesh, the URDF <primitive>) is conditional.
  // This is deliberate, not incidental: a `{someArray.length > 0 &&
  // <Canvas>}`-style gate would remount the whole WebGLRenderer on every
  // intermediate ReAct step where the mesh payload is transiently null/[]
  // (e.g. a non-CAD tool call mid-turn) — exactly the repeated
  // create/destroy cycle that exhausts the browser's finite WebGL context
  // limit and surfaces as "THREE.WebGLRenderer: Context Lost". Keep the
  // canvas mounted and toggle content/placeholders instead, as done here.
  return (
    <div className="viewer3d">
      <Canvas
        ref={canvasRef}
        camera={{ position: [80, 80, 80], fov: 45 }}
        onCreated={({ gl }) => {
          glRef.current = gl;
        }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={1.1} />
        <Grid args={[400, 400]} cellColor="#333" sectionColor="#555" fadeDistance={400} />
        <OrbitControls makeDefault />
        <CameraRig cameraTarget={cameraTarget} />
        <Suspense fallback={null}>
          {meshUrl && !isUrdf && !isGlb && !contextLost && !meshError && (
            <MeshErrorBoundary key={meshUrl} onError={(err) => setMeshError(err.message)}>
              <StlMesh key={meshUrl} url={meshUrl} onSelect={onSelect} />
            </MeshErrorBoundary>
          )}
          {meshUrl && isGlb && !contextLost && !meshError && (
            <MeshErrorBoundary key={meshUrl} onError={(err) => setMeshError(err.message)}>
              <GlbMesh key={meshUrl} url={meshUrl} onSelect={onSelect} onPartsChange={setGlbParts} />
            </MeshErrorBoundary>
          )}
        </Suspense>
        {urdfRobot && !contextLost && (
          <primitive key={urdfRobot.uuid} object={urdfRobot} rotation={[-Math.PI / 2, 0, 0]} />
        )}
      </Canvas>
      {contextLost && <div className="viewer3d__placeholder">Recovering 3D view…</div>}
      {!contextLost && meshError && (
        <div className="viewer3d__placeholder">Failed to render mesh: {meshError}</div>
      )}
      {!contextLost && !meshError && !meshUrl && (
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
      {!contextLost && isGlb && glbParts.length > 1 && (
        <div className="viewer3d__joint-panel">
          <div className="viewer3d__joint-panel-title">Parts</div>
          {glbParts.map((part) => (
            <label key={part.id} className="viewer3d__part-row">
              <input
                type="checkbox"
                checked={part.visible}
                onChange={() => toggleGlbPart(part.id)}
              />
              <span className="viewer3d__part-name" title={part.label}>
                {part.label}
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
