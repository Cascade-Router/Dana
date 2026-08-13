import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import * as THREE from "three";
import type { CameraTarget, CanvasSelection } from "../lib/useChatSocket";
import "./Viewer3D.css";

function StlMesh({ url, onSelect }: { url: string; onSelect: (selection: CanvasSelection) => void }) {
  const geometry = useLoader(STLLoader, url);
  geometry.computeVertexNormals();
  geometry.center();

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

export function Viewer3D({ meshUrl, cameraTarget, onSelect }: Props) {
  return (
    <div className="viewer3d">
      <Canvas camera={{ position: [80, 80, 80], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={1.1} />
        <Grid args={[400, 400]} cellColor="#333" sectionColor="#555" fadeDistance={400} />
        <OrbitControls makeDefault />
        <CameraRig cameraTarget={cameraTarget} />
        <Suspense fallback={null}>
          {meshUrl && <StlMesh key={meshUrl} url={meshUrl} onSelect={onSelect} />}
        </Suspense>
      </Canvas>
      {!meshUrl && <div className="viewer3d__placeholder">No geometry yet — ask Dana to build something.</div>}
    </div>
  );
}
