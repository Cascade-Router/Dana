import { Suspense } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import "./Viewer3D.css";

function StlMesh({ url }: { url: string }) {
  const geometry = useLoader(STLLoader, url);
  geometry.computeVertexNormals();
  geometry.center();

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial color="#4f8ff7" metalness={0.15} roughness={0.55} />
    </mesh>
  );
}

type Props = {
  meshUrl: string | null;
};

export function Viewer3D({ meshUrl }: Props) {
  return (
    <div className="viewer3d">
      <Canvas camera={{ position: [80, 80, 80], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={1.1} />
        <Grid args={[400, 400]} cellColor="#333" sectionColor="#555" fadeDistance={400} />
        <OrbitControls makeDefault />
        <Suspense fallback={null}>{meshUrl && <StlMesh key={meshUrl} url={meshUrl} />}</Suspense>
      </Canvas>
      {!meshUrl && <div className="viewer3d__placeholder">No geometry yet — ask Dana to build something.</div>}
    </div>
  );
}
