import { expect, test } from "@playwright/test";

/**
 * Regression guard for the exact bug class chased across this codebase's
 * own history: a `.glb` artifact whose extension sits in the MIDDLE of the
 * URL path — `/api/cad/artifacts/<name>.glb/download?session_id=...` (see
 * `dana/api/cad.py`'s `download_artifact` route), not at the very end —
 * silently falling through a naive/anchored-at-end loader check into
 * Viewer3D.tsx's STLLoader branch. Feeding real GLB bytes to STLLoader
 * produces a `RangeError: Invalid typed array length: <garbage>` (STLLoader
 * reads 4 garbage bytes at a fixed offset as a binary-STL triangle count
 * and allocates a Float32Array sized off it) — surfaced to the console via
 * Viewer3D's own MeshErrorBoundary.componentDidCatch, which is exactly what
 * this test watches for.
 *
 * No FreeCAD install is required: only the two REST endpoints the CAD
 * viewport actually calls (`GET /api/cad/artifacts`, `GET
 * /api/cad/artifacts/{filename}/download`) are exercised, and both are
 * intercepted here with a synthetic (but byte-for-byte valid) GLB — the
 * real `dana.api.server` process is still running underneath (started by
 * playwright.config.ts's `webServer`) so the WebSocket handshake that
 * assigns a real `session_id` still happens for real.
 */

function buildMinimalGlb(): Buffer {
  // Smallest legal glTF-Binary container: a JSON chunk describing one
  // empty scene, no BIN chunk (optional per the glTF2 spec) — enough for
  // three.js's GLTFLoader to parse successfully and hand back an (empty)
  // scene graph, with none of the "does this test's own fixture geometry
  // happen to be valid" surface area a real exported mesh would add.
  const json = Buffer.from(JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [] }] }), "utf-8");
  const padTo4 = (buf: Buffer, fill: number): Buffer => {
    const remainder = buf.length % 4;
    return remainder === 0 ? buf : Buffer.concat([buf, Buffer.alloc(4 - remainder, fill)]);
  };
  const jsonChunkData = padTo4(json, 0x20); // glTF2 spec: JSON chunk is space-padded

  const jsonChunkHeader = Buffer.alloc(8);
  jsonChunkHeader.writeUInt32LE(jsonChunkData.length, 0);
  jsonChunkHeader.write("JSON", 4, "ascii");

  const totalLength = 12 + jsonChunkHeader.length + jsonChunkData.length;
  const header = Buffer.alloc(12);
  header.write("glTF", 0, "ascii");
  header.writeUInt32LE(2, 4); // version
  header.writeUInt32LE(totalLength, 8);

  return Buffer.concat([header, jsonChunkHeader, jsonChunkData]);
}

const FIXTURE_GLB = buildMinimalGlb();
const FIXTURE_FILENAME = "PlaywrightFixtureMesh.glb";

test("a .glb history artifact whose extension sits mid-path loads via GLTFLoader without console errors", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  // GET /api/cad/artifacts?session_id=... — the CAD tab's own history list.
  await page.route(/\/api\/cad\/artifacts(\?[^/]*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        artifacts: [
          {
            filename: FIXTURE_FILENAME,
            format: "glb",
            size_bytes: FIXTURE_GLB.length,
            modified_at: Date.now() / 1000,
            source: "generated",
          },
        ],
      }),
    })
  );

  // GET /api/cad/artifacts/{filename}/download?session_id=... — same URL
  // shape as the real dana/api/cad.py route: extension mid-path, not at
  // the string's end.
  await page.route(/\/api\/cad\/artifacts\/[^/]+\/download(\?[^/]*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "model/gltf-binary", body: FIXTURE_GLB })
  );

  await page.goto("/");

  // Switch to the CAD tab (App.tsx renders each plugin tab's button with
  // its `name` as visible text — "CAD" here, see plugins/registry.ts).
  await page.getByRole("button", { name: /CAD/ }).click();

  // Waits out both the WebSocket handshake that assigns a real session_id
  // (unlocking useCadArtifacts' fetch) and the lazy CadPlugin chunk load.
  const historyItem = page.getByTitle(FIXTURE_FILENAME);
  await historyItem.waitFor({ state: "visible", timeout: 15_000 });
  await historyItem.click();

  // Lets GLTFLoader's async parse (and, on a regression, STLLoader's throw
  // + MeshErrorBoundary's console.error) actually run before asserting.
  await page.waitForTimeout(1500);

  // The literal strings this test was first asked to watch for
  // ("Invalid typed array length", the exact RangeError text from a real,
  // multi-KB corrupted GLB) plus "WebGL". Verified live that a MINIMAL
  // fixture GLB (this file's own, a few dozen bytes) makes STLLoader
  // misread a garbage-but-small face count, and V8 throws the sibling
  // message "Array buffer allocation failed" instead of "Invalid typed
  // array length: <N>" for that byte-length range — same defect, different
  // wording depending on how big the misread count happens to be. The
  // authoritative, wording-independent signal is Viewer3D's own
  // MeshErrorBoundary.componentDidCatch log line, which fires on ANY mesh
  // load/render failure regardless of the underlying error's exact text —
  // included so this test doesn't silently stop catching the regression if
  // a future browser/three.js version phrases the allocation error a third
  // way.
  const regressionErrors = consoleErrors.filter((text) =>
    /Invalid typed array length|Array buffer allocation failed|WebGL|\[Viewer3D\] mesh failed to load\/render/i.test(
      text
    )
  );
  expect(regressionErrors, `unexpected console errors: ${JSON.stringify(consoleErrors, null, 2)}`).toEqual([]);
});
