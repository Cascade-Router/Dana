import { expect, test } from "@playwright/test";

/**
 * Regression guard for the CAD plugin's 3D Assembly / 2D Blueprint toggle
 * (CadPlugin.tsx) and BlueprintViewer.tsx — the .svg sibling
 * dana.plugins.freecad.techdraw_export.generate_2d_blueprint now produces
 * alongside its .pdf.
 *
 * Two things this specifically watches for, both real risks the
 * implementation was designed around:
 *  1. Viewer3D must NEVER be unmounted by the toggle (its own lifecycle
 *     notes: a conditional mount tears down/recreates the WebGLRenderer,
 *     exhausting the browser's finite WebGL context budget) — the toggle
 *     hides it with CSS (`display: none`) instead. This test asserts the
 *     canvas element still EXISTS in the DOM while the 2D tab is active,
 *     not just that switching doesn't crash.
 *  2. BlueprintViewer resolves its artifact URL the same way
 *     MeshHistoryPicker/CadToolbar already do (`resolveArtifactUrl`) and
 *     renders via a plain `<img>` (the browser's own native SVG rendering,
 *     no client-side DXF/SVG parsing library) — this asserts the `<img>`
 *     actually appears with the right `src` once real (intercepted)
 *     artifact data is available, not just that the placeholder shows.
 *
 * No FreeCAD install required: only the two REST endpoints the CAD
 * viewport actually calls are intercepted (same technique as
 * mesh-loader.spec.ts), with a minimal but valid SVG fixture. The real
 * dana.api.server process is still running underneath (playwright.config.ts's
 * webServer) so the WebSocket handshake that assigns a real session_id still
 * happens for real.
 */

const FIXTURE_FILENAME = "PlaywrightFixtureBlueprint.svg";
const FIXTURE_SVG = Buffer.from(
  '<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 100 100">' +
    '<rect width="100" height="100" fill="white"/><path d="M 10 10 L 90 10 L 90 90 L 10 90 Z" stroke="black" fill="none"/>' +
    "</svg>",
  "utf-8"
);

test("2D Blueprint toggle shows the SVG artifact without unmounting Viewer3D's canvas", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  // GET /api/cad/artifacts?session_id=... — same list BlueprintViewer/
  // CadToolbar/MeshHistoryPicker all share via useCadArtifacts.
  await page.route(/\/api\/cad\/artifacts(\?[^/]*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        artifacts: [
          {
            filename: FIXTURE_FILENAME,
            format: "svg",
            size_bytes: FIXTURE_SVG.length,
            modified_at: Date.now() / 1000,
            source: "generated",
          },
        ],
      }),
    })
  );

  // GET /api/cad/artifacts/{filename}/download?session_id=... — same URL
  // shape dana/api/cad.py's real route serves, extension mid-path.
  await page.route(/\/api\/cad\/artifacts\/[^/]+\/download(\?[^/]*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "image/svg+xml", body: FIXTURE_SVG })
  );

  await page.goto("/");

  // Switch to the CAD tab (App.tsx renders each plugin tab's button with
  // its `name` as visible text — "CAD" here, see plugins/registry.ts).
  await page.getByRole("button", { name: /CAD/ }).click();

  // Viewer3D's own <canvas> (from react-three-fiber's <Canvas>) — waiting
  // for it confirms the lazy CadPlugin chunk loaded and the 3D view is up
  // BEFORE we touch the toggle, so its later continued presence in the DOM
  // is a real assertion, not a false pass from the plugin never having
  // mounted at all.
  const canvas = page.locator(".viewer3d canvas");
  await canvas.waitFor({ state: "visible", timeout: 15_000 });

  await page.getByRole("button", { name: "2D Blueprint" }).click();

  // The canvas must still be IN THE DOM (not unmounted) even though it's
  // now visually hidden via CSS — this is the whole point of the
  // display:none approach over a conditional {viewMode === "3d" && ...}.
  await expect(canvas).toBeAttached();
  await expect(canvas).toBeHidden();

  const blueprintImage = page.locator(".blueprint-viewer__image");
  await blueprintImage.waitFor({ state: "visible", timeout: 10_000 });
  await expect(blueprintImage).toHaveAttribute("src", new RegExp(FIXTURE_FILENAME));

  // Switch back — Viewer3D must reappear (still the SAME canvas element,
  // never having been torn down) and the blueprint image hides in turn.
  await page.getByRole("button", { name: "3D Assembly" }).click();
  await expect(canvas).toBeVisible();
  await expect(blueprintImage).toBeHidden();

  expect(consoleErrors, `unexpected console errors: ${JSON.stringify(consoleErrors, null, 2)}`).toEqual([]);
});

test("2D Blueprint tab shows a placeholder, not a crash, when no blueprint has been generated yet", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(err.message));

  await page.route(/\/api\/cad\/artifacts(\?[^/]*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, artifacts: [] }) })
  );

  await page.goto("/");
  await page.getByRole("button", { name: /CAD/ }).click();
  await page.locator(".viewer3d canvas").waitFor({ state: "visible", timeout: 15_000 });
  await page.getByRole("button", { name: "2D Blueprint" }).click();

  await expect(page.locator(".blueprint-viewer__placeholder")).toBeVisible();
  expect(consoleErrors, `unexpected console errors: ${JSON.stringify(consoleErrors, null, 2)}`).toEqual([]);
});
