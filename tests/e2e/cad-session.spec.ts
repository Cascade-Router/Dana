import { expect, type Page, test } from "@playwright/test";

/**
 * A whole CAD session in the browser, against tests/e2e/mock_backend.py: the
 * real dana.api.server with the trimesh mock engine and a scripted LLM that
 * goes through the real gates (load_capability -> create_plan -> load_specific_tool
 * -> tools) and streams its reply slowly. Covers what was only
 * typechecked before: streamed assistant_delta text, usage_update in the
 * CostBar, the Deliverables bar (macro export, mesh, BOM popover) and the
 * Parameters tab's zero-token rebuild.
 */

const API_BASE = "http://127.0.0.1:8765";
const PLATE_REPLY =
  "Built a 60 x 40 x 5 mm plate and chamfered its top edges by 1 mm. Every dimension is editable in the Parameters tab.";

type Session = { consoleErrors: string[]; sentFrames: string[]; openedUrls: () => Promise<string[]> };

async function openCadSession(page: Page): Promise<Session> {
  const consoleErrors: string[] = [];
  const sentFrames: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(err.message));
  page.on("websocket", (ws) => ws.on("framesent", (frame) => sentFrames.push(String(frame.payload))));
  // DeliverablesBar downloads through window.open; record instead of opening tabs.
  await page.addInitScript(() => {
    const opened: string[] = [];
    (window as unknown as { __opened: string[] }).__opened = opened;
    window.open = ((url?: string | URL) => {
      opened.push(String(url));
      return null;
    }) as typeof window.open;
  });

  await page.goto("/");
  await page.getByRole("button", { name: /CAD/ }).click();
  await page.locator(".viewer3d canvas").waitFor({ state: "visible", timeout: 15_000 });
  await expect(page.getByRole("button", { name: "Send" })).toBeEnabled({ timeout: 15_000 });
  // Layout regressions found by this spec at the default 1280x720: the
  // composer used to overflow so the 3D canvas covered Send, and the "no
  // blueprint" placeholder showed over the canvas in 3D mode.
  const composer = page.locator(".chat-panel__composer");
  expect(await composer.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(0);
  await expect(page.locator(".blueprint-viewer__placeholder")).toBeHidden();
  return {
    consoleErrors,
    sentFrames,
    openedUrls: () => page.evaluate(() => (window as unknown as { __opened: string[] }).__opened),
  };
}

async function send(page: Page, text: string): Promise<void> {
  await page.getByPlaceholder("Ask Dana to do something...").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
}

// At 1280x720 the Terminal tab used to be clipped off the dock's right edge
// (reachable only by scrolling the tab bar). Run once the Plan and Parameters
// badges are showing, which is when the tabs are widest.
async function expectInspectorTabsFit(page: Page): Promise<void> {
  const tabbar = page.locator(".inspector-dock__tabbar");
  expect(await tabbar.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(0);
  const bar = (await tabbar.boundingBox())!;
  const viewport = page.viewportSize()!;
  for (const label of ["Topology", "Plan", "Parameters", "Terminal"]) {
    const tab = page.locator(".inspector-dock__tab", { hasText: label });
    const box = (await tab.boundingBox())!;
    expect(box.x, label).toBeGreaterThanOrEqual(bar.x);
    expect(box.x + box.width, label).toBeLessThanOrEqual(Math.min(bar.x + bar.width, viewport.width));
    await tab.click();
    await expect(tab).toHaveClass(/inspector-dock__tab--active/);
  }
}

function expectNoConsoleErrors(errors: string[]): void {
  expect(errors, `unexpected console errors: ${JSON.stringify(errors, null, 2)}`).toEqual([]);
}

test("a plate session streams its reply, fills the CostBar, exports a macro and rebuilds for 0 tokens", async ({
  page,
}) => {
  test.setTimeout(180_000);
  const session = await openCadSession(page);
  await send(page, "Build a 60 x 40 x 5 mm plate with a 1 mm chamfer on the top edges");

  // --- streaming: partial text first, then replaced by the final message ---
  const streaming = page.locator(".chat-panel__bubble--streaming");
  await expect(streaming).toContainText("Built a", { timeout: 60_000 });
  const partial = (await streaming.innerText()).trim();
  expect(partial.length).toBeLessThan(PLATE_REPLY.length);
  expect(PLATE_REPLY.startsWith(partial)).toBe(true);

  const finalBubble = page.locator(".chat-panel__bubble--assistant", { hasText: PLATE_REPLY });
  await expect(finalBubble).toHaveCount(1, { timeout: 30_000 });
  await expect(streaming).toHaveCount(0);

  // --- usage_update -> CostBar ---
  await expect(page.locator(".cost-bar__model")).toHaveText("e2e/scripted-model");
  await expect(page.locator(".cost-bar__total")).toHaveText(/^\$0\.0\d{3}$|^\$0\.\d{2}$/);

  // --- Deliverables bar renders every control without overflowing ---
  const bar = page.locator(".deliverables-bar");
  await expect(bar).toBeVisible();
  for (const name of ["Macro .py", "STEP", "Blueprint", "PDF", "BOM ▾"]) {
    await expect(bar.getByRole("button", { name, exact: true })).toBeVisible();
  }
  const meshButton = bar.getByRole("button", { name: /^(STL|GLB)$/ });
  await expect(meshButton).toBeEnabled({ timeout: 15_000 });
  const overflow = await bar.evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);

  // --- Macro export: export_python_script -> python_script_exported -> download ---
  await bar.getByRole("button", { name: "Macro .py" }).click();
  await expect.poll(session.openedUrls, { timeout: 15_000 }).toContainEqual(
    expect.stringMatching(/\/api\/cad\/artifacts\/[^/]+\.py\/download\?session_id=/)
  );
  expect(session.sentFrames.some((f) => f.includes('"export_python_script"'))).toBe(true);
  const macroUrl = (await session.openedUrls()).find((u) => u.includes(".py/download"))!;
  const macro = await page.request.get(new URL(macroUrl, API_BASE).toString());
  expect(macro.status()).toBe(200);
  const macroText = await macro.text();
  expect(macroText).toContain("import FreeCAD");
  expect(macroText).toContain("Plate");
  await expect(bar.getByRole("button", { name: "Macro .py" })).toBeEnabled();

  await expectInspectorTabsFit(page);

  // --- Parameters tab: units, dressup warning, edit, Apply & Rebuild ---
  await page.getByRole("button", { name: /Parameters/ }).first().click();
  const length = page.getByLabel("Plate.length");
  await expect(length).toHaveValue("60");
  await expect(page.locator(".parameters-tab__unit").first()).toHaveText("mm");
  await expect(page.locator(".parameters-tab__warning").first()).toBeVisible();

  await length.fill("80");
  const apply = page.getByRole("button", { name: "Apply & Rebuild (0 tokens)" });
  await expect(apply).toBeEnabled();
  await apply.click();
  await expect(page.locator(".parameters-tab__status")).toHaveText(/^Rebuilt/, { timeout: 30_000 });
  await expect(length).toHaveValue("80");
  await expect(page.locator(".cost-bar__rebuilds")).toHaveText("+1 free rebuild");
  expect(session.sentFrames.some((f) => f.includes('"type":"rebuild"'))).toBe(true);

  expectNoConsoleErrors(session.consoleErrors);
});

test("an assembly session's BOM popover opens inside the viewport with its stats", async ({ page }) => {
  test.setTimeout(180_000);
  const session = await openCadSession(page);
  await send(page, "Model a stand and give me its BOM");
  await expect(page.locator(".chat-panel__bubble--assistant", { hasText: "bill of materials is ready" })).toHaveCount(
    1,
    { timeout: 60_000 }
  );

  const bomButton = page.locator(".deliverables-bar").getByRole("button", { name: "BOM ▾" });
  await expect(bomButton).toBeEnabled();
  await bomButton.click();
  const popover = page.getByRole("dialog", { name: "Bill of materials summary" });
  await expect(popover).toBeVisible();
  await expect(popover).toContainText("Stand");
  await expect(popover.locator("dt", { hasText: "Parts" }).locator("+ dd")).toHaveText("2");
  await expect(popover.getByRole("button", { name: "Download CSV" })).toBeEnabled({ timeout: 15_000 });

  const box = await popover.boundingBox();
  const viewport = page.viewportSize()!;
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width);
  expect(box!.y + box!.height).toBeLessThanOrEqual(viewport.height);

  // A model built from assembly tools can't be replayed; the tab says so.
  await page.getByRole("button", { name: /Parameters/ }).first().click();
  await expect(page.getByText(/Rebuild is unavailable/)).toBeVisible();

  expectNoConsoleErrors(session.consoleErrors);
});
