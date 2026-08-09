import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// Stage 9.0 / Pages — site + base are env-overridable for GitHub Actions.
// Local: http://localhost:4321/  |  Pages: https://cascade-router.github.io/Dana/
const onPages =
  process.env.ASTRO_PAGES === "1" || process.env.GITHUB_ACTIONS === "true";
const site =
  process.env.SITE_URL ||
  (onPages ? "https://cascade-router.github.io" : "http://localhost:4321");
const base = process.env.BASE_PATH || (onPages ? "/Dana/" : "/");
// Astro requires a trailing slash on project bases (e.g. /Dana/) so
// `${BASE_URL}dana-logo.png` resolves to /Dana/dana-logo.png — not /Danadana-logo.png.
const normalizedBase = base.endsWith("/") ? base : `${base}/`;

export default defineConfig({
  site,
  base: normalizedBase,
  integrations: [
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  build: {
    inlineStylesheets: "auto",
  },
  compressHTML: true,
  prefetch: {
    prefetchAll: false,
    defaultStrategy: "tap",
  },
});
