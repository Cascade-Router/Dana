import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// Stage 9.0 — static, dark-first Dānā landing (no SSR bloat).
export default defineConfig({
  site: "https://dana.local",
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
