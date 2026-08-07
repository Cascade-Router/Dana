/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#010409",
          900: "#0d1117",
          800: "#161b22",
          700: "#21262d",
          600: "#30363d",
        },
        teal: {
          DEFAULT: "#00f0ff",
          soft: "#5ce1ff",
          dim: "#0891b2",
          glow: "rgba(0, 240, 255, 0.28)",
        },
        amber: {
          DEFAULT: "#ffb000",
          soft: "#ffc94d",
          dim: "#9a6700",
        },
        // Keep violet token aliases mapped to amber so existing classes stay on-brand.
        violet: {
          DEFAULT: "#ffb000",
          soft: "#ffc94d",
          dim: "#9a6700",
          muted: "#8b949e",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        "teal-glow": "0 0 0 1px rgba(0, 240, 255, 0.35)",
        "logo-glow": "0 0 0 1px rgba(0, 240, 255, 0.25)",
      },
      borderRadius: {
        xl: "2px",
        "2xl": "2px",
      },
      keyframes: {
        floaty: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "glow-pulse": {
          "0%, 100%": { opacity: "0.55", filter: "blur(18px)" },
          "50%": { opacity: "0.9", filter: "blur(22px)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        caret: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
      },
      animation: {
        floaty: "floaty 5.5s ease-in-out infinite",
        "glow-pulse": "glow-pulse 4.5s ease-in-out infinite",
        "fade-up": "fade-up 0.7s ease-out both",
        caret: "caret 1s step-end infinite",
      },
    },
  },
  plugins: [],
};
