/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#070b12",
          900: "#0b1220",
          800: "#121a2b",
          700: "#1a2438",
          600: "#243044",
        },
        teal: {
          DEFAULT: "#00ADB5",
          soft: "#33BDC4",
          dim: "#007A80",
          glow: "rgba(0, 173, 181, 0.35)",
        },
        violet: {
          DEFAULT: "#7B2CBF",
          soft: "#9B4DE0",
          dim: "#3D1560",
          muted: "#6B5B95",
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
      },
      boxShadow: {
        "teal-glow": "0 0 0 1px rgba(0, 173, 181, 0.45), 0 0 28px rgba(0, 173, 181, 0.22)",
        "logo-glow": "0 0 40px rgba(0, 173, 181, 0.28), 0 0 80px rgba(123, 44, 191, 0.12)",
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
