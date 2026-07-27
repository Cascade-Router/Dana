# Dānā (دانا) Web — Stage 9.0

Astro + Tailwind dark-first landing for the Dānā cybernetic brand.

## Stack

- **Astro 4** (static output, compressed HTML)
- **Tailwind CSS 3** (dark-mode-first, teal `#00ADB5` + violet accents)
- **Vanilla scroll / typing animations** (no Framer/GSAP on the critical path — IntersectionObserver + CSS composites for ~60fps)

## Routes

| Path | Purpose |
|------|---------|
| `/` | Landing — hero logo float, bento features, routing terminal |
| `/demo` | REST chat setup guide (global bar on every page) |
| `/terminal` | Same guide, terminal framing |

Legacy static site archived under `legacy/`.

## Stage 9.1 — Global REST chat

- Floating bar on every page (`GlobalChat.astro`) — teal accents, `[User (Text)]` / `[Dānā]` lines
- `src/utils/hf_api.ts` → `POST {data:[prompt]}` to `/api/predict` (fallback `/run/predict`)
- History in `sessionStorage` (`dana_chat_v1`) survives in-tab navigations
- Cmd/Ctrl+K opens; Esc collapses; pulsing teal border while waiting
- Cold boot / timeout → “Dānā is warming up…”

```bash
cp .env.example .env
# edit PUBLIC_DANA_HF_API=https://YOUR-SPACE.hf.space
```

## Develop

Requires Node.js ≥ 18.17.

```bash
cd website
npm install
npm run dev
```

Build:

```bash
npm run build
npm run preview
```

## Assets

- `public/dana-logo.png` — copy of `donna/ui/assets/dana_logo_highres.png`
- `public/favicon.svg` — teal DA mark
