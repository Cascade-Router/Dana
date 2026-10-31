# Dānā Web — Stage 9.0

Astro + Tailwind dark-first landing page for Dānā, the agentic CAD engineer. Its copy and numbers come from the repo README, `docs/WHITE_PAPER.md` and the offline evals; update them together.

## Stack

- **Astro 4** (static output, compressed HTML)
- **Tailwind CSS 3** (dark-mode-first, teal `#00ADB5` + violet accents)
- **Vanilla scroll / typing animations** (no Framer/GSAP on the critical path — IntersectionObserver + CSS composites for ~60fps)

## Routes

| Path | Purpose |
|------|---------|
| `/` | Landing — hero, Hugging Face embed, session features, architecture, 3D part viewer, routing terminal |
| `/demo` | The Hugging Face Space in an iframe |
| `/terminal` | REST chat setup guide (global bar on every page) |

Legacy static site archived under `legacy/`.

## Stage 9.1 — Global REST chat

- Floating bar on every page (`GlobalChat.astro`) — teal accents, `[User (Text)]` / `[Dānā]` lines
- `src/utils/hf_api.ts` → `@gradio/client` `predict("/chat", { message })` against the Space (reply text is `data[0]`); one client per page keeps the Space-side session
- History in `sessionStorage` (`dana_chat_v1`) survives in-tab navigations
- Cmd/Ctrl+K opens; Esc collapses; pulsing teal border while waiting
- Cold boot / timeout → “Dānā is warming up…”

```bash
cp .env.example .env
# edit PUBLIC_DANA_HF_API=https://YOUR-SPACE.hf.space
```

## Develop

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

## GitHub Pages

Workflow: `.github/workflows/deploy_website.yml` — builds on `website/**` pushes to `main`, uploads `website/dist`. CI sets `BASE_PATH=/Dana` for project Pages; local keeps `base: "/"`. Enable **Settings → Pages → Source: GitHub Actions**.

## Assets

- `public/dana-logo.png` — the app logo
- `public/models/mounting-plate.glb` — the 3D viewer's model: the Phase 8 live run's mounting plate, rebuilt by the mock CAD engine
- `public/favicon.svg` — teal DA mark
