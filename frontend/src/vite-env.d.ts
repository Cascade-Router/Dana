/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  /** Hugging Face Space base URL (e.g. https://user-space.hf.space) — set
   * this to build a static frontend that talks to app.py's Gradio chat API
   * instead of the native /ws/chat WebSocket. See lib/useChat.ts. */
  readonly VITE_HF_SPACE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
