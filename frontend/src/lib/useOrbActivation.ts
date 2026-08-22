import { useCallback } from "react";
import type { VoiceState } from "./useChatSocket";

/**
 * One click/hotkey does double duty: start listening from idle, or cancel
 * an in-flight listen. No-op while "processing"/"speaking" — there's
 * nothing sensible to do mid-turn (interrupting playback is a possible
 * future addition, not wired here).
 */
export function useOrbActivation(
  state: VoiceState,
  requestListen: () => void,
  cancelListen: () => void
): () => void {
  return useCallback(() => {
    if (state === "idle") requestListen();
    else if (state === "listening") cancelListen();
  }, [state, requestListen, cancelListen]);
}
