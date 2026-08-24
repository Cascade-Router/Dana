import { useChatSocket } from "./useChatSocket";
import { useGradioChat } from "./useGradioChat";

// Picked ONCE, at module load — VITE_HF_SPACE_URL is a build-time Vite env
// var, never a value that changes across a render, so this stays compliant
// with React's Rules of Hooks (exactly one of these two hooks is ever
// called, for the entire lifetime of the app — never a conditional hook
// call inside a component). See useGradioChat's own docstring for exactly
// what differs behind this same call signature/return shape.
export const useChat = import.meta.env.VITE_HF_SPACE_URL ? useGradioChat : useChatSocket;
