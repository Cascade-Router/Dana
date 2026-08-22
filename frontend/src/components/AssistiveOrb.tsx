import type { VoiceState } from "../lib/useChatSocket";
import "./AssistiveOrb.css";

const STATE_COLOR: Record<VoiceState, string> = {
  idle: "#3b82f6",
  listening: "#22d3ee",
  processing: "#f59e0b",
  speaking: "#34d399",
};

type Props = {
  state: VoiceState;
  transcript?: string;
  onActivate?: () => void;
};

// Presentational only — no socket, no window assumptions. OrbOverlay (the
// dedicated always-on-top window) feeds it live voiceState — the ONLY place
// this is currently rendered, so exactly one orb is ever on screen at once
// (the main window itself deliberately renders none of its own).
export function AssistiveOrb({ state, transcript, onActivate }: Props) {
  const color = STATE_COLOR[state];

  return (
    <div className="assistive-orb" style={{ "--orb-color": color } as React.CSSProperties}>
      {state === "listening" && (
        <>
          <span className="assistive-orb__ring assistive-orb__ring--1" />
          <span className="assistive-orb__ring assistive-orb__ring--2" />
        </>
      )}
      <button
        type="button"
        className={`assistive-orb__core assistive-orb__core--${state}`}
        onClick={onActivate}
        title={transcript || state}
        aria-label={`Dana assistant: ${state}`}
      >
        {state === "speaking" && (
          <span className="assistive-orb__waveform">
            <span />
            <span />
            <span />
            <span />
          </span>
        )}
        {state === "processing" && <span className="assistive-orb__spinner" />}
      </button>
    </div>
  );
}
