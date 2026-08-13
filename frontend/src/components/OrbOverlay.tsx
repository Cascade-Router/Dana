import { useEffect } from "react";
import { useChatSocket } from "../lib/useChatSocket";
import "./OrbOverlay.css";

const STATE_COLOR: Record<string, string> = {
  idle: "#3b82f6",
  listening: "#22d3ee",
  processing: "#f59e0b",
  speaking: "#34d399",
};

export function OrbOverlay() {
  const { voiceState } = useChatSocket();

  useEffect(() => {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
  }, []);

  const color = STATE_COLOR[voiceState.state] ?? STATE_COLOR.idle;

  return (
    <div className="orb-overlay">
      <div
        className={`orb orb--${voiceState.state}`}
        style={{ "--orb-color": color } as React.CSSProperties}
        title={voiceState.transcript || voiceState.state}
      />
    </div>
  );
}
