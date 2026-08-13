import { ChatPanel } from "./components/ChatPanel";
import { DAGMonitor } from "./components/DAGMonitor";
import { TerminalDrawer } from "./components/TerminalDrawer";
import { Viewer3D } from "./components/Viewer3D";
import { useChatSocket } from "./lib/useChatSocket";
import "./App.css";

export default function App() {
  const { connection, messages, log, meshUrl, cameraTarget, sendMessage, sendSelection, respondHitl } =
    useChatSocket();

  return (
    <div className="app">
      <div className="app__left">
        <ChatPanel connection={connection} messages={messages} onSend={sendMessage} onHitlRespond={respondHitl} />
        <TerminalDrawer log={log} />
      </div>
      <div className="app__right">
        <Viewer3D meshUrl={meshUrl} cameraTarget={cameraTarget} onSelect={sendSelection} />
        <DAGMonitor log={log} />
      </div>
    </div>
  );
}
