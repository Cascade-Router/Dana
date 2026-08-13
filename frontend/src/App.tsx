import { ChatPanel } from "./components/ChatPanel";
import { TerminalDrawer } from "./components/TerminalDrawer";
import { Viewer3D } from "./components/Viewer3D";
import { useChatSocket } from "./lib/useChatSocket";
import "./App.css";

export default function App() {
  const { connection, messages, log, meshUrl, sendMessage } = useChatSocket();

  return (
    <div className="app">
      <div className="app__left">
        <ChatPanel connection={connection} messages={messages} onSend={sendMessage} />
        <TerminalDrawer log={log} />
      </div>
      <div className="app__right">
        <Viewer3D meshUrl={meshUrl} />
      </div>
    </div>
  );
}
