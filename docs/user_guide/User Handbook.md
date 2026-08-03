## 🎛️ Operating Modes

🟢 Chat Mode (Default) A lightweight, low-latency conversational mode powered by local Llama 3.2. It maintains a rolling memory buffer of your last 5 turns and possesses real-time situational awareness (including the current time, acknowledging you as Amirhosein, and reading live system CPU/RAM load). Tool calling and ReAct loops are strictly bypassed to guarantee fast responses.

🟣 Developer Mode The heavy-duty engineering pipeline. Audio prompts are routed through the MoA (Mixture of Agents) reasoner (DeepSeek-R1) and execute system tools like `draft_cursor_prompt` to dynamically generate tickets on your active patch ledger.

🔵 Vision Mode (Scaffolded) A foundational mode wired to route queries to your local YOLOv8n tracker and vision models.

🟠 Research Mode (Scaffolded) A foundational read-only mode designated for web search and information gathering, strictly sandboxed from local file execution.

## 🎙️ Voice Commands

### 🗣️ Wake & Interaction

- **"Donna"**: Wakes the system from standby. Donna replies with *"Yes?"* using prioritized zero-latency audio and opens the microphone.
- **Implicit Standby**: Waking the system without speaking (or just generating background noise) causes the microphone to silently time out and return to standby without announcing itself.

### 🔄 Mode Switching (Fast-Path Triggers)

- **"Donna, switch to chat mode"** ➔ *"Chat mode active."*
- **"Donna, switch to developer mode"** ➔ *"Developer mode active."*
- **"Donna, switch to vision mode"** ➔ *"Vision mode active."*
- **"Donna, switch to research mode"** ➔ *"Research mode active."*

### 🧠 Memory Management

- **"Donna, clear chat memory"** (or **"Donna, reset conversation"**): Instantly flushes the Chat Mode rolling buffer to prevent context-window overflow and plays *"Memory cleared."*

## ⚙️ Automated System Behaviors

- 🛑 **Barge-In Interruptions**: You can interrupt Donna while she is speaking. Her TTS Output Spooler instantly flushes the queue, stopping playback so you can seamlessly issue new commands.
- 🛡️ **Hardware Protections**: The wake-word listener is physically locked out during heavy LLM model loads (like Ollama warm-ups) to prevent CPU starvation and audio stuttering.
- 🗄️ **Ledger Auto-Archiving**: On boot, any tickets marked `[RESOLVED]` or `[FAILED]` are automatically swept from `patch_ledger.md` to `patch_ledger_archive.md` to keep the active workspace lean.
- 📊 **Telemetry Awareness**: While in Chat Mode, you can ask Donna how the system is running, and she accurately reports the live physical hardware load of your machine.

