# JARVIS — Multimodal AI Assistant

> A continuous, multimodal personal AI assistant powered by a local Ollama LLM with native tool calling. Supports **voice**, **text**, **vision**, and **hand-gesture** input.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10–3.12 | Required by faster-whisper |
| [Ollama](https://ollama.com) running | `ollama serve` must be active |
| A tool-calling model | `ollama pull qwen2.5:7b` (recommended) |
| Webcam | For vision + gesture input |
| Microphone | For voice input |

---

## Quick Start

### 1. Clone & install

```powershell
cd d:\Jarvis
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Pull an Ollama model

```powershell
# Main reasoning model (tool calling)
ollama pull qwen2.5:7b

# Optional: vision model
ollama pull llava:7b
```

Then set `ollama.vision_model: "llava:7b"` in `config.yaml` if you pulled one.

### 3. Run JARVIS

```powershell
# Text only (great for first test)
python main.py --mode text

# Voice + text (hold F9 to speak)
python main.py --mode voice

# Gesture + voice + text + vision (full)
python main.py --mode all
```

---

## Input Modes

| Mode | How it works |
|---|---|
| `text` | Type in the terminal, press Enter |
| `voice` | Hold **F9** → speak → release → JARVIS transcribes and responds |
| `gesture` | Show hand gestures to webcam (runs alongside any other mode) |
| `all` | All modalities active simultaneously |

---

## Gesture Map

| Gesture | Action triggered |
|---|---|
| ✌️ Victory / Peace | `detect_object` — scan scene |
| ☝️ Pointing Up | `select_object` — pick object |
| 🤏 Pinch | `zoom_object` — zoom in |
| 🤚 Open Palm | Cancel / stop current action |
| 👍 Thumbs Up | Confirm last action |
| ✊ Fist | `delete_object` |

---

## Configuration

All settings live in [`config.yaml`](config.yaml):

- **`ollama.model`** — reasoning model (default: `qwen2.5:7b`)
- **`ollama.vision_model`** — vision model (`null` to stub)
- **`stt.model_size`** — Whisper model size (`base` → `large-v3-turbo`)
- **`tts.backend`** — `pyttsx3` (offline) or `edge-tts` (internet, better quality)
- **`input.ptt_key`** — push-to-talk hotkey (default: `F9`)
- **`permissions.allow_shell`** — enable/disable shell execution

---

## Project Structure

```
d:\Jarvis\
├── main.py                    ← entry point
├── config.yaml                ← all settings
├── requirements.txt
└── jarvis\
    ├── system_prompt.py       ← JARVIS system prompt
    ├── config.py              ← config loader
    ├── orchestrator.py        ← agentic loop (LLM ↔ tools)
    ├── context.py             ← shared state (objects, history)
    ├── inputs\
    │   ├── voice_input.py     ← faster-whisper STT
    │   ├── gesture_input.py   ← MediaPipe gesture recognition
    │   ├── text_input.py      ← terminal readline
    │   └── vision_capture.py  ← OpenCV camera capture
    └── tools\
        ├── vision_tools.py    ← detect/select/zoom/analyze
        ├── manipulation_tools.py ← move/rotate/duplicate/delete
        ├── system_tools.py    ← open_app/execute_command/read_file
        ├── web_tools.py       ← search_web
        └── speech_tools.py    ← speak
```

---

## Language Support

JARVIS detects and responds in **English**, **Tamil**, or **Tanglish** (mixed) automatically. Whisper handles multilingual transcription natively — no extra setup needed.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Connection refused` from Ollama | Run `ollama serve` in a separate terminal |
| Gesture not detected | Ensure webcam is not in use by another app; check `camera_index` in config |
| Audio not captured | Check microphone permissions in Windows Settings |
| `keyboard` module needs admin | Run terminal as Administrator for global hotkey support |
