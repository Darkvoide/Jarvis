"""JARVIS — entry point.

Usage:
    python main.py --mode text
    python main.py --mode voice
    python main.py --mode gesture
    python main.py --mode all
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

# Ensure Windows terminal supports Tamil / Unicode characters properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import colorlog

from jarvis.config import load_config
from jarvis.context import JarvisContext
from jarvis.orchestrator import Orchestrator
from jarvis.tools import ALL_TOOLS

# ── Logging setup ──────────────────────────────────────────────────────────────


def _setup_logging(level: str, log_file: str) -> None:
    fmt = "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(fmt))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        handlers=[handler, file_handler])


# ── Core ───────────────────────────────────────────────────────────────────────


def _inject_context(ctx: JarvisContext) -> None:
    """Wire the shared context into tool modules that need it."""
    from jarvis.tools import vision_tools, manipulation_tools
    vision_tools._set_context(ctx)
    manipulation_tools._set_context(ctx)


def _make_reply_handler(orch: Orchestrator) -> callable:
    """Return a thread-safe callback that runs a turn in the orchestrator."""
    lock = threading.Lock()

    def handle(user_input: str) -> None:
        with lock:
            print(f"\n🎧 You: {user_input}")
            response = orch.handle(user_input)
            print(f"🤖 JARVIS: {response}\n")

            # Auto-speak response if TTS is available
            try:
                from jarvis.tools.speech_tools import speak
                speak(response)
            except Exception:
                pass

    return handle


# ── Mode runners ───────────────────────────────────────────────────────────────


def run_text_mode(orch: Orchestrator) -> None:
    from jarvis.inputs.text_input import TextListener
    handle = _make_reply_handler(orch)
    listener = TextListener(on_text=handle)
    print("\n✅  JARVIS text mode active. Type a message and press Enter.")
    print("    Type 'exit' to quit.\n")
    listener.run()


def run_voice_mode(orch: Orchestrator, cfg) -> None:
    from jarvis.inputs.voice_input import VoiceListener
    handle = _make_reply_handler(orch)
    listener = VoiceListener(on_transcript=handle)
    listener.start()
    print(f"\n✅  JARVIS voice mode active. Hold [{cfg.input.ptt_key}] to speak.\n")

    # Keep main thread alive
    try:
        import time
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        listener.stop()


def run_all_mode(orch: Orchestrator, cfg) -> None:
    """Start all input modalities concurrently."""
    from jarvis.inputs.text_input import TextListener
    from jarvis.inputs.voice_input import VoiceListener

    handle = _make_reply_handler(orch)

    # Voice listener (background)
    voice = VoiceListener(on_transcript=handle)
    voice.start()

    # Gesture watcher (background) — only if enabled
    gesture = None
    if cfg.input.gesture_enabled:
        from jarvis.inputs.gesture_input import GestureWatcher
        gesture = GestureWatcher(on_intent=handle)
        gesture.start()
        print(f"✅  Gesture input active. Show gestures to the webcam.")

    print(f"\n✅  JARVIS FULL mode active.")
    print(f"    🎙  Hold [{cfg.input.ptt_key}] to speak.")
    print(f"    ⌨️   Or type below and press Enter.")
    print(f"    Type 'exit' to quit.\n")

    # Text input on main thread (blocks until user types 'exit' or Ctrl+C)
    text = TextListener(on_text=handle)
    text.run()

    # Clean shutdown in correct order
    voice.stop()
    if gesture is not None:
        gesture.stop()


# ── Entry ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — Multimodal AI Assistant",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "voice", "gesture", "all"],
        default="text",
        help="Input modality to activate (default: text)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a custom config.yaml",
    )
    args = parser.parse_args()

    # Load config
    from pathlib import Path
    cfg = load_config(Path(args.config) if args.config else None)
    _setup_logging(cfg.logging.level, cfg.logging.file)

    logger = logging.getLogger(__name__)
    logger.info("JARVIS starting — mode=%s, model=%s", args.mode, cfg.ollama.model)

    # Bootstrap context + orchestrator
    ctx = JarvisContext()
    _inject_context(ctx)
    orch = Orchestrator(ctx, tools=ALL_TOOLS)

    print("\n" + "═" * 55)
    print("   J A R V I S  —  Multimodal AI Assistant")
    print("═" * 55)
    print(f"   Model : {cfg.ollama.model}")
    print(f"   Mode  : {args.mode}")
    print(f"   STT   : {cfg.stt.model_size} / {cfg.stt.device}")
    print(f"   TTS   : {cfg.tts.backend}")
    print("═" * 55 + "\n")

    try:
        if args.mode == "text":
            run_text_mode(orch)
        elif args.mode == "voice":
            run_voice_mode(orch, cfg)
        elif args.mode == "gesture":
            # Gesture alone: gestures produce text intents → orchestrator
            from jarvis.inputs.gesture_input import GestureWatcher
            handle = _make_reply_handler(orch)
            watcher = GestureWatcher(on_intent=handle)
            watcher.start()
            print("✅  JARVIS gesture mode active. Show gestures to the webcam.")
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                watcher.stop()
        elif args.mode == "all":
            run_all_mode(orch, cfg)

    except KeyboardInterrupt:
        print("\n👋  JARVIS shutting down.")
    finally:
        from jarvis.inputs.vision_capture import release_camera
        release_camera()
        logger.info("JARVIS shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
