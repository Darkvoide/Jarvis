"""
STEP 4 — TTS-only test (Edge TTS).
Only run this if step2 printed your words correctly (STT is fine).
Tests the TTS pipeline independently: does JARVIS speak back correctly?

HOW TO USE:
  python diagnose/step4_tts_only.py
"""
import asyncio
import ctypes
import os
import tempfile
import time

TEST_TEXTS = [
    ("English", "Hello, I am JARVIS. Your voice assistant is working correctly."),
    ("Tamil",   "வணக்கம். நான் ஜார்விஸ். உங்கள் குரல் உதவியாளர் சரியாக வேலை செய்கிறது."),
]

def _play_mp3(path: str) -> None:
    """Play an MP3 file using Windows MCI (no external player needed)."""
    if os.name != "nt":
        print("  (Non-Windows: skipping playback, file saved for manual check)")
        return
    winmm = ctypes.windll.winmm
    alias = f"tts_test_{int(time.time() * 1000)}"
    winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
    winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
    winmm.mciSendStringW(f'close {alias}', None, 0, None)


async def _speak(text: str, voice: str, label: str) -> None:
    try:
        import edge_tts
    except ImportError:
        print("❌ edge-tts not installed. Run: pip install edge-tts")
        return

    print(f"\n🔊 Speaking [{label}] via edge-tts...")
    print(f"   Voice : {voice}")
    print(f"   Text  : {text[:60]}...")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        size = os.path.getsize(tmp_path)
        print(f"   File size : {size} bytes", end="  ")
        if size < 500:
            print("← ⛔ Too small — TTS request probably failed (check internet)")
        else:
            print("← ✅ Generated OK")
            _play_mp3(tmp_path)
            print("   ✅ Playback done")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def main():
    print("=" * 60)
    print("  TTS Diagnostic — Edge TTS Standalone Test")
    print("=" * 60)
    print("  You should hear two sentences (English + Tamil).")
    print("  No speech = network issue or edge-tts not installed.")
    print("=" * 60)

    await _speak(TEST_TEXTS[0][1], "en-US-ChristopherNeural", TEST_TEXTS[0][0])
    await _speak(TEST_TEXTS[1][1], "ta-IN-PallaviNeural",     TEST_TEXTS[1][0])

    print("\n" + "=" * 60)
    print("  Results:")
    print("  Heard both  → TTS is working ✅")
    print("  Heard none  → Network issue or edge-tts not installed")
    print("  Heard only English → Tamil voice not available (region issue)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
