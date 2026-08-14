"""Speech output tool — TTS via edge-tts (neural online) or pyttsx3 (offline fallback)."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import re
import tempfile
import threading
import time

from jarvis.config import get_config

logger = logging.getLogger(__name__)

# Module-level pyttsx3 engine
_pyttsx3_engine = None
_engine_lock = threading.Lock()


def _is_tamil(text: str) -> bool:
    """Check if text contains Tamil Unicode characters."""
    return any("\u0b80" <= char <= "\u0bff" for char in text)


def _clean_for_speech(text: str) -> str:
    """Strip markdown symbols and format text for natural pronunciation."""
    # Remove markdown code blocks and urls
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`.*?`", "", text)
    text = re.sub(r"https?://\S+", "", text)
    # Remove bold, italics, headers, bullets
    text = re.sub(r"[*#_~>\[\]]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _play_audio_file(file_path: str) -> bool:
    """Play an audio file synchronously using Windows MCI (zero external UI/window)."""
    file_path = os.path.abspath(file_path)
    if os.name == "nt":
        winmm = ctypes.windll.winmm
        alias = f"jarvis_tts_{int(time.time() * 1000)}"
        winmm.mciSendStringW(f'close {alias}', None, 0, None)
        ret = winmm.mciSendStringW(f'open "{file_path}" type mpegvideo alias {alias}', None, 0, None)
        if ret == 0:
            winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
            winmm.mciSendStringW(f'close {alias}', None, 0, None)
            return True
    return False


def _get_pyttsx3_engine():
    global _pyttsx3_engine
    with _engine_lock:
        if _pyttsx3_engine is None:
            import pyttsx3
            cfg = get_config().tts
            _pyttsx3_engine = pyttsx3.init()
            _pyttsx3_engine.setProperty("rate", cfg.rate)
            _pyttsx3_engine.setProperty("volume", cfg.volume)
        return _pyttsx3_engine


def _speak_pyttsx3(text: str) -> None:
    engine = _get_pyttsx3_engine()
    with _engine_lock:
        engine.say(text)
        engine.runAndWait()


async def _generate_edge_tts(text: str, voice: str, output_path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _speak_edge_tts(text: str) -> bool:
    """Speak using Microsoft Edge Neural TTS with auto Tamil/English voice selection."""
    cfg = get_config().tts

    # Auto-pick natural voice based on language
    if _is_tamil(text):
        voice = "ta-IN-PallaviNeural"  # Clear natural Tamil voice
    else:
        voice = cfg.edge_voice or "en-US-ChristopherNeural"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        asyncio.run(_generate_edge_tts(text, voice, tmp_path))
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            _play_audio_file(tmp_path)
            return True
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def speak(text: str) -> dict:
    """Speak the given text aloud using the configured TTS engine.

    Args:
        text: The text to speak. Keep it concise and spoken-friendly.

    Returns:
        A dict with status and the spoken text.
    """
    if not text or not text.strip():
        return {"error": "No text provided to speak."}

    spoken_text = _clean_for_speech(text)
    if not spoken_text:
        return {"error": "No pronounceable text provided."}

    logger.info("speak: %s", spoken_text[:80])
    cfg = get_config().tts

    # Prefer edge-tts for high-quality audio
    if cfg.backend == "edge-tts":
        try:
            success = _speak_edge_tts(spoken_text)
            if success:
                return {"status": "ok", "spoken": spoken_text}
        except Exception as exc:
            logger.warning("edge-tts failed (%s), falling back to pyttsx3", exc)

    # Fallback to offline pyttsx3
    try:
        _speak_pyttsx3(spoken_text)
        return {"status": "ok", "spoken": spoken_text}
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return {"error": f"TTS failed: {exc}", "spoken": spoken_text}
