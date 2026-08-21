"""Voice input — hands-free continuous listening + push-to-talk STT using faster-whisper.

Modes:
  1. Continuous (Hands-Free): Listens to the microphone, auto-detects speech,
     records until silence, and automatically transcribes.
  2. Push-to-Talk: Hold the configured PTT key (default: F9) to record audio.
"""

from __future__ import annotations

from collections import deque
import logging
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from jarvis.config import get_config
from jarvis.tools.speech_tools import is_tts_playing

# Ensure Windows stdout/stderr handles Unicode characters cleanly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono
CHUNK_SIZE = 1024     # Read chunks (~64ms per chunk)
PRE_ROLL_CHUNKS = 6   # ~384ms pre-speech audio buffer so first syllables are never cut off


class VoiceListener:
    """Listens to the microphone and emits transcribed speech to a callback.

    Args:
        on_transcript: Callback called with the transcribed text string
                       whenever speech is captured and processed.
    """

    def __init__(self, on_transcript: Callable[[str], None]) -> None:
        self.on_transcript = on_transcript
        self._cfg = get_config()
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_speaking_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start listening in a background thread and pre-warm model."""
        # Pre-warm model in background so the first speech is transcribed instantly
        threading.Thread(target=self._load_model, daemon=True).start()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        mode = getattr(self._cfg.stt, "mode", "continuous")
        if mode == "ptt":
            print(f"🎙️  Voice active (Push-to-Talk): Hold [{self._cfg.input.ptt_key}] to speak.")
        else:
            print("🎙️  Voice active (Hands-Free): Speak directly into your microphone anytime!")

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._stop_event.set()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load faster-whisper model and keep warm in memory."""
        if self._model is None:
            with self._is_speaking_lock:
                if self._model is None:
                    from faster_whisper import WhisperModel
                    stt = self._cfg.stt
                    logger.info("Loading speech recognition model (%s, %s)...", stt.model_size, stt.device)
                    self._model = WhisperModel(
                        stt.model_size,
                        device=stt.device,
                        compute_type=stt.compute_type,
                        cpu_threads=4,
                        num_workers=1,
                    )
                    logger.info("Speech recognition model ready.")
        return self._model

    def _run(self) -> None:
        """Background thread: runs hands-free VAD or PTT loop."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            logger.error("sounddevice missing: %s", exc)
            return

        mode = getattr(self._cfg.stt, "mode", "continuous")

        if mode == "ptt":
            self._run_ptt(sd)
        else:
            self._run_continuous(sd)

    def _run_continuous(self, sd) -> None:
        """Hands-free voice activity detection (VAD) with adaptive noise calibration."""
        # Sensible thresholds for clear voice detection
        DEFAULT_ENERGY_THRESHOLD = 0.012   # Lowered from 0.020 so normal speech is caught
        SILENCE_DURATION = 0.65            # Seconds of silence after speech to complete utterance
        MAX_RECORD_SECONDS = 15.0          # Max recording turn duration

        silence_start = None
        is_speaking = False
        recorded_chunks: list[np.ndarray] = []
        pre_roll_buffer: deque[np.ndarray] = deque(maxlen=PRE_ROLL_CHUNKS)
        _tts_ended_at: list[float] = [0.0]  # mutable container for post-TTS cooldown tracking

        # Baseline noise calibration
        noise_samples = []
        calibrated_threshold = DEFAULT_ENERGY_THRESHOLD

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
            ) as stream:
                while not self._stop_event.is_set():
                    # If JARVIS is currently speaking TTS output, mute mic listening to avoid echo loops
                    if is_tts_playing():
                        is_speaking = False
                        silence_start = None
                        recorded_chunks.clear()
                        pre_roll_buffer.clear()
                        time.sleep(0.05)
                        _tts_ended_at[0] = time.time()  # mark when TTS last finished
                        continue

                    # After TTS ends, hold a short cooldown so room echo doesn't trigger the mic
                    if time.time() - _tts_ended_at[0] < 0.9:
                        time.sleep(0.02)
                        continue

                    data, overflow = stream.read(CHUNK_SIZE)
                    if overflow:
                        continue

                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk**2)))

                    # Ambient noise calibration for first ~1 second
                    if len(noise_samples) < 15:
                        noise_samples.append(rms)
                        if len(noise_samples) == 15:
                            avg_noise = float(np.mean(noise_samples))
                            calibrated_threshold = max(DEFAULT_ENERGY_THRESHOLD, avg_noise * 2.2)
                            logger.info(
                                "Microphone calibrated: noise_floor=%.4f, speech_threshold=%.4f",
                                avg_noise, calibrated_threshold,
                            )
                        continue

                    if rms > calibrated_threshold:
                        if not is_speaking:
                            is_speaking = True
                            print("🎙️  [Listening...]")
                            recorded_chunks.clear()
                            # Prepend pre-roll buffer so the start of words is never cut off
                            recorded_chunks.extend(list(pre_roll_buffer))
                        recorded_chunks.append(chunk)
                        silence_start = None
                    elif is_speaking:
                        recorded_chunks.append(chunk)
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            # Speech ended -> immediately process
                            is_speaking = False
                            silence_start = None
                            audio = np.concatenate(recorded_chunks, axis=0)
                            recorded_chunks.clear()
                            if len(audio) >= SAMPLE_RATE * 0.35:  # At least 350ms
                                threading.Thread(
                                    target=self._transcribe_and_emit,
                                    args=(audio,),
                                    daemon=True,
                                ).start()
                    else:
                        pre_roll_buffer.append(chunk)

                    # Guard against runaway recording
                    if is_speaking and len(recorded_chunks) * CHUNK_SIZE > SAMPLE_RATE * MAX_RECORD_SECONDS:
                        is_speaking = False
                        silence_start = None
                        audio = np.concatenate(recorded_chunks, axis=0)
                        recorded_chunks.clear()
                        threading.Thread(
                            target=self._transcribe_and_emit,
                            args=(audio,),
                            daemon=True,
                        ).start()

        except Exception as exc:
            err = str(exc)
            # PaErrorCode -9983: stream stopped (happens on Ctrl+C shutdown) — not a real error
            if "-9983" in err or "Stream is stopped" in err:
                logger.debug("Audio stream closed cleanly.")
            else:
                logger.error("Continuous voice input error: %s", exc)

    def _run_ptt(self, sd) -> None:
        """Push-to-talk key recording."""
        try:
            import keyboard
        except ImportError as exc:
            logger.error("keyboard missing for PTT mode: %s", exc)
            return

        ptt_key = self._cfg.input.ptt_key.lower()

        def _audio_callback(indata, frames, time_info, status):
            if self._recording and not is_tts_playing():
                self._frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_audio_callback,
            ):
                while not self._stop_event.is_set():
                    if keyboard.is_pressed(ptt_key):
                        if not self._recording:
                            self._recording = True
                            self._frames.clear()
                            print(f"🎙️  [Recording: Hold {ptt_key.upper()}...]")
                    else:
                        if self._recording:
                            self._recording = False
                            print("⚡  [Transcribing speech...]")
                            if self._frames:
                                audio = np.concatenate(self._frames, axis=0).flatten()
                                self._frames.clear()
                                threading.Thread(
                                    target=self._transcribe_and_emit,
                                    args=(audio,),
                                    daemon=True,
                                ).start()
                    time.sleep(0.015)
        except Exception as exc:
            logger.error("PTT voice input error: %s", exc)

    def _transcribe_and_emit(self, audio: np.ndarray) -> None:
        """Transcribe audio with faster-whisper and emit to JARVIS."""
        try:
            model = self._load_model()
            stt = self._cfg.stt

            # Build VAD parameters from config
            vad_params = None
            if stt.vad_filter:
                vad_params = dict(
                    min_silence_duration_ms=getattr(stt, "vad_min_silence_ms", 300),
                    threshold=0.3,
                )

            # Transcribe — beam_size=1 for fast turnaround
            segments, info = model.transcribe(
                audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                language=stt.language,
                initial_prompt=getattr(stt, "initial_prompt", None),
                vad_filter=stt.vad_filter,
                vad_parameters=vad_params,
                no_speech_threshold=getattr(stt, "no_speech_threshold", 0.6),
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            # Clean and filter out hallucinated artifacts or empty noise
            if not text:
                return

            # Remove noise hallucinations (e.g. repeated dots or single characters)
            cleaned = text.replace(".", "").replace(",", "").replace("-", "").strip()
            if len(cleaned) < 2:
                return

            # Reject low language-confidence transcripts (background TV/room noise produces
            # low-confidence language detections like [tr 46%] or [id 42%])
            min_prob = getattr(stt, "min_language_prob", 0.0)
            if min_prob > 0.0 and info.language_probability < min_prob:
                logger.debug(
                    "Skipping low-confidence transcript [%s %.0f%%]: %s",
                    info.language, info.language_probability * 100, text,
                )
                return

            logger.info("Transcribed [%s %.0f%%]: %s", info.language, info.language_probability * 100, text)
            self.on_transcript(text)
        except Exception as exc:
            logger.error("Transcription error: %s", exc)

