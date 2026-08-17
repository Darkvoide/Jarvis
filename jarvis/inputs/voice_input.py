"""Voice input — hands-free continuous listening + push-to-talk STT using faster-whisper.

Modes:
  1. Continuous (Hands-Free): Listens to the microphone, auto-detects speech,
     records until silence, and automatically transcribes.
  2. Push-to-Talk: Hold the configured PTT key (default: F9) to record audio.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from jarvis.config import get_config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono
CHUNK_SIZE = 1024     # Read chunks


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
        """Hands-free voice activity detection (VAD) with fast turnaround."""
        ENERGY_THRESHOLD = 0.020   # RMS amplitude threshold for speech
        SILENCE_DURATION = 0.45    # Seconds of silence after speech to trigger instant transcription (was 0.8s)
        MAX_RECORD_SECONDS = 15.0  # Max recording turn duration

        silence_start = None
        is_speaking = False
        recorded_chunks: list[np.ndarray] = []

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
            ) as stream:
                while not self._stop_event.is_set():
                    data, overflow = stream.read(CHUNK_SIZE)
                    if overflow:
                        continue

                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk**2)))

                    if rms > ENERGY_THRESHOLD:
                        if not is_speaking:
                            is_speaking = True
                            print("🎙️  [Listening...]")
                            recorded_chunks.clear()
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
                            if len(audio) >= SAMPLE_RATE * 0.3:  # At least 300ms
                                threading.Thread(
                                    target=self._transcribe_and_emit,
                                    args=(audio,),
                                    daemon=True,
                                ).start()

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
            if self._recording:
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
                    min_silence_duration_ms=getattr(stt, "vad_min_silence_ms", 400),
                    threshold=0.3,
                )

            # Transcribe — beam_size=1 for speed, or 5 for max accuracy
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

            # Filter: reject low language-confidence outputs (noisy room, TV in background)
            min_prob = getattr(stt, "min_language_prob", 0.0)
            if min_prob > 0.0 and info.language_probability < min_prob:
                logger.debug(
                    "Skipping transcript — language prob %.2f < %.2f threshold (lang=%s)",
                    info.language_probability, min_prob, info.language,
                )
                return

            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                logger.info("Transcribed [%s %.0f%%]: %s", info.language, info.language_probability * 100, text)
                self.on_transcript(text)
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
