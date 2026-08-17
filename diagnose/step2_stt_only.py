"""
STEP 2 - Minimal STT-only test.  No JARVIS, no orchestrator, no TTS, no gestures.
Just: mic -> faster-whisper -> print.

If this prints what you say  -> the mic+STT path is fine (check TTS with step4).
If this prints nothing/junk  -> step3 already found your mic at 44100 Hz native rate.
                                 sounddevice resamples to 16000 Hz automatically.
                                 If speech is still not recognized, try DEVICE_INDEX = 5
                                 (Microphone Array at 44100 Hz, the clean entry).

HOW TO USE:
  1. Put on wired earphones  <- eliminates all echo before writing any code
  2. Run:  .venv\\Scripts\\python.exe diagnose\\step2_stt_only.py
  3. Speak something in English or Tamil
  4. Watch: does the terminal print your words?
"""
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# -- EDIT THESE IF NEEDED -----------------------------------------------------
MODEL_SIZE        = "base"   # base = fastest. "small" = better Tamil accuracy.
DEVICE_INDEX      = -1       # -1 = system default (index 1, Realtek Array 44100Hz)
                             # Try 5 or 9 if default does not work.
SILENCE_THRESHOLD = 0.015    # RMS level to detect speech. Lower = more sensitive.
SILENCE_SECS      = 0.5      # Seconds of silence before sending to Whisper.
# -----------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK       = 1024

print("=" * 60)
print("  STT Diagnostic - Minimal Whisper Test")
print("=" * 60)
dev = sd.query_devices(DEVICE_INDEX if DEVICE_INDEX >= 0 else sd.default.device[0])
print(f"  Model       : {MODEL_SIZE}")
print(f"  Microphone  : [{DEVICE_INDEX}] {dev['name']}")
print(f"  Native rate : {int(dev['default_samplerate'])} Hz  (resampled to {SAMPLE_RATE} Hz for Whisper)")
print("=" * 60)

print("\nLoading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
print("Model ready. Speak now (Ctrl+C to stop).\n")

recorded = []
silence_start = None
is_speaking   = False

stream_kwargs = dict(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK)
if DEVICE_INDEX >= 0:
    stream_kwargs["device"] = DEVICE_INDEX

try:
    with sd.InputStream(**stream_kwargs) as stream:
        while True:
            data, overflow = stream.read(CHUNK)
            if overflow:
                continue

            chunk = data.flatten()
            rms   = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    recorded.clear()
                    print(f"[Listening... RMS={rms:.4f}]")
                recorded.append(chunk)
                silence_start = None

            elif is_speaking:
                recorded.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_SECS:
                    is_speaking   = False
                    silence_start = None
                    audio = np.concatenate(recorded, axis=0)

                    if len(audio) < SAMPLE_RATE * 0.3:
                        print("  (too short, skipped)")
                        continue

                    print("  Transcribing...")
                    segs, info = model.transcribe(
                        audio,
                        beam_size=1,
                        temperature=0.0,
                        language=None,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                        no_speech_threshold=0.6,
                    )
                    text = " ".join(s.text.strip() for s in segs).strip()
                    prob = f"{info.language_probability:.0%}"

                    if text:
                        print(f"\n  RESULT [{info.language} {prob}]: \"{text}\"\n")
                    else:
                        print(f"  (nothing transcribed -- lang_prob={prob})\n")

except KeyboardInterrupt:
    print("\nStopped.")
