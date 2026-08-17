"""
STEP 3 — Raw audio capture to .wav file.
Only run this if step2 prints nothing or garbage.

This records 5 seconds of audio from the mic and saves it as 'diagnose/raw_capture.wav'.
Then:
  1. Open raw_capture.wav in any player (VLC, Windows Media Player, Audacity)
  2. Listen to it — does it sound like YOUR voice, clear and correct speed?
     YES, clear → the mic is fine, Whisper is the problem (sample rate mismatch?)
     Chipmunked / fast → sample rate is wrong (mic runs at 44.1kHz, JARVIS reads at 16kHz)
     Silent / very quiet → wrong mic device index, check step1 and set DEVICE_INDEX

HOW TO USE:
  python diagnose/step3_raw_audio.py
"""
import wave
import numpy as np
import sounddevice as sd

# ── EDIT IF NEEDED ─────────────────────────────────────────────────────────────
DEVICE_INDEX   = -1     # -1 = system default. Set to correct index from step1.
RECORD_SECONDS = 5      # how many seconds to record
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000    # Whisper expects 16 kHz
OUTPUT_FILE = "diagnose/raw_capture.wav"

device_info = sd.query_devices(DEVICE_INDEX if DEVICE_INDEX >= 0 else sd.default.device[0])
native_rate = int(device_info["default_samplerate"])

print("=" * 60)
print("  Raw Audio Capture Diagnostic")
print("=" * 60)
print(f"  Microphone     : [{DEVICE_INDEX}] {device_info['name']}")
print(f"  Native rate    : {native_rate} Hz")
print(f"  Capture rate   : {SAMPLE_RATE} Hz  (Whisper target)")
print(f"  Duration       : {RECORD_SECONDS} seconds")
print(f"  Output file    : {OUTPUT_FILE}")
print("=" * 60)

if native_rate != SAMPLE_RATE:
    print(f"\n!! MISMATCH: Mic native rate ({native_rate} Hz) != target ({SAMPLE_RATE} Hz)")
    print("   sounddevice will resample automatically, but if you hear chipmunk audio,")
    print("   that confirms the mismatch is causing your STT problem.")
else:
    print(f"\nOK Sample rates match ({SAMPLE_RATE} Hz) -- no resampling needed.")

stream_kwargs = dict(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)
if DEVICE_INDEX >= 0:
    stream_kwargs["device"] = DEVICE_INDEX

print(f"\nREC Recording {RECORD_SECONDS} seconds -- SPEAK NOW...\n")
audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), **stream_kwargs)
sd.wait()

# Print RMS so you can see if audio was captured at all
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"   RMS level: {rms:.5f}  ", end="")
if rms < 0.001:
    print("<-- SILENT -- likely wrong device index")
elif rms < 0.01:
    print("<-- Very quiet -- speak louder or move mic closer")
else:
    print("<-- Good signal level")

# Save as 16-bit PCM WAV
audio_int16 = (audio * 32767).astype(np.int16)
with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)   # 16-bit
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())

print(f"\nSaved: {OUTPUT_FILE}")
print("\nNext: Open the file and listen.")
print("  Clear voice     -> Mic OK, Whisper may have config issue")
print("  Chipmunked/fast -> Sample rate mismatch -- set correct DEVICE_INDEX")
print("  Silent          -> Wrong mic index -- go back to step1")
