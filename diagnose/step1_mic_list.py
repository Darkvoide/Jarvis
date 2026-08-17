"""
STEP 1 — List every microphone device on this machine.
Run this first. It prints the index and name of every input device.
Then check: is your real microphone listed? What is its index?
"""
import sounddevice as sd

print("=" * 60)
print("  Microphone Device List")
print("=" * 60)

devices = sd.query_devices()
default_in = sd.default.device[0]

for idx, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        marker = "  << DEFAULT" if idx == default_in else ""
        print(f"  [{idx:2d}] {dev['name']}{marker}")
        print(f"        channels={dev['max_input_channels']}  sample_rate={int(dev['default_samplerate'])} Hz")

print("=" * 60)
print(f"\nDefault input device index: {default_in}")
print("\nIf your mic is NOT the default, note its index.")
print("Set  input.camera_index  in config.yaml, or edit DEVICE_INDEX below in step2.")
