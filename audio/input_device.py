import sounddevice as sd
from config.config import PREFERRED_DEVICE_INDEX

def list_audio_input_devices():
    devices = sd.query_devices()
    input_devices = [
        (index, device['name'], device['default_samplerate'])
        for index, device in enumerate(devices)
        if device['max_input_channels'] > 0
    ]
    return input_devices

def select_input_device():

    if PREFERRED_DEVICE_INDEX is not None:
        try:
            device_info = sd.query_devices(PREFERRED_DEVICE_INDEX)
            return PREFERRED_DEVICE_INDEX, int(device_info['default_samplerate'])
        except Exception as e:
            print(f"⚠️ Could not use preferred device index ({PREFERRED_DEVICE_INDEX}): {e}")


    input_devices = list_audio_input_devices()

    if not input_devices:
        raise RuntimeError("No audio input devices found.")
    print("\nAvailable audio input devices:")

    for idx, name, rate in input_devices:
        print(f"[{idx}] {name} — Default sample rate: {int(rate)} Hz")

    while True:
        try:
            selection = int(input("\nEnter the index of the input device you want to use: "))
            for idx, _, rate in input_devices:
                if idx == selection:
                    return idx, int(rate)
            print("Invalid index. Please select one from the list.")
        except ValueError:
            print("Please enter a valid integer.")
