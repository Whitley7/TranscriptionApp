import sounddevice as sd

def list_audio_input_devices():
    devices = sd.query_devices()
    input_devices = [
        (index, device['name'], device['default_samplerate'])
        for index, device in enumerate(devices)
        if device['max_input_channels'] > 0
    ]
    return input_devices

def select_input_device():
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
