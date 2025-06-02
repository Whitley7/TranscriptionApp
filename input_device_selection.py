import sounddevice as sd

#Return a list of tuples: (index, name) for input devices only
def list_audio_input_devices():
    devices = sd.query_devices()
    input_devices = [
        (index, device['name'])
        for index, device in enumerate(devices)
        if device['max_input_channels'] > 0
    ]
    return input_devices

#Display input devices and return the selected device index
def select_input_device():
    input_devices = list_audio_input_devices()

    if not input_devices:
        raise RuntimeError("No audio input devices found")

    print("\nAvailable audio input devices:")
    for idx, name in input_devices:
        print(f"[{idx}] {name}")

    while True:
        try:
            selection = int(input("\nEnter the index of the input device you want to use: "))
            if any(idx == selection for idx, _ in input_devices):
                return selection
            else:
                print("Invalid index. Please select one from the list")
        except ValueError:
            print("Please enter a valid integer")