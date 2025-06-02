import sounddevice as sd
import numpy as np
import wave
import time
import os
from datetime import datetime
from input_device_selection import select_input_device
from config import *

os.makedirs(SAVE_DIR, exist_ok=True)

AUDIO_FORMAT = np.int16

#Buffer to accumulate small blocks until full chunk is ready
audio_buffer = []

def record_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print("Recording status:", status)

    audio_buffer.append(indata.copy())

    #Check if total buffer duration reaches CHUNK_DURATION
    total_samples = sum(block.shape[0] for block in audio_buffer)
    if total_samples >= int(sample_rate * CHUNK_DURATION):
        #Concatenate blocks and cut to exactly CHUNK_DURATION
        full_chunk = np.concatenate(audio_buffer, axis=0)
        chunk_data = full_chunk[:int(sample_rate * CHUNK_DURATION)]
        del audio_buffer[:]
        process_and_save_chunk(chunk_data, sample_rate)

def record_chunks(device_index, sr):
    global sample_rate
    sample_rate = sr  #make sample_rate accessible in callback

    with sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=sample_rate,
        dtype=AUDIO_FORMAT,
        callback=record_callback
    ):
        print("Recording started...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nRecording stopped.")

def process_and_save_chunk(chunk, sample_rate):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(SAVE_DIR, f"chunk_{timestamp}.wav")
    save_chunk(chunk, filename, sample_rate)

def save_chunk(audio_data, filename, sample_rate):
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(AUDIO_FORMAT).itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.astype(AUDIO_FORMAT).tobytes())
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Error saving chunk: {e}")

#Implementation test
if __name__ == "__main__":
    device_index, sample_rate = select_input_device()
    record_chunks(device_index, sample_rate)
