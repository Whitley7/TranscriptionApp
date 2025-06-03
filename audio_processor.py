import sounddevice as sd
import numpy as np
import wave
import time
import os
from datetime import datetime
from input_device_selection import select_input_device
from config import *

# Create save directory if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

AUDIO_FORMAT = np.int16

# Advanced parameters
FRAME_DURATION = 0.02         # 20 ms frames
OVERLAP_DURATION = 0.5        # seconds of overlap between chunks

# Runtime buffers
frame_queue = []              # holds short frames
sample_rate = None


def record_callback(indata, frames, time_info, status):
    if status:
        print("Recording status:", status)
    frame_queue.append(indata.copy())


def record_chunks(device_index, sr):
    global sample_rate
    sample_rate = sr
    chunk_size = int(CHUNK_DURATION * sample_rate)
    frame_size = int(FRAME_DURATION * sample_rate)
    overlap_size = int(OVERLAP_DURATION * sample_rate)

    print(f"🎙️ Recording started with {CHUNK_DURATION}s chunks and {OVERLAP_DURATION}s overlap...")

    with sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=sample_rate,
        dtype=AUDIO_FORMAT,
        blocksize=frame_size,
        callback=record_callback
    ):
        buffer = []
        try:
            while True:
                if not frame_queue:
                    time.sleep(0.01)
                    continue

                frame = frame_queue.pop(0)
                buffer.append(frame)

                total_samples = sum(f.shape[0] for f in buffer)
                if total_samples >= chunk_size:
                    # Join all frames and crop to chunk_size
                    full_chunk = np.concatenate(buffer, axis=0)[:chunk_size]

                    # Extract the overlap tail
                    tail = full_chunk[-overlap_size:].copy()

                    # Process and save chunk
                    process_and_save_chunk(full_chunk, sample_rate)

                    # Reset buffer with tail (for overlap)
                    buffer = [tail]

        except KeyboardInterrupt:
            print("\n⏹️ Recording stopped.")


def save_chunk(audio_data, filename, sample_rate):
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(AUDIO_FORMAT).itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.astype(AUDIO_FORMAT).tobytes())
        print(f"💾 Saved: {filename}")
    except Exception as e:
        print(f"❌ Error saving chunk: {e}")


def process_and_save_chunk(chunk, sample_rate):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(SAVE_DIR, f"chunk_{timestamp}.wav")
    save_chunk(chunk, filename, sample_rate)


# --- Run Block ---
if __name__ == "__main__":
    device_index, sample_rate = select_input_device()
    record_chunks(device_index, sample_rate)
