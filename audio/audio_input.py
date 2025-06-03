import sounddevice as sd
import queue

from config.config import FRAME_DURATION, CHANNELS, AUDIO_FORMAT
from core.logger import logger

frame_queue = queue.Queue()

def record_callback(indata, frames, time_info, status):
    """
    Callback function triggered by sounddevice.InputStream.

    Called each time a new audio block is available. It places the block
    into the shared frame_queue for asynchronous processing.

    Args:
        indata (np.ndarray): Incoming audio data (shape: [blocksize, channels]).
        frames (int): Number of frames in this block (not used here).
        time_info (dict): Dictionary containing timing information.
        status (CallbackFlags): Indicates buffer overflows, underflows, etc.

    Returns:
        None
    """
    if status:
        logger.warning(f"Stream status warning: {status}")
    frame_queue.put(indata.copy())


def start_input_stream(device_index, sample_rate):
    """
    Initializes and returns a sounddevice.InputStream for audio input.

    Args:
        device_index (int): Index of the input audio device to use.
        sample_rate (int): Sample rate in Hz for the stream.

    Returns:
        sounddevice.InputStream: Configured stream (not started yet).
    """
    blocksize = int(FRAME_DURATION * sample_rate)  # e.g., 0.02 * 16000 = 320 samples

    # Get human-readable device name
    try:
        device_name = sd.query_devices(device_index)['name']
    except Exception:
        device_name = "Unknown"

    logger.info(f"Initialized audio input stream | Device: {device_index} ({device_name}) | "
                f"Sample rate: {sample_rate} Hz | Block size: {blocksize} samples")

    return sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=sample_rate,
        dtype=AUDIO_FORMAT,
        blocksize=blocksize,
        callback=record_callback
    )
