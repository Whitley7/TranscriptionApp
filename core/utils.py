import wave
import os
from core.logger import logger
from typing import Optional


def normalize_audio(chunk: np.ndarray) -> np.ndarray:
    """
    Normalize an audio chunk to the range [-1.0, 1.0].

    Args:
        chunk (np.ndarray): Raw audio chunk as integer samples.

    Returns:
        np.ndarray: Normalized audio as float32.
    """
    peak = np.max(np.abs(chunk))
    if peak > 0:
        return chunk.astype(np.float32) / peak
    return chunk.astype(np.float32)


def process_audio_chunk(chunk: np.ndarray, filename: str, duration: float, sample_rate: int) -> Optional[np.ndarray]:
    """
    Processes an audio chunk: filters silence, normalizes, logs, and returns processed chunk.

    Returns:
        np.ndarray or None: normalized chunk if not silent, else None.
    """
    volume = np.mean(np.abs(chunk))

    if volume < SILENCE_THRESHOLD:
        log_chunk_info(filename=filename, volume=volume, duration=duration, skipped=True, reason="Below silence threshold")
        return None

    normalized = normalize_audio(chunk)
    scaled_chunk = (normalized * 32767).astype(np.int16)

    log_chunk_info(filename=filename, volume=volume, duration=duration)
    return scaled_chunk



def save_wav(chunk: np.ndarray, filename: str, sample_rate: int):
    """
    Save an audio chunk as a WAV file.

    Args:
        chunk (np.ndarray): Audio data to save.
        filename (str): Output file path.
        sample_rate (int): Sample rate in Hz.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(AUDIO_FORMAT).itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk.astype(AUDIO_FORMAT).tobytes())
        logger.debug(f"✅ WAV saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving WAV: {e}")


def log_chunk_info(filename: str, volume: float, duration: float, skipped: bool = False, reason: Optional[str] = None):
    """
    Logs information about processed audio chunks.

    Args:
        filename (str): Path to the audio file or chunk ID.
        volume (float): Average absolute volume of the chunk.
        duration (float): Duration in seconds.
        skipped (bool): Whether the chunk was skipped (not saved).
        reason (str, optional): Reason for skipping, if applicable.

    Returns:
        None
    """
    if skipped:
        reason_str = f" | Reason: {reason}" if reason else ""
        logger.info(f"Skipped chunk | Volume: {volume:.4f} | Duration: {duration:.2f}s{reason_str}")
    else:
        logger.info(f"Saved chunk: {filename} | Volume: {volume:.4f} | Duration: {duration:.2f}s")
