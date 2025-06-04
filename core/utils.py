# core/utils.py
import wave
import os
import numpy as np
import webrtcvad

from config.config import (
    SILENCE_THRESHOLD, CHANNELS, AUDIO_FORMAT as CONFIG_AUDIO_FORMAT,
    FRAME_DURATION, SAMPLE_RATE, VAD_MODE, RMS_PREFILTER_THRESHOLD
)
from typing import Optional, Dict
import json

# Initialize VAD instance using VAD_MODE from config
try:
    _VAD_MODE_CONFIG = VAD_MODE
    if not (0 <= _VAD_MODE_CONFIG <= 3):
        _VAD_MODE_CONFIG = max(0, min(3, _VAD_MODE_CONFIG))
    vad = webrtcvad.Vad(_VAD_MODE_CONFIG)
except Exception:
    vad = webrtcvad.Vad(1)


def is_chunk_speech(audio_chunk_int16: np.ndarray, sample_rate: int, logger) -> bool:
    if audio_chunk_int16.dtype != np.int16:
        logger.warning(f"is_chunk_speech expected np.int16, got {audio_chunk_int16.dtype}. Attempting conversion.")
        try:
            audio_chunk_int16 = audio_chunk_int16.astype(np.int16)
        except ValueError:
            logger.error("Failed to convert audio_chunk to np.int16. Treating as non-speech.")
            return False

    frame_duration_ms = int(FRAME_DURATION * 1000)
    if frame_duration_ms not in [10, 20, 30]:
        logger.warning(f"VAD: Configured FRAME_DURATION ({FRAME_DURATION}s -> {frame_duration_ms}ms) not supported. Using 30ms.")
        frame_duration_ms = 30

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_sample = np.dtype(np.int16).itemsize
    voiced_frames_count = 0
    total_frames_in_chunk = 0

    if len(audio_chunk_int16) < samples_per_frame:
        logger.debug(f"is_chunk_speech: Chunk too short ({len(audio_chunk_int16)} samples)")
        return False

    offset = 0
    while offset + samples_per_frame <= len(audio_chunk_int16):
        frame = audio_chunk_int16[offset: offset + samples_per_frame]
        offset += samples_per_frame
        total_frames_in_chunk += 1
        frame_bytes = frame.tobytes()

        expected_byte_len = samples_per_frame * bytes_per_sample
        actual_byte_len = len(frame_bytes)
        if actual_byte_len != expected_byte_len:
            logger.error(f"Frame byte length mismatch for VAD: expected {expected_byte_len}, got {actual_byte_len}")
            continue

        try:
            is_speech_result = vad.is_speech(frame_bytes, sample_rate)
            if is_speech_result:
                voiced_frames_count += 1
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            continue

    if total_frames_in_chunk == 0:
        logger.debug("is_chunk_speech: No frames were processed.")
        return False

    ratio_voiced = voiced_frames_count / total_frames_in_chunk
    speech_detected = ratio_voiced >= SILENCE_THRESHOLD

    logger.debug(f"VAD result: {'Speech' if speech_detected else 'Silence'} "
                 f"(Voiced: {voiced_frames_count}/{total_frames_in_chunk}, "
                 f"Ratio: {ratio_voiced:.2f}, Threshold: {SILENCE_THRESHOLD})")

    return speech_detected


def process_audio_chunk_for_speech(audio_chunk_int16: np.ndarray, sample_rate: int, chunk_id_str: str, logger) -> Optional[np.ndarray]:
    if audio_chunk_int16.dtype != np.int16:
        logger.error(f"ProcessChunk ({chunk_id_str}): Expected np.int16, got {audio_chunk_int16.dtype}. Skipping.")
        return None

    if len(audio_chunk_int16) == 0:
        logger.debug(f"ProcessChunk ({chunk_id_str}): Received empty audio chunk. Skipping.")
        return None

    audio_float_for_rms = audio_chunk_int16.astype(np.float32) / 32768.0
    rms_energy = np.sqrt(np.mean(audio_float_for_rms ** 2))

    logger.debug(f"ProcessChunk ({chunk_id_str}): RMS energy = {rms_energy:.6f} (Threshold: {RMS_PREFILTER_THRESHOLD})")

    if rms_energy < RMS_PREFILTER_THRESHOLD:
        log_chunk_info(chunk_id_str, rms_energy, len(audio_chunk_int16) / sample_rate, skipped=True,
                       reason=f"RMS ({rms_energy:.6f}) < threshold ({RMS_PREFILTER_THRESHOLD})", logger=logger)
        return None

    try:
        if is_chunk_speech(audio_chunk_int16, sample_rate, logger):
            logger.debug(f"ProcessChunk ({chunk_id_str}): Speech DETECTED by VAD.")
            return audio_chunk_int16
        else:
            log_chunk_info(chunk_id_str, rms_energy, len(audio_chunk_int16) / sample_rate, skipped=True,
                           reason="No speech detected by VAD", logger=logger)
            return None
    except Exception as e:
        logger.error(f"ProcessChunk ({chunk_id_str}): Error during VAD. {e}", exc_info=True)
        log_chunk_info(chunk_id_str, rms_energy, len(audio_chunk_int16) / sample_rate, skipped=True,
                       reason=f"Error during VAD: {e}", logger=logger)
        return None


def save_wav(chunk_int16: np.ndarray, filename: str, sample_rate: int, logger):
    if chunk_int16.dtype != np.int16:
        logger.error(f"save_wav expects np.int16 data, received {chunk_int16.dtype}. Attempting cast.")
        chunk_int16 = chunk_int16.astype(np.int16)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(np.int16).itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk_int16.tobytes())
        logger.debug(f"WAV saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving WAV {filename}: {e}", exc_info=True)


def log_chunk_info(chunk_identifier: str, volume_metric: float, duration: float, skipped: bool = False,
                   reason: Optional[str] = None, logger=None):
    if skipped:
        reason_str = f" | Reason: {reason}" if reason else ""
        logger.info(f"Skipped chunk {chunk_identifier} | Vol: {volume_metric:.4f} | Duration: {duration:.2f}s{reason_str}")
    else:
        logger.info(f"Processed chunk {chunk_identifier} | Vol: {volume_metric:.4f} | Duration: {duration:.2f}s")


def save_transcript(transcript: Dict, output_path: str, logger=None):
    """
    Saves transcript to a JSON file.

    Args:
        transcript (dict): The transcript object.
        output_path (str): Path to save the JSON file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        if logger:
            logger.info(f"Transcript saved: {output_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save transcript to {output_path}: {e}", exc_info=True)
