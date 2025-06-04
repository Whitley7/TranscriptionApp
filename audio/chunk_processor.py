from itertools import count as counter
import numpy as np
import time
import os
import queue
import threading

from config.config import (
    CHUNK_DURATION, OVERLAP_DURATION, SAMPLE_RATE as CONFIG_SAMPLE_RATE,
    MIN_SILENCE_TO_LOG_S
)
from core.utils import process_audio_chunk_for_speech, save_wav


def chunk_processor(
    sample_rate: int,
    shutdown_event: threading.Event,
    stats,
    audio_input_manager,
    session,
    logger,
    transcription_queue: queue.Queue
):
    if sample_rate != CONFIG_SAMPLE_RATE:
        logger.warning(f"Chunk processor started with sample_rate {sample_rate}Hz, "
                       f"but config SAMPLE_RATE is {CONFIG_SAMPLE_RATE}Hz. Using {sample_rate}Hz.")

    buffer = []
    chunk_size_samples = int(CHUNK_DURATION * sample_rate)
    overlap_size_samples = int(OVERLAP_DURATION * sample_rate)
    effective_new_audio_per_step_s = CHUNK_DURATION - OVERLAP_DURATION

    if effective_new_audio_per_step_s <= 0:
        logger.error(f"CHUNK_DURATION ({CHUNK_DURATION}s) must be > OVERLAP_DURATION ({OVERLAP_DURATION}s).")
        effective_new_audio_per_step_s = CHUNK_DURATION

    chunk_num_generator = counter(start=1)
    cumulative_silent_s = 0.0

    logger.info("Chunk processor thread started. Waiting for audio frames...")

    while not shutdown_event.is_set():
        try:
            frame = audio_input_manager.frame_queue.get(timeout=0.5)
            buffer.append(frame)
            total_samples = sum(f.shape[0] for f in buffer)

            if total_samples >= chunk_size_samples:
                concatenated_audio = np.concatenate(buffer, axis=0)
                current_chunk = concatenated_audio[:chunk_size_samples].copy()

                if 0 < overlap_size_samples <= len(current_chunk):
                    tail = current_chunk[-overlap_size_samples:].copy()
                    remaining = concatenated_audio[chunk_size_samples:]
                    buffer = [tail] + ([remaining] if len(remaining) > 0 else [])
                else:
                    buffer = [concatenated_audio[chunk_size_samples:]] if total_samples > chunk_size_samples else []

                chunk_id = next(chunk_num_generator)
                chunk_id_str = f"chunk_{chunk_id:04d}"

                speech_audio = process_audio_chunk_for_speech(current_chunk, sample_rate, chunk_id_str, logger)

                if speech_audio is not None:
                    if cumulative_silent_s >= MIN_SILENCE_TO_LOG_S:
                        logger.info(f"Speech resumed ({chunk_id_str}) after ~{cumulative_silent_s:.1f}s of silence.")
                    cumulative_silent_s = 0.0

                    os.makedirs(session.audio_dir, exist_ok=True)
                    filename = os.path.join(session.audio_dir, f"{chunk_id_str}.wav")
                    save_wav(speech_audio, filename, sample_rate, logger)
                    stats.increment_saved()

                    chunk_duration = len(speech_audio) / sample_rate
                    stats.add_chunk_duration(chunk_duration)

                    logger.info(f"Saved speech chunk: {filename} | Duration: {chunk_duration:.2f}s")

                    # 👇 Now passes chunk_index along with the rest
                    transcription_queue.put((filename, chunk_id_str, chunk_id))

                else:
                    cumulative_silent_s += effective_new_audio_per_step_s
                    stats.increment_skipped(reason="vad")
                    logger.debug(f"Cumulative silence now approx: {cumulative_silent_s:.1f}s")

        except queue.Empty:
            logger.debug("Audio frame queue empty after timeout.")
            continue
        except Exception as e:
            logger.error(f"Error in chunk processor loop: {e}", exc_info=True)
            time.sleep(0.1)

    logger.info("Chunk processor thread gracefully shut down.")
