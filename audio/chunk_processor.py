from itertools import count as counter
import numpy as np
import time
from audio_input import frame_queue
from core.utils import *
import os
import queue

def chunk_processor(sample_rate: int, shutdown_event, stats):
    """
    Thread function that accumulates incoming frames from the audio queue,
    assembles them into overlapping chunks, and saves them to disk.

    Args:
        sample_rate (int): The audio sample rate in Hz.
        shutdown_event (threading.Event): Flag to signal graceful shutdown.
        :param stats:
    """
    buffer = []
    chunk_size = int(CHUNK_DURATION * sample_rate)
    overlap_size = int(OVERLAP_DURATION * sample_rate)
    chunk_counter = counter()

    while not shutdown_event.is_set():
        try:
            # Use timeout to allow periodic shutdown checks
            frame = frame_queue.get(timeout=0.5)

            buffer.append(frame)

            total_samples = sum(f.shape[0] for f in buffer)
            if total_samples >= chunk_size:
                full_chunk = np.concatenate(buffer, axis=0)[:chunk_size]
                tail = full_chunk[-overlap_size:].copy()

                duration = len(full_chunk) / sample_rate
                chunk_id = next(chunk_counter)
                filename = os.path.join(SESSION_AUDIO_DIR, f"chunk_{chunk_id:03}.wav")

                processed = process_audio_chunk(full_chunk, filename, duration, sample_rate)
                if processed is not None:
                    save_wav(processed, filename, sample_rate)
                    stats.saved_chunks += 1
                else:
                    stats.skipped_chunks += 1

                buffer = [tail]

        except queue.Empty:
            continue

        except Exception as e:
            from core.logger import logger
            logger.error(f"Error in chunk processor: {e}", exc_info=True)
            time.sleep(0.1)
