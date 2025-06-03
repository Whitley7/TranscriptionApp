import threading
import time
from audio.input_device import select_input_device
from audio.audio_input import start_input_stream
from audio.chunk_processor import chunk_processor
from core.logger import logger
from config.session_stats import SessionStats
from datetime import datetime
from config.session import SESSION_ID

import os

#Shutdown flag
shutdown_event = threading.Event()

def setup_session_folders():
    os.makedirs(SESSION_AUDIO_DIR, exist_ok=True)
    os.makedirs(SESSION_LOG_DIR, exist_ok=True)

def main():
    """
    Main entry point of the application. Handles device selection,
    launches audio input stream and background chunk processing thread.
    """
    setup_session_folders()

    device_index, sample_rate = select_input_device()
    logger.info(f"Selected device index {device_index} | Sample rate: {sample_rate} Hz")

    #Start chunk processor thread with shutdown_event
    stats = SessionStats()
    session_start = datetime.now()
    processor_thread = threading.Thread(
        target=chunk_processor,
        args=(sample_rate, shutdown_event, stats),
        daemon=True,
        name="ChunkProcessor"
    )
    processor_thread.start()
    logger.info("Chunk processor thread started.")

    try:
        with start_input_stream(device_index, sample_rate):
            logger.info("Audio input stream started. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Signaling shutdown...")
        shutdown_event.set()
        processor_thread.join()
        logger.info("All threads shut down cleanly.")

        session_end = datetime.now()
        duration = session_end - session_start

        logger.info("===== SESSION SUMMARY =====")
        logger.info(f"Session ID: {SESSION_ID}")
        logger.info(f"Started at: {session_start}")
        logger.info(f"Ended at:   {session_end}")
        logger.info(f"Duration:   {duration}")
        logger.info(f"Chunks saved: {stats.saved_chunks}")
        logger.info(f"Chunks skipped: {stats.skipped_chunks}")
        logger.info("=============================")

    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)



if __name__ == "__main__":
    main()
