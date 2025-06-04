# main.py
import threading
import time
from datetime import datetime
from audio.input_device import select_input_device
from audio.audio_input import AudioInputManager
from audio.chunk_processor import chunk_processor
from core.logger import setup_logger
from config.session_stats import SessionStats
from config.session import SessionManager
from config.config import SAMPLE_RATE as CONFIG_APP_SAMPLE_RATE

# Shutdown flag
shutdown_event = threading.Event()


def main():
    """
    Main entry point of the application. Handles device selection,
    launches audio input stream and background chunk processing thread.
    """
    session = SessionManager()  # Initialize session (creates dirs + ID)
    logger = setup_logger(session.session_id, session.log_dir)  # Initialize logger for this session

    logger.info(f"Session ID: {session.session_id}")
    logger.info(f"Audio directory: {session.audio_dir}")
    logger.info(f"Log directory: {session.log_dir}")

    # Device selection
    device_index, device_native_sample_rate = select_input_device()
    app_sample_rate = CONFIG_APP_SAMPLE_RATE

    logger.info(f"Selected device index: {device_index} (Native SR: {device_native_sample_rate} Hz). "
                f"Application will attempt to use configured sample rate: {app_sample_rate} Hz.")

    # Start audio input manager and session statistics
    audio_manager = AudioInputManager(app_sample_rate, device_index, logger)
    stats = SessionStats()
    session_start = datetime.now()

    # Launch background chunk processor
    processor_thread = threading.Thread(
        target=chunk_processor,
        args=(app_sample_rate, shutdown_event, stats, audio_manager, session, logger),  # Pass session
        daemon=True,
        name="ChunkProcessor",
    )
    processor_thread.start()
    logger.info("Chunk processor thread started.")

    stream = None
    try:
        stream = audio_manager.start_stream()

        if stream.samplerate != app_sample_rate:
            logger.warning(f"Audio stream started with actual sample rate {stream.samplerate} Hz, "
                           f"though {app_sample_rate} Hz was requested.")
        else:
            logger.info(f"Audio input stream successfully started at {stream.samplerate} Hz.")

        stream.start()
        logger.info("Audio input stream running. Press Ctrl+C to stop.")

        while not shutdown_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Signaling shutdown...")
    except Exception as e:
        logger.error(f"Error during audio stream setup or main loop: {e}", exc_info=True)
    finally:
        logger.info("Initiating shutdown sequence...")
        shutdown_event.set()

        if stream is not None:
            try:
                logger.info("Stopping audio stream...")
                stream.stop()
                stream.close()
                logger.info("Audio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping/closing audio stream: {e}", exc_info=True)

        if processor_thread.is_alive():
            logger.info("Waiting for chunk processor thread to join...")
            processor_thread.join(timeout=5.0)
            if processor_thread.is_alive():
                logger.warning("Chunk processor thread did not join in time.")
            else:
                logger.info("Chunk processor thread joined.")

        logger.info("All threads shut down cleanly.")
        session_end = datetime.now()
        duration = session_end - session_start

        logger.info("===== SESSION SUMMARY =====")
        logger.info(f"Session ID: {session.session_id}")
        logger.info(f"Started at: {session_start}")
        logger.info(f"Ended at:   {session_end}")
        logger.info(f"Duration:   {duration}")
        logger.info(f"Chunks saved: {stats.saved_chunks}")
        logger.info(f"Chunks skipped: {stats.skipped_chunks}")
        logger.info("=============================")


if __name__ == "__main__":
    main()
