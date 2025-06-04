# main.py
import os
import threading
import time
import queue
from datetime import datetime

from audio.input_device import select_input_device
from audio.audio_input import AudioInputManager
from audio.chunk_processor import chunk_processor
from core.logger import setup_logger
from config.session_stats import SessionStats
from config.session import SessionManager
from config.config import SAMPLE_RATE as CONFIG_APP_SAMPLE_RATE
from transcriber import send_to_asr


# Shutdown flag
shutdown_event = threading.Event()
transcription_queue = queue.Queue()

def transcriber_worker(transcription_queue, session, stats, shutdown_event, logger):
    """
    Background worker that receives audio chunk filepaths,
    transcribes them, and handles paragraph-mode writing with final flush.
    """
    while not shutdown_event.is_set():
        try:
            filepath, chunk_id, chunk_index = transcription_queue.get(timeout=1)
            send_to_asr(filepath, chunk_id, chunk_index, session, stats, logger)
        except queue.Empty:
            continue
        except Exception as e:
            if logger:
                logger.error(f"Transcriber worker error: {e}", exc_info=True)

    # Fallback: Final paragraph flush at shutdown
    if hasattr(session, "paragraph_buffer") and session.paragraph_buffer:
        final_text = " ".join(session.paragraph_buffer).strip()
        timestamp = getattr(session, "paragraph_start_time", 0.0)
        final_line = f"[{timestamp:.2f}] {final_text}"
        final_path = os.path.join(session.transcript_dir, "final_transcript.txt")

        try:
            with open(final_path, "a", encoding="utf-8") as f:
                f.write(final_line + "\n\n")
            if logger:
                logger.info("Final paragraph flushed at shutdown.")
        except Exception as e:
            if logger:
                logger.error(f"Failed to flush final paragraph: {e}", exc_info=True)

def main():
    """
    Main entry point of the application. Handles device selection,
    launches audio input stream and background chunk processing thread.
    """
    session = SessionManager()
    logger = setup_logger(session.session_id, session.log_dir)

    logger.info(f"Session ID: {session.session_id}")
    logger.info(f"Audio directory: {session.audio_dir}")
    logger.info(f"Log directory: {session.log_dir}")

    device_index, device_native_sample_rate = select_input_device()
    app_sample_rate = CONFIG_APP_SAMPLE_RATE

    logger.info(f"Selected device index: {device_index} (Native SR: {device_native_sample_rate} Hz). "
                f"Application will attempt to use configured sample rate: {app_sample_rate} Hz.")

    audio_manager = AudioInputManager(app_sample_rate, device_index, logger)
    stats = SessionStats()
    session_start = datetime.now()

    # Start chunk processor thread
    processor_thread = threading.Thread(
        target=chunk_processor,
        args=(app_sample_rate, shutdown_event, stats, audio_manager, session, logger, transcription_queue),
        daemon=True,
        name="ChunkProcessor",
    )
    processor_thread.start()
    logger.info("Chunk processor thread started.")

    # Start transcriber thread
    transcriber_thread = threading.Thread(
        target=transcriber_worker,
        args=(transcription_queue, session, stats, shutdown_event, logger),
        daemon=True,
        name="TranscriberWorker",
    )
    transcriber_thread.start()
    logger.info("Transcriber thread started.")

    stream = None
    try:
        stream = audio_manager.start_stream()
        if stream is None:
            logger.error("audio_manager.start_stream() returned None — stream creation failed.")
            return
        else:
            logger.info(f"Audio input stream object created: {stream}")

        try:
            stream.start()
            logger.info("Audio input stream running. Press Ctrl+C to stop.")
        except Exception as e:
            logger.error(f"Failed to start audio input stream: {e}", exc_info=True)
            return

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

        if transcriber_thread.is_alive():
            logger.info("Waiting for transcriber thread to join...")
            transcriber_thread.join(timeout=5.0)
            if transcriber_thread.is_alive():
                logger.warning("Transcriber thread did not join in time.")
            else:
                logger.info("Transcriber thread joined.")

        logger.info("All threads shut down cleanly.")

        session_end = datetime.now()
        duration = session_end - session_start

        avg_latency, min_latency, max_latency, stddev_latency = stats.latency_summary()
        avg_duration = stats.average_chunk_duration()
        most_lang, most_lang_count = stats.most_common_language()
        skip_reasons_str = ", ".join(f"{k}: {v}" for k, v in stats.skip_reasons.items())

        logger.info("===== SESSION SUMMARY =====")
        logger.info(f"Session ID: {session.session_id}")
        logger.info(f"Started at: {session_start}")
        logger.info(f"Ended at:   {session_end}")
        logger.info(f"Duration:   {duration}")
        logger.info(f"Chunks saved: {stats.saved_chunks}")
        logger.info(f"Chunks skipped: {stats.skipped_chunks} ({skip_reasons_str})")
        logger.info(
            f"Average latency: {avg_latency:.2f}s (min: {min_latency:.2f}s, max: {max_latency:.2f}s, stddev: {stddev_latency:.2f}s)")
        logger.info(f"Avg chunk duration: {avg_duration:.2f}s")
        logger.info(f"First transcription latency: {stats.first_latency_value:.2f}s")
        logger.info(f"Most detected language: {most_lang} ({most_lang_count})")
        logger.info("=============================")

if __name__ == "__main__":
    main()
