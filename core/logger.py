# core/logger.py
import logging
import os
from logging import LogRecord
from logging.handlers import RotatingFileHandler


def setup_logger(session_id: str, log_dir: str) -> logging.Logger:
    """
    Sets up a session-specific logger with rotating file and console output.

    Args:
        session_id (str): Unique session identifier
        log_dir (str): Path to the log directory

    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "session.log")

    # Custom filter to inject session ID into each log record
    class SessionFilter(logging.Filter):
        def filter(self, record: LogRecord) -> bool:
            record.session_id = session_id
            return True

    formatter = logging.Formatter(
        "%(asctime)s [%(session_id)s] [%(threadName)s] [%(levelname)s] %(message)s"
    )

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SessionFilter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SessionFilter())

    logger = logging.getLogger("transcription_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Avoid duplicate handlers if reinitialized
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
