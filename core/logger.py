import logging
import os
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from config.session import SESSION_ID
from config.config import SESSION_LOG_DIR

LOG_PATH = os.path.join(SESSION_LOG_DIR, "session.log")
os.makedirs(SESSION_LOG_DIR, exist_ok=True)

# Custom filter to inject session_id into log records
class SessionFilter(logging.Filter):
    def filter(self, record: object) -> bool | LogRecord:
        record.session_id = SESSION_ID
        return True

# Formatter with session ID
formatter = logging.Formatter(
    "%(asctime)s [%(session_id)s] [%(threadName)s] [%(levelname)s] %(message)s"
)

# File handler
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(formatter)
file_handler.addFilter(SessionFilter())

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.addFilter(SessionFilter())

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Expose app-level logger
logger = logging.getLogger("transcription_app")
