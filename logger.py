import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")

# Define rotating file handler (5MB max, keep 3 backups)
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s"
))

# Console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for development
    handlers=[file_handler, console_handler]
)

# Expose this logger
logger = logging.getLogger("transcription_app")
