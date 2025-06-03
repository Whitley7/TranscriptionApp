from datetime import datetime
import os

SESSION_ID = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Full root folder for this session
SESSION_ROOT = os.path.join("../sessions", SESSION_ID)
SESSION_AUDIO_DIR = os.path.join(SESSION_ROOT, "audio_chunks")
SESSION_LOG_DIR = os.path.join(SESSION_ROOT, "logs")

# Ensure folders exist at import time
os.makedirs(SESSION_AUDIO_DIR, exist_ok=True)
os.makedirs(SESSION_LOG_DIR, exist_ok=True)
