import os
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_root = os.path.join("sessions", self.session_id)
        self.audio_dir = os.path.join(self.session_root, "audio_chunks")
        self.log_dir = os.path.join(self.session_root, "logs")

        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
