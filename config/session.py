import os
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_root = os.path.join(self.project_root, "sessions", self.session_id)

        self.audio_dir = os.path.join(self.session_root, "audio_chunks")
        self.log_dir = os.path.join(self.session_root, "logs")
        self.transcript_dir = os.path.join(self.session_root, "transcripts")

        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)
