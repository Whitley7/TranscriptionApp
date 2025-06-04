# session_stats.py
from dataclasses import dataclass, field
import threading

@dataclass
class SessionStats:
    saved_chunks: int = 0
    skipped_chunks: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def increment_saved(self):
        with self._lock:
            self.saved_chunks += 1

    def increment_skipped(self):
        with self._lock:
            self.skipped_chunks += 1

    def snapshot(self):
        with self._lock:
            return self.saved_chunks, self.skipped_chunks
