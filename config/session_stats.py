from dataclasses import dataclass, field
import threading
import statistics
from collections import Counter

@dataclass
class SessionStats:
    saved_chunks: int = 0
    skipped_chunks: int = 0
    transcription_latencies: list = field(default_factory=list)
    chunk_durations: list = field(default_factory=list)
    skip_reasons: Counter = field(default_factory=Counter)
    detected_languages: Counter = field(default_factory=Counter)
    first_latency_recorded: bool = False
    first_latency_value: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def increment_saved(self):
        with self._lock:
            self.saved_chunks += 1

    def increment_skipped(self, reason: str = "unknown"):
        with self._lock:
            self.skipped_chunks += 1
            self.skip_reasons[reason] += 1

    def add_latency(self, value: float):
        with self._lock:
            self.transcription_latencies.append(value)
            if not self.first_latency_recorded:
                self.first_latency_value = value
                self.first_latency_recorded = True

    def add_chunk_duration(self, value: float):
        with self._lock:
            self.chunk_durations.append(value)

    def add_detected_language(self, lang: str):
        with self._lock:
            self.detected_languages[lang] += 1

    def latency_summary(self):
        with self._lock:
            if not self.transcription_latencies:
                return 0.0, 0.0, 0.0, 0.0
            avg = sum(self.transcription_latencies) / len(self.transcription_latencies)
            min_l = min(self.transcription_latencies)
            max_l = max(self.transcription_latencies)
            std_l = statistics.stdev(self.transcription_latencies) if len(self.transcription_latencies) > 1 else 0.0
            return avg, min_l, max_l, std_l

    def average_chunk_duration(self):
        with self._lock:
            if self.chunk_durations:
                return sum(self.chunk_durations) / len(self.chunk_durations)
            return 0.0

    def most_common_language(self):
        with self._lock:
            if self.detected_languages:
                return self.detected_languages.most_common(1)[0]
            return None, 0
