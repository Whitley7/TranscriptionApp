from dataclasses import dataclass

@dataclass
class SessionStats:
    saved_chunks: int = 0
    skipped_chunks: int = 0
