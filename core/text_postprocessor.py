from difflib import SequenceMatcher

class TranscriptBuffer:
    """
    Deduplicates transcription output using:
    1. Token overlap (for chunk boundaries)
    2. Fuzzy full-line similarity (for near duplicates across chunks)
    """
    def __init__(self, window_size: int = 7, fuzzy_threshold: float = 0.85, memory_lines: int = 8):
        self.last_tail = []
        self.last_cleaned = ""
        self.window_size = window_size
        self.fuzzy_threshold = fuzzy_threshold
        self.line_history = []
        self.memory_lines = memory_lines

    def deduplicate(self, new_text: str) -> str:
        new_tokens = new_text.strip().split()
        new_line = " ".join(new_tokens).strip()

        # Exact match suppression
        if new_line.lower() == self.last_cleaned.lower():
            return ""

        # Token overlap suppression (prefix match)
        if self.last_tail:
            min_window = min(self.window_size, len(new_tokens), len(self.last_tail))
            if min_window >= 3:
                match_ratio = SequenceMatcher(None, self.last_tail[-min_window:], new_tokens[:min_window]).ratio()
                if match_ratio >= self.fuzzy_threshold:
                    new_tokens = new_tokens[min_window:]
                    new_line = " ".join(new_tokens).strip()

        # Full-line fuzzy match against recent history
        for prev in self.line_history[-self.memory_lines:]:
            sim = SequenceMatcher(None, prev.lower(), new_line.lower()).ratio()
            if sim >= 0.87:  # You can tune this if needed
                return ""  # Skip this line

        # Update buffers
        self.last_tail = (self.last_tail + new_tokens)[-self.window_size:]
        self.last_cleaned = new_line
        self.line_history.append(new_line)
        return new_line

def trim_chunk_overlap(prev_tokens: list[str], new_text: str, min_match: int = 5) -> str:
    new_tokens = new_text.strip().split()
    matcher = SequenceMatcher(None, prev_tokens, new_tokens)
    match = matcher.find_longest_match(0, len(prev_tokens), 0, len(new_tokens))

    if match.size >= min_match and match.b == 0:
        return " ".join(new_tokens[match.size:])
    return new_text

def remove_repeated_words(text: str, max_repeat: int = 2) -> str:
    """
    Removes excessive repetition of the same word in a row.
    Keeps up to `max_repeat` consecutive instances of a word.
    """
    tokens = text.strip().split()
    output = []
    repeat_count = 1

    for i, word in enumerate(tokens):
        if i > 0 and word.lower() == tokens[i - 1].lower():
            repeat_count += 1
            if repeat_count > max_repeat:
                continue
        else:
            repeat_count = 1
        output.append(word)

    return " ".join(output)
