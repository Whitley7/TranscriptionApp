import time
import os
from typing import Dict, Optional
from faster_whisper import WhisperModel
from config.config import CHUNK_DURATION, OVERLAP_DURATION, SAVE_PER_CHUNK_JSON
from core.utils import save_transcript
from core.text_postprocessor import (
    TranscriptBuffer,
    trim_chunk_overlap,
    remove_repeated_words
)

MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "float16"

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe_audio(audio_path: str, beam_size: int = 5, language: Optional[str] = None, logger=None) -> Dict:
    if logger:
        logger.info(f"Transcribing: {audio_path} | beam_size={beam_size} | lang={language or 'auto'}")
    segments, info = model.transcribe(audio_path, beam_size=beam_size, language=language)
    results = {
        "language": info.language,
        "duration": info.duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    }
    if logger:
        logger.info(f"Transcription complete: {audio_path} | Language: {info.language} | Duration: {info.duration:.2f}s")
    return results


def send_to_asr(audio_filepath: str, chunk_id_str: str, chunk_index: int, session, stats, logger=None):
    try:
        start_time = time.time()
        transcript = transcribe_audio(audio_filepath, beam_size=5, language=None, logger=logger)
        latency = time.time() - start_time
        stats.add_latency(latency)
        stats.add_detected_language(transcript["language"])
        stats.add_chunk_duration(transcript["duration"])

        if SAVE_PER_CHUNK_JSON:
            raw_path = os.path.join(session.transcript_dir, f"{chunk_id_str}.json")
            save_transcript(transcript, raw_path, logger=logger)

        segments = transcript["segments"]
        if not segments:
            return

        # Merge segment texts
        merged_text = ". ".join(s["text"] for s in segments).strip()

        # Global time offset
        step_duration = CHUNK_DURATION - OVERLAP_DURATION
        chunk_offset = (chunk_index - 1) * step_duration
        global_start = chunk_offset + segments[0]["start"]

        # Init once
        if not hasattr(session, "dedup_buffer"):
            session.dedup_buffer = TranscriptBuffer()
        if not hasattr(session, "token_history"):
            session.token_history = []
        if not hasattr(session, "paragraph_buffer"):
            session.paragraph_buffer = []
        if not hasattr(session, "paragraph_start_time"):
            session.paragraph_start_time = global_start

        # Deduplicate & clean
        cleaned = session.dedup_buffer.deduplicate(merged_text)
        if not cleaned:
            return

        cleaned = trim_chunk_overlap(session.token_history[-20:], cleaned)
        cleaned = remove_repeated_words(cleaned)

        # Update token history
        session.token_history += cleaned.split()
        session.token_history = session.token_history[-100:]

        # Track start time for paragraph
        if not session.paragraph_buffer:
            session.paragraph_start_time = global_start

        # Append to paragraph buffer
        session.paragraph_buffer.append(cleaned)

        # Build current paragraph
        paragraph = " ".join(session.paragraph_buffer).strip()
        timestamp = session.paragraph_start_time
        final_path = os.path.join(session.transcript_dir, "final_transcript.txt")

        # Overwrite last paragraph line in file
        if os.path.exists(final_path):
            with open(final_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    lines[-1] = f"[{timestamp:.2f}] {paragraph}\n"
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                else:
                    f.write(f"[{timestamp:.2f}] {paragraph}\n")
        else:
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(f"[{timestamp:.2f}] {paragraph}\n")

        if logger:
            logger.info(f"{chunk_id_str} | updated paragraph: {paragraph[:60]}...")

    except Exception as e:
        if logger:
            logger.error(f"ASR failed for {chunk_id_str}: {e}", exc_info=True)
