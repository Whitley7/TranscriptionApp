import time
import os
from typing import Dict, Optional
from faster_whisper import WhisperModel
from core.utils import save_transcript


MODEL_SIZE = "small"
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

def send_to_asr(audio_filepath: str, chunk_id_str: str, session, stats, logger=None):
    """
    Transcribes audio and saves transcript to session.transcript_dir.
    Updates session stats with latency, duration, language.
    """
    if logger:
        logger.info(f"ASR: Transcribing {chunk_id_str}...")

    try:
        start_time = time.time()

        transcript = transcribe_audio(
            audio_path=audio_filepath,
            beam_size=5,
            language=None,
            logger=logger
        )

        latency = time.time() - start_time
        stats.add_latency(latency)

        # Record language and duration stats
        stats.add_detected_language(transcript["language"])
        stats.add_chunk_duration(transcript["duration"])

        if logger:
            logger.info(f"Latency: {latency:.2f}s | Language: {transcript['language']} | Duration: {transcript['duration']:.2f}s")

        transcript_filename = f"{chunk_id_str}.json"
        transcript_path = os.path.join(session.transcript_dir, transcript_filename)
        save_transcript(transcript, transcript_path, logger=logger)

    except Exception as e:
        if logger:
            logger.error(f"ASR failed for {chunk_id_str}: {e}", exc_info=True)