# config.py
import numpy as np


# Audio capture and processing configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 2  # seconds
FRAME_DURATION = 0.03  # seconds (30ms for VAD)
OVERLAP_DURATION = 0.25  # seconds
PREFERRED_DEVICE_INDEX = 2
AUDIO_FORMAT = np.int16

# VAD & silence handling
VAD_MODE = 1  # Aggressiveness: 0 (most sensitive) to 3 (least)
SILENCE_THRESHOLD = 0.25  # Ratio of voiced frames
RMS_PREFILTER_THRESHOLD = 0.003  # RMS cutoff for float audio
MIN_SILENCE_TO_LOG_S = 5.0  # Minimum silence duration to log a resume
