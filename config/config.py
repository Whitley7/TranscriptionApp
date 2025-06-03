import numpy as np
from session import SESSION_ID
import os

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 2 #seconds
SAVE_DIR = "audio_chunks"
PREFERRED_DEVICE_INDEX = 2
FRAME_DURATION = 0.02
AUDIO_FORMAT = np.int16
OVERLAP_DURATION = 0.25
SILENCE_THRESHOLD = 0.25

SESSION_ROOT = os.path.join("../sessions", SESSION_ID)
SESSION_AUDIO_DIR = os.path.join(SESSION_ROOT, "audio_chunks")
SESSION_LOG_DIR = os.path.join(SESSION_ROOT, "logs")

