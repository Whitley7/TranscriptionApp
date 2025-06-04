# audio/audio_input.py
import sounddevice as sd
import queue
import numpy as np
import time

from config.config import FRAME_DURATION, CHANNELS, AUDIO_FORMAT as CONFIG_AUDIO_FORMAT

class AudioInputManager:
    def __init__(self, sample_rate: int, device_index: int, logger):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.logger = logger
        self.frame_queue = queue.Queue()
        self.last_log_time = 0
        self.log_interval_s = 10

    def record_callback(self, indata, frames, time_info, status):
        current_time = time.time()
        if status:
            self.logger.warning(f"Audio stream status: {status}")

        if current_time - self.last_log_time > self.log_interval_s:
            min_val = np.min(indata)
            max_val = np.max(indata)
            mean_abs_val = np.mean(np.abs(indata.astype(np.float32)))

            self.logger.info(
                f"Raw Audio Input: dtype={indata.dtype}, shape={indata.shape}, "
                f"min={min_val}, max={max_val}, mean_abs={mean_abs_val:.4f}"
            )
            self.last_log_time = current_time

            if indata.dtype == np.int16 and (max_val < 500 and max_val != 0):
                self.logger.warning("Low input level detected! Max amplitude < 500. Check microphone volume.")

        self.frame_queue.put(indata.copy())

    def start_stream(self):
        blocksize = int(FRAME_DURATION * self.sample_rate)

        try:
            device_info = sd.query_devices(self.device_index)
            device_name = device_info['name']
            self.logger.info(f"Selected audio input device: {self.device_index} - {device_name} "
                             f"(Max input channels: {device_info['max_input_channels']})")

            if CHANNELS > device_info['max_input_channels']:
                self.logger.error(f"Configured CHANNELS ({CHANNELS}) exceeds device capability.")
        except Exception as e:
            self.logger.error(f"Could not query device {self.device_index}: {e}", exc_info=True)
            return None

        self.logger.info(f"Initializing audio input stream at {self.sample_rate} Hz "
                         f"with blocksize {blocksize} and dtype {CONFIG_AUDIO_FORMAT}")

        try:
            stream = sd.InputStream(
                device=self.device_index,
                channels=CHANNELS,
                samplerate=self.sample_rate,
                dtype=CONFIG_AUDIO_FORMAT,
                blocksize=blocksize,
                callback=self.record_callback
            )
            self.logger.info("InputStream object successfully created")
            return stream
        except Exception as e:
            self.logger.error(f"Failed to create InputStream: {e}", exc_info=True)
            return None