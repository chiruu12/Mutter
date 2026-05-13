import tempfile

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


class Recorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._buffer: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=2048,
        )
        self._stream.start()

    def _callback(self, indata, frames, time, status) -> None:
        self._buffer.append(indata.copy())

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return np.concatenate(self._buffer) if self._buffer else np.zeros((0, self.channels))

    def stop_and_save(self) -> str:
        audio = self.stop()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, self.sample_rate, audio)
        tmp.close()
        return tmp.name
