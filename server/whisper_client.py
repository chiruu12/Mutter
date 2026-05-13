import platform
from pathlib import Path

import numpy as np


class WhisperClient:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._use_mlx = platform.system() == "Darwin"

    def transcribe_file(self, audio_path: str | Path) -> str:
        if self._use_mlx:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}",
            )
        else:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_name, compute_type="int8")
            segments, _ = model.transcribe(str(audio_path))
            return " ".join(seg.text for seg in segments).strip()
        return result["text"].strip()

    def transcribe_array(self, audio: np.ndarray) -> str:
        if self._use_mlx:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}",
            )
            return result["text"].strip()

        import tempfile
        from scipy.io import wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, 16000, audio)
            return self.transcribe_file(tmp.name)
