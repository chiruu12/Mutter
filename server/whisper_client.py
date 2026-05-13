from pathlib import Path

import numpy as np


class WhisperClient:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model_path = f"mlx-community/whisper-{model_name}"

    def transcribe_file(self, audio_path: str | Path) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._model_path,
        )
        return result["text"].strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
        )
        return result["text"].strip()
