import logging
import platform
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("mutter.whisper")

_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class WhisperClient:
    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        self.model_name = model_name
        self._use_mlx = platform.system() == "Darwin"
        self._repo = _MLX_REPOS.get(model_name, f"mlx-community/whisper-{model_name}")

    def transcribe_file(self, audio_path: str | Path) -> str:
        t0 = time.perf_counter()
        if self._use_mlx:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self._repo,
            )
            text = result["text"].strip()
        else:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_name, compute_type="int8")
            segments, _ = model.transcribe(str(audio_path))
            text = " ".join(seg.text for seg in segments).strip()
        elapsed = time.perf_counter() - t0
        log.info("[whisper] transcribed in %.1fs (%d chars)", elapsed, len(text))
        return text

    def transcribe_array(self, audio: np.ndarray) -> str:
        if self._use_mlx:
            import mlx_whisper

            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._repo,
            )
            text = result["text"].strip()
            elapsed = time.perf_counter() - t0
            log.info("[whisper] transcribed array in %.1fs", elapsed)
            return text

        import tempfile

        from scipy.io import wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, 16000, audio)
            tmp_path = Path(tmp.name)
        try:
            return self.transcribe_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
