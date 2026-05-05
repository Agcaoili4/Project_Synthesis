"""STT adapter wrapping faster-whisper (CTranslate2 build of OpenAI Whisper).

Loads the model lazily by default, with an explicit warm-up hook for the
daemon so the first voice turn does not pay the model-load cost.
"""

import asyncio
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


class WhisperSTTEngine:
    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def _ensure_loaded(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    async def warm_up(self) -> None:
        """Load the model before the first user turn."""
        await asyncio.to_thread(self._ensure_loaded)

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono PCM array sampled at 16 kHz."""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    async def transcribe_file(self, path: str | Path) -> str:
        return await asyncio.to_thread(self._transcribe_sync, str(path))

    def _transcribe_sync(self, audio: np.ndarray | str) -> str:
        model = self._ensure_loaded()
        segments, _info = model.transcribe(
            audio,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
