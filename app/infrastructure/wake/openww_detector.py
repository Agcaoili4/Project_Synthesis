"""Wake-word detector wrapping openWakeWord.

Feeds 80 ms chunks (1280 samples @ 16 kHz) into the model and reports when the
configured wake phrase fires. Pretrained `hey_jarvis` is used for v0; a custom
`hey_synthesis` will be trained as a follow-up step.
"""

import numpy as np
from openwakeword.model import Model


class OpenWakeWordDetector:
    def __init__(
        self,
        wakeword: str = "hey_jarvis",
        threshold: float = 0.5,
        inference_framework: str = "onnx",
    ) -> None:
        self._model = Model(
            wakeword_models=[wakeword], inference_framework=inference_framework
        )
        self._wakeword = wakeword
        self._threshold = threshold
        self._cooldown_chunks = 0  # don't double-fire across consecutive chunks

    def feed(self, chunk: np.ndarray) -> bool:
        """Return True iff the wake phrase was just detected."""
        # openWakeWord wants int16 PCM, sounddevice gives us float32 in [-1, 1].
        int16_chunk = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
        scores = self._model.predict(int16_chunk)
        score = float(scores.get(self._wakeword, 0.0))
        if self._cooldown_chunks > 0:
            self._cooldown_chunks -= 1
            return False
        if score >= self._threshold:
            self._cooldown_chunks = 25  # ~2 seconds of suppression
            return True
        return False

    def reset(self) -> None:
        self._model.reset()
        self._cooldown_chunks = 0
