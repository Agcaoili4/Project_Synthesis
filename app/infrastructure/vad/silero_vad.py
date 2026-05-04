"""Voice-activity detector wrapping Silero VAD.

Tracks per-chunk speech probability and exposes a simple 'has the speaker
gone silent for N ms' check, used by the daemon to know when to stop recording.
"""

import numpy as np
import torch
from silero_vad import load_silero_vad


SAMPLE_RATE = 16000


class SileroVADGate:
    def __init__(self, silence_ms: int = 900, speech_threshold: float = 0.5) -> None:
        self._model = load_silero_vad(onnx=False)
        self._silence_ms = silence_ms
        self._threshold = speech_threshold
        self._silence_run_ms = 0
        self._heard_speech = False

    def reset(self) -> None:
        self._silence_run_ms = 0
        self._heard_speech = False

    def feed(self, chunk: np.ndarray, chunk_ms: int = 80) -> bool:
        """Return True while we should keep recording, False once user has stopped."""
        # Silero expects 512-sample windows at 16 kHz. Pad/truncate as needed.
        win = self._to_silero_window(chunk)
        with torch.no_grad():
            prob = float(self._model(torch.from_numpy(win), SAMPLE_RATE).item())
        if prob >= self._threshold:
            self._heard_speech = True
            self._silence_run_ms = 0
            return True
        if self._heard_speech:
            self._silence_run_ms += chunk_ms
            return self._silence_run_ms < self._silence_ms
        # Pre-speech silence: keep going (don't time out before the user starts).
        return True

    @staticmethod
    def _to_silero_window(chunk: np.ndarray, target: int = 512) -> np.ndarray:
        if chunk.shape[0] >= target:
            return chunk[:target].astype(np.float32, copy=False)
        out = np.zeros(target, dtype=np.float32)
        out[: chunk.shape[0]] = chunk.astype(np.float32, copy=False)
        return out
