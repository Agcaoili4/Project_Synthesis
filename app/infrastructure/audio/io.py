"""Microphone and speaker I/O for the daemon.

Designed around small float32 mono chunks at 16 kHz so they can flow through
openWakeWord, Silero VAD, and faster-whisper without resampling.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd


CHUNK_MS = 80           # 80 ms chunks → 1280 samples @ 16 kHz, matches openWakeWord's expected frame
SAMPLE_RATE = 16000


class MicStream:
    """Async iterator yielding (1280,) float32 numpy arrays from the default mic."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = CHUNK_MS,
        device: int | str | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_size = sample_rate * chunk_ms // 1000
        self._device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> "MicStream":
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):
            if status:
                # XRuns happen if downstream is too slow — they're informational.
                return
            chunk = indata[:, 0].copy().astype(np.float32)
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
            except asyncio.QueueFull:
                pass

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=callback,
            device=self._device,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        while True:
            chunk = await self._queue.get()
            yield chunk


def play_chime(frequency_hz: float = 880.0, duration_s: float = 0.12) -> None:
    """Quick, non-blocking 'I'm listening' tone."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    envelope = np.exp(-3 * t / duration_s)
    tone = (0.25 * envelope * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    sd.play(tone, SAMPLE_RATE, blocking=False)
