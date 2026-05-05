"""Microphone and speaker I/O for the daemon.

Designed around small float32 mono chunks at 16 kHz so they can flow through
openWakeWord, Silero VAD, and faster-whisper without resampling.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


CHUNK_MS = 80           # 80 ms chunks → 1280 samples @ 16 kHz, matches openWakeWord's expected frame
SAMPLE_RATE = 16000

# Below this RMS the mic is effectively silent (digital zero plus float noise).
# A reasonable speaking-into-mic chunk has RMS in the 0.01–0.3 range.
MIN_HEALTHY_RMS = 1e-5

log = logging.getLogger("synthesis.audio")


@dataclass(frozen=True)
class MicHealth:
    """Result of a startup mic-level probe."""

    rms: float
    peak: float
    chunks_observed: int
    device_name: str

    @property
    def healthy(self) -> bool:
        return self.rms > MIN_HEALTHY_RMS


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
                # Surface sounddevice status (input overflow, etc.) instead of
                # dropping it silently — these can mask real mic issues.
                log.warning("sounddevice status: %s", status)
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


def probe_mic_health(
    duration_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
) -> MicHealth:
    """Synchronously read ~`duration_s` of mic audio and measure level.

    Used at startup to fail fast when:
      * mic permission was denied (we get all-zero buffers on macOS)
      * the wrong input device is selected
      * the mic is hardware-muted

    Blocks the calling thread until the read completes. Run this BEFORE the
    main async loop opens its own InputStream — sounddevice does not allow
    two concurrent input streams on the same device.
    """
    n_frames = int(sample_rate * duration_s)
    audio = sd.rec(
        n_frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    )
    flat = audio.reshape(-1)
    rms = float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    device_index = device if device is not None else sd.default.device[0]
    try:
        device_name = sd.query_devices(device_index)["name"]
    except Exception:
        device_name = "unknown"
    return MicHealth(
        rms=rms,
        peak=peak,
        chunks_observed=int(n_frames / (sample_rate * CHUNK_MS / 1000)),
        device_name=str(device_name),
    )


def resolve_audio_device(device: str | int | None) -> str | int | None:
    """Normalize env-provided sounddevice selectors.

    sounddevice accepts either a numeric device index or a name substring. Empty
    env values should behave like the system default instead of selecting "".
    Some dotenv parsers treat `INPUT_DEVICE=  # comment` as the literal value
    "# comment", so comment-only values are treated as unset too.
    """
    if device is None:
        return None
    if isinstance(device, int):
        return device
    normalized = device.strip()
    if not normalized or normalized.startswith("#"):
        return None
    try:
        return int(normalized)
    except ValueError:
        return normalized


def explain_unhealthy_mic(health: MicHealth) -> str:
    """Render an actionable error message when probe_mic_health fails."""
    return (
        f"microphone is silent (rms={health.rms:.6f}, peak={health.peak:.6f}, "
        f"device={health.device_name!r}).\n"
        "Most common cause on macOS: microphone permission for your terminal "
        "was denied. Open System Settings → Privacy & Security → Microphone "
        "and enable access for the terminal app you're running this from "
        "(Terminal, iTerm, VS Code, etc.). You may need to fully quit and "
        "relaunch the terminal afterward.\n"
        "If permission is granted, check that the right device is selected: "
        "run `python -c \"import sounddevice as sd; print(sd.query_devices())\"` "
        "to list devices, then set INPUT_DEVICE in .env."
    )


def play_chime(frequency_hz: float = 880.0, duration_s: float = 0.12) -> None:
    """Quick, non-blocking 'I'm listening' tone."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    envelope = np.exp(-3 * t / duration_s)
    tone = (0.25 * envelope * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    sd.play(tone, SAMPLE_RATE, blocking=False)
