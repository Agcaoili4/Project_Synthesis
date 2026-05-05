"""TTS adapter for local MLX-Audio Kokoro voices.

MLX-Audio is optional and mainly targets Apple Silicon. This adapter keeps
`say` available as a fallback so the daemon can still speak if Kokoro is not
installed yet or the model cannot be loaded.
"""

import asyncio
import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf


log = logging.getLogger("synthesis.tts.kokoro")
FADE_MS = 8
SEGMENT_GAP_MS = 12
OUTPUT_START_PAD_MS = 30
OUTPUT_END_PAD_MS = 80
PEAK_HEADROOM = 0.92
# Level listener cadence for the visualizer. The renderer smooths these
# ~20 updates/sec into its 60 FPS animation loop.
LEVEL_CHUNK_MS = 50

LevelListener = Callable[[np.ndarray, int], None]


class TTSEngine(Protocol):
    async def speak(self, text: str) -> None: ...

    async def synthesize_to_file(self, text: str, path: str) -> None: ...


@dataclass(frozen=True)
class AudioSegment:
    samples: np.ndarray
    sample_rate: int


class MLXKokoroTTSEngine:
    """Generate speech locally with MLX-Audio Kokoro and play it via sounddevice."""

    def __init__(
        self,
        model_name: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        sample_rate: int = 24000,
        output_device: int | str | None = None,
        fallback: TTSEngine | None = None,
    ) -> None:
        self._model_name = model_name
        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code
        self._sample_rate = sample_rate
        self._output_device = output_device
        self._fallback = fallback
        self._model = None
        self._level_listener: LevelListener | None = None

    def set_level_listener(self, listener: LevelListener | None) -> None:
        """Register a callback that receives slices of about-to-play PCM.

        The visualizer uses this to follow Synthesis's own voice during
        SPEAKING. The callback runs inside the playback worker thread, so it
        must stay quick and avoid async primitives.
        """
        self._level_listener = listener

    async def warm_up(self) -> None:
        """Load the model at daemon startup to avoid first-reply delay."""
        try:
            await asyncio.to_thread(self._load_model)
        except Exception as exc:
            if self._fallback is None:
                raise
            log.warning("Kokoro warm-up failed; will use fallback TTS: %s", exc)

    async def speak(self, text: str) -> None:
        """Speak `text` aloud and return when audio playback finishes."""
        if not text.strip():
            return
        try:
            audio = await asyncio.to_thread(self._render_audio, text)
        except Exception as exc:
            if self._fallback is None:
                raise
            log.warning("Kokoro synthesis failed; falling back to say: %s", exc)
            await self._fallback.speak(text)
            return

        if audio is not None:
            await asyncio.to_thread(self._play, audio)

    async def synthesize_to_file(self, text: str, path: str) -> None:
        """Write speech to a WAV-compatible path instead of playing it."""
        if not text.strip():
            return
        try:
            audio = await asyncio.to_thread(self._render_audio, text)
        except Exception as exc:
            if self._fallback is None:
                raise
            log.warning("Kokoro file synthesis failed; falling back to say: %s", exc)
            await self._fallback.synthesize_to_file(text, path)
            return
        if audio is None:
            return
        await asyncio.to_thread(sf.write, str(Path(path)), audio.samples, audio.sample_rate)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install the optional Kokoro stack "
                'with `uv pip install --python .venv/bin/python -e ".[kokoro]"`.'
            ) from exc

        self._model = load_model(self._model_name)
        return self._model

    def _synthesize_segments(self, text: str) -> list[AudioSegment]:
        model = self._load_model()
        segments: list[AudioSegment] = []
        for result in model.generate(
            text,
            voice=self._voice,
            speed=self._speed,
            lang_code=self._lang_code,
        ):
            sample_rate = int(getattr(result, "sample_rate", self._sample_rate))
            segments.append(
                AudioSegment(
                    samples=self._normalize_audio(result.audio),
                    sample_rate=sample_rate,
                )
            )
        return segments

    def _render_audio(self, text: str) -> AudioSegment | None:
        segments = self._synthesize_segments(text)
        if not segments:
            return None
        sample_rate = segments[0].sample_rate
        if any(segment.sample_rate != sample_rate for segment in segments):
            raise RuntimeError("Kokoro returned segments with mixed sample rates.")

        cleaned = [self._prepare_for_playback(segment.samples, sample_rate) for segment in segments]
        audio = self._join_segments(cleaned, sample_rate)
        audio = self._limit_peak(audio)
        return AudioSegment(samples=audio, sample_rate=sample_rate)

    def _normalize_audio(self, audio) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32)
        samples = np.squeeze(samples)
        if samples.ndim == 0:
            samples = samples.reshape(1)
        if samples.ndim > 1:
            channel_axis = 0 if samples.shape[0] <= samples.shape[-1] else 1
            samples = samples.mean(axis=channel_axis)
        return samples.astype(np.float32)

    def _prepare_for_playback(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = self._remove_dc_offset(samples)
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
        return self._fade_edges(samples, sample_rate)

    def _remove_dc_offset(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32)
        return (samples - np.mean(samples, dtype=np.float64)).astype(np.float32)

    def _fade_edges(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if samples.size < 2:
            return samples.astype(np.float32)
        faded = samples.copy().astype(np.float32)
        fade_samples = min(int(sample_rate * FADE_MS / 1000), faded.size // 2)
        if fade_samples <= 1:
            return faded
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        faded[:fade_samples] *= fade_in
        faded[-fade_samples:] *= fade_out
        return faded

    def _join_segments(self, segments: list[np.ndarray], sample_rate: int) -> np.ndarray:
        if len(segments) == 1:
            return segments[0]
        gap_samples = int(sample_rate * SEGMENT_GAP_MS / 1000)
        gap = np.zeros(gap_samples, dtype=np.float32)
        pieces: list[np.ndarray] = []
        for index, segment in enumerate(segments):
            pieces.append(segment)
            if index < len(segments) - 1 and gap_samples > 0:
                pieces.append(gap)
        return np.concatenate(pieces).astype(np.float32)

    def _limit_peak(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32)
        peak = float(np.max(np.abs(samples)))
        if peak > PEAK_HEADROOM:
            samples = samples * (PEAK_HEADROOM / peak)
        return np.clip(samples, -PEAK_HEADROOM, PEAK_HEADROOM).astype(np.float32)

    def _play(self, segment: AudioSegment) -> None:
        playback = self._prepare_for_output_device(segment)
        samples = playback.samples
        sample_rate = playback.sample_rate

        # Publish levels immediately before each write so the visualizer tracks
        # the audio the user is about to hear.
        chunk_size = max(1, int(sample_rate * LEVEL_CHUNK_MS / 1000))
        listener = self._level_listener

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=self._output_device,
            latency="high",
        ) as stream:
            for offset in range(0, samples.size, chunk_size):
                slice_ = samples[offset:offset + chunk_size]
                if listener is not None:
                    try:
                        listener(slice_, sample_rate)
                    except Exception as exc:  # noqa: BLE001
                        # Visualizer telemetry is optional; speech playback is not.
                        log.debug("tts level listener raised: %s", exc)
                stream.write(slice_.reshape(-1, 1))

    def _prepare_for_output_device(self, segment: AudioSegment) -> AudioSegment:
        sample_rate = self._output_sample_rate() or segment.sample_rate
        samples = self._resample_audio(segment.samples, segment.sample_rate, sample_rate)
        samples = self._fade_edges(samples, sample_rate)
        samples = self._pad_silence(samples, sample_rate)
        samples = self._limit_peak(samples)
        return AudioSegment(samples=samples, sample_rate=sample_rate)

    def _output_sample_rate(self) -> int | None:
        try:
            device_info = sd.query_devices(self._output_device, "output")
            return int(round(float(device_info["default_samplerate"])))
        except Exception as exc:
            log.warning("could not resolve output device sample rate; using Kokoro rate: %s", exc)
            return None

    def _resample_audio(
        self,
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate == target_rate or samples.size == 0:
            return samples.astype(np.float32)

        from scipy.signal import resample_poly

        divisor = gcd(source_rate, target_rate)
        up = target_rate // divisor
        down = source_rate // divisor
        return resample_poly(samples, up, down).astype(np.float32)

    def _pad_silence(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        start_samples = int(sample_rate * OUTPUT_START_PAD_MS / 1000)
        end_samples = int(sample_rate * OUTPUT_END_PAD_MS / 1000)
        if start_samples <= 0 and end_samples <= 0:
            return samples.astype(np.float32)
        return np.pad(samples, (start_samples, end_samples)).astype(np.float32)
