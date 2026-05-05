"""TTS engine construction for the daemon."""

import logging
from typing import Protocol

from app.core.config import Settings
from app.infrastructure.audio.io import resolve_audio_device
from app.infrastructure.tts.mlx_kokoro_engine import MLXKokoroTTSEngine
from app.infrastructure.tts.say_engine import SayTTSEngine


log = logging.getLogger("synthesis.tts")


class TTSEngine(Protocol):
    async def speak(self, text: str) -> None: ...

    async def synthesize_to_file(self, text: str, path: str) -> None: ...


def build_tts_engine(settings: Settings) -> TTSEngine:
    """Build the configured TTS engine, keeping macOS `say` as fallback."""
    fallback = SayTTSEngine(voice=settings.say_voice, rate=settings.say_rate)
    engine = settings.tts_engine.strip().lower()

    if engine == "say":
        return fallback

    if engine in {"kokoro", "mlx", "mlx-audio", "mlx_kokoro"}:
        log.info(
            "using MLX-Audio Kokoro TTS. model=%s voice=%s",
            settings.kokoro_model,
            settings.kokoro_voice,
        )
        return MLXKokoroTTSEngine(
            model_name=settings.kokoro_model,
            voice=settings.kokoro_voice,
            speed=settings.kokoro_speed,
            lang_code=settings.kokoro_lang_code,
            sample_rate=settings.kokoro_sample_rate,
            output_device=resolve_audio_device(settings.output_device),
            fallback=fallback,
        )

    raise ValueError(
        f"unsupported TTS_ENGINE={settings.tts_engine!r}; expected 'say' or 'kokoro'"
    )
