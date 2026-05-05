import pytest
import numpy as np

from app.core.config import Settings
from app.infrastructure.tts.mlx_kokoro_engine import AudioSegment, MLXKokoroTTSEngine


class FakeFallback:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.files: list[tuple[str, str]] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def synthesize_to_file(self, text: str, path: str) -> None:
        self.files.append((text, path))


@pytest.mark.asyncio
async def test_kokoro_speak_uses_fallback_when_generation_fails(monkeypatch):
    fallback = FakeFallback()
    tts = MLXKokoroTTSEngine(fallback=fallback)

    def fail_generation(text: str):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(tts, "_synthesize_segments", fail_generation)

    await tts.speak("hello")

    assert fallback.spoken == ["hello"]


@pytest.mark.asyncio
async def test_kokoro_file_synthesis_uses_fallback_when_generation_fails(monkeypatch):
    fallback = FakeFallback()
    tts = MLXKokoroTTSEngine(fallback=fallback)

    def fail_generation(text: str):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(tts, "_synthesize_segments", fail_generation)

    await tts.synthesize_to_file("hello", "out.aiff")

    assert fallback.files == [("hello", "out.aiff")]


@pytest.mark.asyncio
async def test_kokoro_speak_ignores_blank_text(monkeypatch):
    fallback = FakeFallback()
    tts = MLXKokoroTTSEngine(fallback=fallback)
    monkeypatch.setattr(
        tts,
        "_synthesize_segments",
        lambda text: pytest.fail("blank text should not synthesize"),
    )

    await tts.speak("   ")

    assert fallback.spoken == []


@pytest.mark.asyncio
async def test_kokoro_speak_plays_one_cleaned_waveform(monkeypatch):
    tts = MLXKokoroTTSEngine()
    played: list[AudioSegment] = []

    monkeypatch.setattr(
        tts,
        "_synthesize_segments",
        lambda text: [
            AudioSegment(
                samples=np.linspace(0.8, -0.2, 480, dtype=np.float32),
                sample_rate=24000,
            ),
            AudioSegment(
                samples=np.linspace(-0.7, 0.4, 480, dtype=np.float32),
                sample_rate=24000,
            ),
        ],
    )
    monkeypatch.setattr(tts, "_play", lambda segment: played.append(segment))

    await tts.speak("hello")

    assert len(played) == 1
    audio = played[0].samples
    assert played[0].sample_rate == 24000
    assert np.max(np.abs(audio)) <= 0.92
    assert audio[0] == pytest.approx(0.0)
    assert audio[-1] == pytest.approx(0.0)


def test_kokoro_render_rejects_mixed_sample_rates(monkeypatch):
    tts = MLXKokoroTTSEngine()
    monkeypatch.setattr(
        tts,
        "_synthesize_segments",
        lambda text: [
            AudioSegment(samples=np.zeros(16, dtype=np.float32), sample_rate=24000),
            AudioSegment(samples=np.zeros(16, dtype=np.float32), sample_rate=22050),
        ],
    )

    with pytest.raises(RuntimeError, match="mixed sample rates"):
        tts._render_audio("hello")


def test_kokoro_output_preparation_resamples_and_pads(monkeypatch):
    tts = MLXKokoroTTSEngine()
    monkeypatch.setattr(tts, "_output_sample_rate", lambda: 48000)

    segment = AudioSegment(
        samples=np.linspace(-0.5, 0.5, 240, dtype=np.float32),
        sample_rate=24000,
    )

    prepared = tts._prepare_for_output_device(segment)

    assert prepared.sample_rate == 48000
    assert prepared.samples.dtype == np.float32
    assert len(prepared.samples) > len(segment.samples) * 2
    assert prepared.samples[0] == pytest.approx(0.0)
    assert prepared.samples[-1] == pytest.approx(0.0)
    assert np.max(np.abs(prepared.samples)) <= 0.92


def test_factory_builds_say_engine(monkeypatch):
    import app.infrastructure.tts.factory as factory

    class FakeSay:
        def __init__(self, voice: str, rate: int) -> None:
            self.voice = voice
            self.rate = rate

    monkeypatch.setattr(factory, "SayTTSEngine", FakeSay)

    tts = factory.build_tts_engine(Settings(tts_engine="say", say_voice="Moira"))

    assert isinstance(tts, FakeSay)
    assert tts.voice == "Moira"


def test_factory_builds_kokoro_with_say_fallback(monkeypatch):
    import app.infrastructure.tts.factory as factory

    class FakeSay:
        def __init__(self, voice: str, rate: int) -> None:
            self.voice = voice
            self.rate = rate

    class FakeKokoro:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(factory, "SayTTSEngine", FakeSay)
    monkeypatch.setattr(factory, "MLXKokoroTTSEngine", FakeKokoro)

    tts = factory.build_tts_engine(
        Settings(
            tts_engine="kokoro",
            kokoro_model="mlx-community/Kokoro-82M-bf16",
            kokoro_voice="af_heart",
        )
    )

    assert isinstance(tts, FakeKokoro)
    assert tts.kwargs["model_name"] == "mlx-community/Kokoro-82M-bf16"
    assert tts.kwargs["voice"] == "af_heart"
    assert isinstance(tts.kwargs["fallback"], FakeSay)
