import asyncio

import numpy as np
from pytest import MonkeyPatch

from app.infrastructure.stt import whisper_engine
from app.infrastructure.stt.whisper_engine import WhisperSTTEngine


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    instances: list["_FakeWhisperModel"] = []

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_kwargs: dict[str, object] | None = None
        _FakeWhisperModel.instances.append(self)

    def transcribe(
        self, audio: np.ndarray | str, **kwargs: object
    ) -> tuple[list[_Segment], object]:
        _ = audio
        self.transcribe_kwargs = kwargs
        return [_Segment(" hello "), _Segment("world")], object()


def test_default_model_prefers_lower_latency_base_en() -> None:
    stt = WhisperSTTEngine()

    assert stt._model_name == "base.en"


def test_warm_up_loads_model(monkeypatch: MonkeyPatch) -> None:
    _FakeWhisperModel.instances.clear()
    monkeypatch.setattr(whisper_engine, "WhisperModel", _FakeWhisperModel)

    stt = WhisperSTTEngine(model_name="base.en", device="cpu", compute_type="int8")
    asyncio.run(stt.warm_up())

    assert len(_FakeWhisperModel.instances) == 1
    assert _FakeWhisperModel.instances[0].model_name == "base.en"


def test_transcribe_uses_low_latency_options(monkeypatch: MonkeyPatch) -> None:
    _FakeWhisperModel.instances.clear()
    monkeypatch.setattr(whisper_engine, "WhisperModel", _FakeWhisperModel)

    stt = WhisperSTTEngine()
    text = asyncio.run(stt.transcribe(np.zeros(16000, dtype=np.float32)))

    assert text == "hello world"
    assert _FakeWhisperModel.instances[0].transcribe_kwargs == {
        "beam_size": 1,
        "vad_filter": False,
        "condition_on_previous_text": False,
        "without_timestamps": True,
    }
