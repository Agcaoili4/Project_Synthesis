"""Integration: speak text via macOS `say` then transcribe it with faster-whisper.

Slow (~5s) — loads the Whisper model. Marked `integration` so unit-test runs
stay fast.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.stt.whisper_engine import WhisperSTTEngine
from app.infrastructure.tts.say_engine import SayTTSEngine


@pytest.mark.integration
async def test_say_then_whisper_recovers_phrase():
    spoken = "The capital of France is Paris."

    with tempfile.TemporaryDirectory() as d:
        aiff_path = Path(d) / "spoken.aiff"
        wav_path = Path(d) / "spoken.wav"

        tts = SayTTSEngine(voice="Moira")
        await tts.synthesize_to_file(spoken, str(aiff_path))
        assert aiff_path.exists() and aiff_path.stat().st_size > 1000

        # AIFF → mono float32 16 kHz WAV (Whisper's preferred input)
        data, sr = sf.read(str(aiff_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            data = resample_poly(data, 16000, sr).astype(np.float32)
        sf.write(str(wav_path), data, 16000)

        stt = WhisperSTTEngine(model_name="small.en")
        transcript = await stt.transcribe_file(wav_path)

    norm = transcript.lower().strip().rstrip(".").rstrip("?").rstrip("!")
    assert "paris" in norm
    assert "capital" in norm
    assert "france" in norm
