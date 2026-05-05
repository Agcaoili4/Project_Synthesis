"""Integration test: load a real openWakeWord model and feed it audio.

Slow (~3s) — instantiates the ONNX runtime and the pretrained hey_jarvis model.
"""

import numpy as np
import pytest

from app.infrastructure.wake.openww_detector import OpenWakeWordDetector


CHUNK_SIZE = 1280  # 80 ms @ 16 kHz, matches the daemon's MicStream output


@pytest.mark.integration
def test_pretrained_hey_jarvis_loads_and_does_not_fire_on_silence():
    detector = OpenWakeWordDetector(wakeword="hey_jarvis", threshold=0.5)

    assert detector.score_key == "hey_jarvis"

    # Feed 2 seconds of digital silence; the detector must not false-fire.
    silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
    fires = sum(detector.feed(silence) for _ in range(25))
    assert fires == 0


@pytest.mark.integration
def test_pretrained_hey_jarvis_does_not_fire_on_white_noise():
    detector = OpenWakeWordDetector(wakeword="hey_jarvis", threshold=0.5)
    rng = np.random.default_rng(0)
    fires = 0
    for _ in range(25):
        chunk = rng.normal(scale=0.05, size=CHUNK_SIZE).astype(np.float32)
        fires += detector.feed(chunk)
    # Noise might tickle the model occasionally; allow at most 1 fire in 25 chunks.
    assert fires <= 1


@pytest.mark.integration
def test_reset_clears_cooldown():
    detector = OpenWakeWordDetector(wakeword="hey_jarvis", threshold=0.5)
    detector._cooldown_remaining = 10  # simulate a recent fire
    detector.reset()
    assert detector._cooldown_remaining == 0
