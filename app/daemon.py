"""Project Synthesis daemon — the always-on audio loop.

Handles audio processing, transcription, LLM call, and TTS playback.

State machine:
    IDLE          -> waiting for wake word
    LISTENING     -> recording the user's utterance until silence
    TRANSCRIBING  -> running faster-whisper on the buffer
    THINKING      -> POSTing to the brain
    SPEAKING      -> playing the reply through `say`

Run with:    python -m app.daemon
The brain (uvicorn app.main:app) must be running too.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from enum import Enum, auto

import httpx
import numpy as np

from app.core.config import get_settings
from app.infrastructure.audio.io import CHUNK_MS, MicStream, play_chime
from app.infrastructure.stt.whisper_engine import WhisperSTTEngine
from app.infrastructure.tts.say_engine import SayTTSEngine
from app.infrastructure.vad.silero_vad import SileroVADGate
from app.infrastructure.wake.openww_detector import OpenWakeWordDetector


log = logging.getLogger("synthesis.daemon")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()


MAX_UTTERANCE_S = 15
PRE_ROLL_MS = 320  # keep this much audio from before the wake fired


async def run_loop() -> None:
    s = get_settings()
    session_id = f"daemon-{uuid.uuid4().hex[:8]}"
    brain_url = f"http://{s.brain_host}:{s.brain_port}/converse"

    log.info("loading models...")
    wake = OpenWakeWordDetector(wakeword=s.wake_model, threshold=s.wake_threshold)
    vad = SileroVADGate(silence_ms=s.vad_silence_ms)
    stt = WhisperSTTEngine(
        model_name=s.whisper_model,
        device=s.whisper_device,
        compute_type=s.whisper_compute_type,
    )
    tts = SayTTSEngine(voice=s.say_voice, rate=s.say_rate)
    log.info("models ready. session=%s. say '%s' to begin.", session_id, s.wake_model)

    state = State.IDLE
    pre_roll = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
    utterance: list[np.ndarray] = []
    utterance_started_at = 0.0

    async with httpx.AsyncClient(timeout=120.0) as http, MicStream() as mic:
        async for chunk in mic:
            if state is State.IDLE:
                pre_roll.append(chunk)
                if wake.feed(chunk):
                    log.info("wake!")
                    play_chime()
                    state = State.LISTENING
                    vad.reset()
                    utterance = list(pre_roll)
                    utterance_started_at = time.monotonic()

            elif state is State.LISTENING:
                utterance.append(chunk)
                still_speaking = vad.feed(chunk, chunk_ms=CHUNK_MS)
                duration = time.monotonic() - utterance_started_at
                if not still_speaking or duration > MAX_UTTERANCE_S:
                    state = State.TRANSCRIBING
                    audio = np.concatenate(utterance)
                    log.info("captured %.2fs of audio. transcribing...", len(audio) / 16000)

                    transcript = await stt.transcribe(audio)
                    log.info("you: %s", transcript)

                    if not transcript.strip():
                        log.info("empty transcript — back to idle.")
                        state = State.IDLE
                        wake.reset()
                        continue

                    state = State.THINKING
                    try:
                        resp = await http.post(
                            brain_url,
                            json={"transcript": transcript, "session_id": session_id},
                        )
                        resp.raise_for_status()
                        reply = resp.json()["reply"]
                    except Exception as e:
                        log.error("brain error: %s", e)
                        reply = "I had trouble reaching the brain. Please check that uvicorn is running."

                    log.info("synthesis: %s", reply)
                    state = State.SPEAKING
                    await tts.speak(reply)

                    state = State.IDLE
                    wake.reset()
                    pre_roll.clear()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        log.info("bye.")


if __name__ == "__main__":
    main()
