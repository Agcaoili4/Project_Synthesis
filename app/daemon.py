"""Project Synthesis daemon — the always-on audio loop.

Handles audio processing, transcription, LLM call, and TTS playback.

State machine:
    IDLE          -> waiting for wake word
    LISTENING     -> recording the user's utterance until silence
    TRANSCRIBING  -> running faster-whisper on the buffer
    THINKING      -> POSTing to the brain
    SPEAKING      -> playing the reply through the configured local TTS engine

Run with:    python -m app.daemon
The brain (uvicorn app.main:app) must be running too.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from enum import Enum, auto
from pathlib import Path

import httpx
import numpy as np

from app.core.config import get_settings
from app.infrastructure.audio.io import (
    CHUNK_MS,
    MicStream,
    explain_unhealthy_mic,
    play_chime,
    probe_mic_health,
    resolve_audio_device,
)
from app.infrastructure.stt.whisper_engine import WhisperSTTEngine
from app.infrastructure.tts.factory import build_tts_engine
from app.infrastructure.vad.silero_vad import SileroVADGate
from app.infrastructure.wake.openww_detector import OpenWakeWordDetector


log = logging.getLogger("synthesis.daemon")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    input_device = resolve_audio_device(s.input_device)

    mic_health = probe_mic_health(device=input_device)
    if not mic_health.healthy:
        raise RuntimeError(explain_unhealthy_mic(mic_health))
    log.info(
        "mic ready. device=%s rms=%.6f peak=%.6f",
        mic_health.device_name,
        mic_health.rms,
        mic_health.peak,
    )

    log.info("loading models...")
    model_load_started_at = time.monotonic()
    wake = OpenWakeWordDetector(
        wakeword=s.wake_model,
        threshold=s.wake_threshold,
        project_root=PROJECT_ROOT,
    )
    vad = SileroVADGate(silence_ms=s.vad_silence_ms)
    stt = WhisperSTTEngine(
        model_name=s.whisper_model,
        device=s.whisper_device,
        compute_type=s.whisper_compute_type,
    )
    tts = build_tts_engine(s)
    log.info("warming whisper model...")
    await stt.warm_up()
    if hasattr(tts, "warm_up"):
        log.info("warming tts engine...")
        await tts.warm_up()
    log.info(
        "models ready in %.2fs. session=%s. say '%s' to begin.",
        time.monotonic() - model_load_started_at,
        session_id,
        wake.score_key,
    )

    state = State.IDLE
    pre_roll = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
    utterance: list[np.ndarray] = []
    utterance_started_at = 0.0
    turn_started_at = 0.0

    async with (
        httpx.AsyncClient(timeout=120.0) as http,
        MicStream(device=input_device) as mic,
    ):
        headers = (
            {"Authorization": f"Bearer {s.brain_api_token}"}
            if s.brain_api_token
            else None
        )
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
                    turn_started_at = utterance_started_at

            elif state is State.LISTENING:
                utterance.append(chunk)
                still_speaking = vad.feed(chunk, chunk_ms=CHUNK_MS)
                duration = time.monotonic() - utterance_started_at
                if not vad.heard_speech and duration > s.no_speech_timeout_s:
                    log.info(
                        "no speech heard %.2fs after wake — back to idle.",
                        duration,
                    )
                    state = State.IDLE
                    wake.reset()
                    pre_roll.clear()
                    utterance = []
                    continue
                if not still_speaking or duration > MAX_UTTERANCE_S:
                    state = State.TRANSCRIBING
                    capture_finished_at = time.monotonic()
                    audio = np.concatenate(utterance)
                    captured_audio_s = len(audio) / 16000
                    log.info("captured %.2fs of audio. transcribing...", captured_audio_s)

                    stt_started_at = time.monotonic()
                    transcript = await stt.transcribe(audio)
                    stt_finished_at = time.monotonic()
                    if s.log_conversation_text:
                        log.info("you: %s", transcript)
                    else:
                        log.info("transcribed %d chars.", len(transcript))

                    if not transcript.strip():
                        log.info(
                            "empty transcript — back to idle. timing: capture=%.2fs stt=%.2fs total=%.2fs",
                            capture_finished_at - turn_started_at,
                            stt_finished_at - stt_started_at,
                            time.monotonic() - turn_started_at,
                        )
                        state = State.IDLE
                        wake.reset()
                        pre_roll.clear()
                        utterance = []
                        continue

                    state = State.THINKING
                    brain_started_at = time.monotonic()
                    try:
                        resp = await http.post(
                            brain_url,
                            json={"transcript": transcript, "session_id": session_id},
                            headers=headers,
                        )
                        resp.raise_for_status()
                        reply = resp.json()["reply"]
                    except Exception as e:
                        log.error("brain error: %s", e)
                        reply = "I had trouble reaching the brain. Please check that uvicorn is running."
                    brain_finished_at = time.monotonic()

                    if s.log_conversation_text:
                        log.info("synthesis: %s", reply)
                    else:
                        log.info("received reply with %d chars.", len(reply))
                    state = State.SPEAKING
                    tts_started_at = time.monotonic()
                    await tts.speak(reply)
                    tts_finished_at = time.monotonic()
                    log.info(
                        "turn timing: capture=%.2fs audio=%.2fs stt=%.2fs brain=%.2fs tts=%.2fs total=%.2fs",
                        capture_finished_at - turn_started_at,
                        captured_audio_s,
                        stt_finished_at - stt_started_at,
                        brain_finished_at - brain_started_at,
                        tts_finished_at - tts_started_at,
                        tts_finished_at - turn_started_at,
                    )

                    state = State.IDLE
                    wake.reset()
                    pre_roll.clear()
                    utterance = []


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
