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
import contextlib
import logging
import os
import subprocess
import sys
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
    SAMPLE_RATE as MIC_SAMPLE_RATE,
    MicStream,
    explain_unhealthy_mic,
    play_chime,
    probe_mic_health,
    resolve_audio_device,
)
from app.infrastructure.events import VisualizerEventBus, compute_audio_levels
from app.infrastructure.stt.whisper_engine import WhisperSTTEngine
from app.infrastructure.tts.factory import build_tts_engine
from app.infrastructure.vad.silero_vad import SileroVADGate
from app.infrastructure.wake import build_wake_detector


log = logging.getLogger("synthesis.daemon")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()


MAX_UTTERANCE_S = 15
VISUALIZER_SCRIPT = PROJECT_ROOT / "scripts" / "synthesis_visualizer.py"


def build_visualizer_command(settings) -> list[str]:
    """Build the pygame visualizer command used when the daemon owns the mic."""
    return [
        sys.executable,
        str(VISUALIZER_SCRIPT),
        "--no-mic",
        "--bus-host",
        settings.visualizer_bus_host,
        "--bus-port",
        str(settings.visualizer_bus_port),
    ]


async def start_visualizer_process(settings) -> subprocess.Popen | None:
    """Launch the visualizer as a child process, if configured and possible."""
    if not getattr(settings, "visualizer_auto_start", False):
        return None
    if not getattr(settings, "visualizer_bus_enabled", False):
        log.info("visualizer autostart skipped because VISUALIZER_BUS_ENABLED=false.")
        return None
    if not VISUALIZER_SCRIPT.exists():
        log.warning("visualizer autostart skipped; missing %s", VISUALIZER_SCRIPT)
        return None

    cmd = build_visualizer_command(settings)
    env = os.environ.copy()
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        log.warning("visualizer autostart failed: %s", exc)
        return None

    await asyncio.sleep(0.25)
    if proc.poll() is not None:
        log.warning("visualizer exited immediately with code %s.", proc.returncode)
        return None

    log.info("visualizer started. pid=%s", proc.pid)
    return proc


def stop_visualizer_process(proc: subprocess.Popen | None) -> None:
    """Terminate the autostarted visualizer without affecting manual windows."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


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
    wake = build_wake_detector(s, project_root=PROJECT_ROOT)
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
    pre_roll_chunks = max(1, s.wake_pre_roll_ms // CHUNK_MS)
    pre_roll = deque(maxlen=pre_roll_chunks)
    utterance: list[np.ndarray] = []
    utterance_started_at = 0.0
    turn_started_at = 0.0

    bus: VisualizerEventBus | None = None
    if s.visualizer_bus_enabled:
        bus = VisualizerEventBus(host=s.visualizer_bus_host, port=s.visualizer_bus_port)

    def set_state(new: State) -> None:
        # Keep daemon state and visualizer state in one place.
        nonlocal state
        state = new
        if bus is not None:
            bus.publish_state(new.name.lower())

    # During playback the mic may be quiet or picking up the speakers. Kokoro
    # can report its own PCM levels, which makes the visualizer follow the
    # assistant voice directly. Engines without PCM access ignore this hook.
    if bus is not None and hasattr(tts, "set_level_listener"):
        def _on_tts_chunk(samples: np.ndarray, sample_rate: int) -> None:
            levels = compute_audio_levels(samples, sample_rate)
            bus.publish_levels(
                levels["rms"], levels["low"], levels["mid"], levels["high"], source="tts",
            )
        tts.set_level_listener(_on_tts_chunk)

    # Mic chunks arrive every 80 ms. Publishing every other chunk keeps the
    # bus quiet while still giving the visualizer enough signal to animate.
    mic_level_skip = 2
    mic_level_counter = 0

    async with contextlib.AsyncExitStack() as stack:
        if bus is not None:
            await stack.enter_async_context(bus)
        visualizer_proc = await start_visualizer_process(s)
        stack.callback(stop_visualizer_process, visualizer_proc)
        # Publish after the bus starts so newly opened visualizers begin in a
        # known state instead of waiting for the next user turn.
        set_state(State.IDLE)

        http = await stack.enter_async_context(httpx.AsyncClient(timeout=120.0))
        mic = await stack.enter_async_context(MicStream(device=input_device))
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
                    wake.reset()
                    if s.wake_chime_enabled:
                        play_chime()
                    set_state(State.LISTENING)
                    vad.reset()
                    utterance = list(pre_roll)
                    utterance_started_at = time.monotonic()
                    turn_started_at = utterance_started_at
                    mic_level_counter = 0

            elif state is State.LISTENING:
                utterance.append(chunk)
                still_speaking = vad.feed(chunk, chunk_ms=CHUNK_MS)
                if bus is not None:
                    mic_level_counter += 1
                    if mic_level_counter % mic_level_skip == 0:
                        levels = compute_audio_levels(chunk, MIC_SAMPLE_RATE)
                        bus.publish_levels(
                            levels["rms"],
                            levels["low"],
                            levels["mid"],
                            levels["high"],
                            source="mic",
                        )
                duration = time.monotonic() - utterance_started_at
                if not vad.heard_speech and duration > s.no_speech_timeout_s:
                    log.info(
                        "no speech heard %.2fs after wake — back to idle.",
                        duration,
                    )
                    set_state(State.IDLE)
                    wake.reset()
                    pre_roll.clear()
                    utterance = []
                    continue
                if not still_speaking or duration > MAX_UTTERANCE_S:
                    set_state(State.TRANSCRIBING)
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
                        set_state(State.IDLE)
                        wake.reset()
                        pre_roll.clear()
                        utterance = []
                        continue

                    set_state(State.THINKING)
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
                    set_state(State.SPEAKING)
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

                    set_state(State.IDLE)
                    wake.reset()
                    pre_roll.clear()
                    utterance = []

# Daemon entry point, can be interrupted when user press Ctrl+C
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
