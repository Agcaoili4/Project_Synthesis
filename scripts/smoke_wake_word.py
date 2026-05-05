"""Standalone wake-word smoke test.

Runs the openWakeWord detector against your live mic and prints when it fires.
Use this to verify mic permission + wake word independently before the full
daemon. Run with:    .venv/bin/python scripts/smoke_wake_word.py

The first time you run this, macOS will prompt for microphone access.
Approve it; subsequent runs won't ask again.
"""

import asyncio
import logging
from pathlib import Path

from app.core.config import get_settings
from app.infrastructure.audio.io import (
    MicStream,
    explain_unhealthy_mic,
    probe_mic_health,
    resolve_audio_device,
)
from app.infrastructure.wake import build_wake_detector


PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    s = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("smoke")
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

    log.info("loading wake detector (engine=%s)...", s.wake_engine)
    wake = build_wake_detector(s, project_root=PROJECT_ROOT)
    log.info("listening for '%s' (Ctrl-C to exit).", wake.score_key)

    async with MicStream(device=input_device) as mic:
        async for chunk in mic:
            if wake.feed(chunk):
                log.info("WAKE!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSee you around!")
