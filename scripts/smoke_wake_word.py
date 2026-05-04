"""Standalone wake-word smoke test.

Runs the openWakeWord detector against your live mic and prints when it fires.
Use this to verify mic permission + wake word independently before the full
daemon. Run with:    .venv/bin/python scripts/smoke_wake_word.py

The first time you run this, macOS will prompt for microphone access.
Approve it; subsequent runs won't ask again.
"""

import asyncio
import logging

from app.core.config import get_settings
from app.infrastructure.audio.io import MicStream
from app.infrastructure.wake.openww_detector import OpenWakeWordDetector


async def main() -> None:
    s = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("smoke")

    log.info("loading wake word model '%s'...", s.wake_model)
    wake = OpenWakeWordDetector(wakeword=s.wake_model, threshold=s.wake_threshold)
    log.info("listening. Say 'Hey Jarvis' (Ctrl-C to exit).")

    async with MicStream() as mic:
        async for chunk in mic:
            if wake.feed(chunk):
                log.info("WAKE!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye.")
