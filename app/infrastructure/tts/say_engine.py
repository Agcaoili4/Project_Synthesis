"""TTS adapter using the macOS `say` command.

Pros: zero install, native arm64, ~50ms to start speaking, ships with Moira (Irish female).
Cons: less expressive than Piper/ElevenLabs. We'll upgrade in v0.5.
"""

import asyncio
import shutil


class SayTTSEngine:
    def __init__(self, voice: str = "Moira", rate: int = 185) -> None:
        self._voice = voice
        self._rate = rate
        if shutil.which("say") is None:
            raise RuntimeError(
                "macOS `say` not found on PATH. This adapter only works on macOS."
            )

    async def speak(self, text: str) -> None:
        """Speak `text` aloud and return when audio playback finishes."""
        if not text.strip():
            return
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", self._voice, "-r", str(self._rate), text
        )
        await proc.wait()

    async def synthesize_to_file(self, text: str, path: str) -> None:
        """Write speech to an AIFF file at `path` instead of playing it."""
        if not text.strip():
            return
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", self._voice, "-r", str(self._rate), "-o", path, text
        )
        await proc.wait()
