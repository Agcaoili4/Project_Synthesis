"""Loopback event bus for live Synthesis UI surfaces.

Design goals:
    * Keep the daemon independent from pygame and dashboard code.
    * Treat the visualizer as optional; a missing or slow subscriber must not
      interrupt wake detection, transcription, or TTS playback.
    * Accept publishes from worker threads. TTS playback and audio callbacks
      do not run on the bus event loop.

Protocol — newline-delimited JSON, UTF-8. Each event is one of:
    {"type": "state",  "value": "idle|listening|transcribing|thinking|speaking",
     "ts": 1735689600.0}
    {"type": "level",  "source": "mic|tts",
     "rms": 0.42, "low": 0.31, "mid": 0.55, "high": 0.18,
     "ts": 1735689600.0}
    {"type": "hello",  "version": 1, "ts": 1735689600.0}   # sent on connect
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

import numpy as np


log = logging.getLogger("synthesis.events")

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# If a visualizer stops reading, drop it before its socket buffer can slow the daemon.
MAX_CLIENT_BACKLOG_BYTES = 256 * 1024


# ---------------------------------------------------------------------------
# Audio-level extraction (used by the daemon for mic and TTS PCM)
# ---------------------------------------------------------------------------

def compute_audio_levels(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """RMS + 3-band FFT energy from a chunk of mono float32 audio.

    Values are normalized to roughly 0..1 so the visualizer can treat mic and
    TTS audio the same way. Empty or all-zero input returns zeroes.
    """
    if samples is None or samples.size == 0:
        return {"rms": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0}

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32, copy=False)

    rms_raw = float(np.sqrt(np.mean(samples * samples)))
    rms = math.tanh(rms_raw * 6.0)

    # A power-of-two window keeps FFT bins predictable even when callers pass
    # chunks with different lengths.
    n = 1024 if samples.size >= 1024 else 1 << max(1, int(math.log2(max(1, samples.size))))
    if samples.size > n:
        window_samples = samples[-n:]
    else:
        window_samples = np.pad(samples, (0, n - samples.size))

    window = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(window_samples * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, sample_rate))

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return 0.0
        return float(spec[m].mean())

    low = math.tanh(band(60, 250) / 4.0)
    mid = math.tanh(band(250, 2000) / 2.0)
    high = math.tanh(band(2000, 8000) / 1.0)

    return {"rms": rms, "low": low, "mid": mid, "high": high}


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class VisualizerEventBus:
    """Asyncio TCP server that broadcasts NDJSON events to every subscriber.

    Lifecycle:
        bus = VisualizerEventBus()
        await bus.start()          # non-fatal if port is busy
        bus.publish_state("idle")  # safe to call from any thread, any time
        bus.publish_levels(...)
        await bus.stop()
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """True if the server is listening. Publishes are no-ops otherwise."""
        return self._enabled

    @property
    def address(self) -> tuple[str, int]:
        return (self._host, self._port)

    async def start(self) -> None:
        """Bind and listen. Logs and stays disabled if the port is busy."""
        self._loop = asyncio.get_running_loop()
        try:
            self._server = await asyncio.start_server(
                self._handle_client, host=self._host, port=self._port
            )
        except OSError as exc:
            log.warning(
                "visualizer event bus could not bind %s:%d (%s) — running without it.",
                self._host,
                self._port,
                exc,
            )
            self._server = None
            self._enabled = False
            return
        self._enabled = True
        log.info("visualizer event bus listening on %s:%d", self._host, self._port)

    async def __aenter__(self) -> "VisualizerEventBus":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def stop(self) -> None:
        if self._server is None:
            self._enabled = False
            return
        self._enabled = False
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception:  # noqa: BLE001
            pass

        for writer in list(self._clients):
            self._drop_client(writer)
        self._clients.clear()
        self._server = None
        log.info("visualizer event bus stopped.")

    # -- publish -----------------------------------------------------------

    def publish_state(self, state: str) -> None:
        self._publish({"type": "state", "value": state, "ts": time.time()})

    def publish_levels(
        self,
        rms: float,
        low: float,
        mid: float,
        high: float,
        source: str = "mic",
    ) -> None:
        self._publish(
            {
                "type": "level",
                "source": source,
                "rms": rms,
                "low": low,
                "mid": mid,
                "high": high,
                "ts": time.time(),
            }
        )

    def _publish(self, event: dict[str, Any]) -> None:
        """Queue an event without blocking audio or TTS worker threads."""
        if not self._enabled or not self._clients:
            return
        if self._loop is None or self._loop.is_closed():
            return
        # Serialize before crossing threads so the event loop only writes bytes.
        try:
            payload = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            log.debug("dropping malformed event: %s (%s)", event, exc)
            return
        try:
            self._loop.call_soon_threadsafe(self._broadcast, payload)
        except RuntimeError:
            pass

    def _broadcast(self, payload: bytes) -> None:
        """Runs on the bus loop. Writes to every client; drops slow ones."""
        for writer in list(self._clients):
            transport = writer.transport
            if transport is None or transport.is_closing():
                self._drop_client(writer)
                continue
            try:
                if writer.transport.get_write_buffer_size() > MAX_CLIENT_BACKLOG_BYTES:
                    log.warning("visualizer subscriber too slow — dropping.")
                    self._drop_client(writer)
                    continue
                writer.write(payload)
            except Exception as exc:  # noqa: BLE001
                log.debug("write to subscriber failed: %s", exc)
                self._drop_client(writer)

    # -- client lifecycle --------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        log.info("visualizer subscriber connected from %s", peer)
        self._clients.add(writer)
        try:
            hello = json.dumps(
                {"type": "hello", "version": PROTOCOL_VERSION, "ts": time.time()},
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            writer.write(hello)
            await writer.drain()

            # Subscribers are read-only. This read waits for disconnects.
            while True:
                data = await reader.read(4096)
                if not data:
                    break
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("subscriber loop ended: %s", exc)
        finally:
            log.info("visualizer subscriber disconnected: %s", peer)
            self._drop_client(writer)

    def _drop_client(self, writer: asyncio.StreamWriter) -> None:
        self._clients.discard(writer)
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
