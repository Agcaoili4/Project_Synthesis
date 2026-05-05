"""Streaming-Whisper wake-word detector.

A drop-in alternative to ``OpenWakeWordDetector`` that does not require a
trained ``.onnx`` model — useful when openWakeWord training is blocked
(e.g., Apple Silicon piper-phonemize wheels) but the assistant still needs
to react to a literal phrase like "hey synthesis".

How it works:
    * Each ``feed(chunk)`` call appends the 80 ms / 16 kHz mono float32
      chunk to a rolling window (default 1.2 s) and returns immediately.
    * A background worker thread runs Whisper (faster-whisper "tiny.en" by
      default — ~75 MB, ~150–300 ms on CPU) over snapshots of that window
      every ``poll_ms``, but only when the audio actually contains energy
      above the silence floor — running Whisper on silence both wastes CPU
      and produces hallucinated text.
    * If the transcript loosely matches the configured phrase, an internal
      "fire" flag is set. The next ``feed`` call returns ``True`` and arms
      a cooldown to prevent duplicate wake events.

Trade-offs vs. trained openWakeWord:
    * ~5–10% of one CPU core steady-state (vs. ~1%).
    * ~600–900 ms wake-to-fire latency (vs. ~250 ms).
    * Zero training. Works on any platform Whisper supports. Retargeting
      only requires a different phrase setting.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Iterable

import numpy as np


log = logging.getLogger("synthesis.wake.whisper")

SAMPLE_RATE = 16_000
CHUNK_MS = 80  # matches MicStream — kept loose for safety
DEFAULT_COOLDOWN_CHUNKS = 25  # ~2 s @ 80 ms chunks; same default as openWW path

# Skip Whisper on near-silent windows. Quiet-room noise on tested M-series
# mics stayed below this level, and transcribing silence often invents words.
SILENCE_RMS_FLOOR = 0.005


_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return _WORD_RE.findall(text.lower())


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance. Used for a fuzzy wake-word match.

    Whisper sometimes hears 'synthesis' as 'synthesises', 'synthesia',
    'cynthesis', etc. Edit distance ≤ 2 catches most of those.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def phrase_matches(transcript: str, phrase: str, max_word_distance: int = 2) -> bool:
    """Return True if the transcript "contains" the phrase, fuzzily.

    A phrase like "hey synthesis" matches if the transcript contains a run
    of consecutive tokens whose edit distance to the phrase tokens (per word)
    is within ``max_word_distance``. Punctuation and case are ignored.
    """
    phrase_words = _normalize(phrase)
    if not phrase_words:
        return False
    transcript_words = _normalize(transcript)
    if len(transcript_words) < len(phrase_words):
        return False

    n = len(phrase_words)
    for start in range(len(transcript_words) - n + 1):
        window = transcript_words[start:start + n]
        ok = True
        for w_phrase, w_actual in zip(phrase_words, window):
            allowed = max_word_distance if len(w_phrase) > 4 else 1
            if _levenshtein(w_phrase, w_actual) > allowed:
                ok = False
                break
        if ok:
            return True
    return False


class WhisperWakeDetector:
    """Streaming-Whisper wake detector with the same surface as
    ``OpenWakeWordDetector`` so the daemon can use either interchangeably.

    Public API:
        feed(chunk) -> bool   # True when this chunk observes the wake phrase
        reset()
        score_key             # informational label used for logs/HUD
    """

    def __init__(
        self,
        phrase: str = "hey synthesis",
        model_name: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        window_ms: int = 1200,
        poll_ms: int = 350,
        cooldown_chunks: int = DEFAULT_COOLDOWN_CHUNKS,
        silence_rms: float = SILENCE_RMS_FLOOR,
        max_word_distance: int = 2,
    ) -> None:
        if not phrase.strip():
            raise ValueError("wake phrase must be non-empty")
        if cooldown_chunks < 0:
            raise ValueError("cooldown_chunks must be non-negative")

        self._phrase = phrase.strip()
        self._max_word_distance = max_word_distance
        self._window_chunks = max(2, window_ms // CHUNK_MS)
        # Shorter windows rarely contain enough speech to identify the phrase.
        self._min_samples = int(SAMPLE_RATE * 0.25)
        self._poll_s = max(0.05, poll_ms / 1000.0)
        self._cooldown_default = cooldown_chunks
        self._silence_rms = silence_rms

        # Keep phrase-matching tests importable without the Whisper runtime.
        from faster_whisper import WhisperModel  # noqa: PLC0415

        log.info(
            "loading whisper wake model '%s' (%s/%s) — this may take 30–60s "
            "on first run while the weights download from Hugging Face.",
            model_name,
            device,
            compute_type,
        )
        load_started = time.monotonic()
        self._model: object = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        log.info(
            "whisper wake model loaded in %.2fs.",
            time.monotonic() - load_started,
        )

        self._buffer: deque[np.ndarray] = deque(maxlen=self._window_chunks)
        self._lock = threading.Lock()
        self._fire = False
        self._cooldown_remaining = 0
        self._stop_event = threading.Event()
        # Lets the worker skip repeated transcription of the same audio window.
        self._buffer_version = 0
        self._last_seen_version = -1
        # Protects against stale wake hits. If reset() runs while Whisper is
        # transcribing, the worker must ignore that old transcript.
        self._reset_session = 0

        self._worker = threading.Thread(
            target=self._run_worker,
            name="whisper-wake-worker",
            daemon=True,
        )
        self._worker.start()

    # -- public API used by the daemon -------------------------------------

    @property
    def score_key(self) -> str:
        """Informational label, mirrors the openWakeWord property."""
        return self._phrase.replace(" ", "_")

    def feed(self, chunk: np.ndarray) -> bool:
        """Append a chunk and report whether the wake phrase fired."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)
        with self._lock:
            self._buffer.append(chunk)
            self._buffer_version += 1
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return False
            if self._fire:
                self._fire = False
                self._cooldown_remaining = self._cooldown_default
                return True
        return False

    def reset(self) -> None:
        """Drop pending audio and clear any queued fire/cooldown state."""
        with self._lock:
            self._buffer.clear()
            self._fire = False
            self._cooldown_remaining = 0
            self._buffer_version += 1
            self._last_seen_version = self._buffer_version
            self._reset_session += 1

    def stop(self) -> None:
        """Tell the worker thread to exit. Safe to call multiple times."""
        self._stop_event.set()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass

    # -- worker ------------------------------------------------------------

    def _snapshot(self) -> tuple[np.ndarray, int] | None:
        """Take a copy of the rolling buffer plus the reset-session it belongs to.

        The session is what the worker uses after transcription to verify the
        audio it analyzed is still the audio the detector cares about — i.e.
        ``reset()`` hasn't been called in the meantime.
        """
        with self._lock:
            if self._buffer_version == self._last_seen_version:
                return None
            self._last_seen_version = self._buffer_version
            if not self._buffer:
                return None
            chunks: Iterable[np.ndarray] = list(self._buffer)
            session = self._reset_session
        audio = np.concatenate(list(chunks)).astype(np.float32, copy=False)
        if audio.size < self._min_samples:
            return None
        return audio, session

    def _is_silent(self, audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        return rms < self._silence_rms

    def _transcribe(self, audio: np.ndarray) -> str:
        # faster-whisper accepts float32 mono at 16 kHz.
        try:
            segments, _info = self._model.transcribe(
                audio,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                # The wake phrase is English; skip per-window language detection.
                language="en",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("whisper wake transcribe failed: %s", exc)
            return ""
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _run_worker(self) -> None:
        log.info("whisper wake worker running, listening for %r", self._phrase)
        while not self._stop_event.is_set():
            self._stop_event.wait(self._poll_s)
            if self._stop_event.is_set():
                break

            snap = self._snapshot()
            if snap is None:
                continue
            audio, session = snap
            if self._is_silent(audio):
                continue

            with self._lock:
                in_cooldown = self._cooldown_remaining > 0
            if in_cooldown:
                continue

            text = self._transcribe(audio)
            if not text:
                continue

            matched = phrase_matches(text, self._phrase, self._max_word_distance)
            if matched:
                with self._lock:
                    if session != self._reset_session:
                        # This transcript belongs to audio from before reset().
                        log.debug(
                            "whisper wake match dropped after reset (text=%r)",
                            text,
                        )
                        continue
                    self._fire = True
                log.info("whisper wake matched: %r", text)
            else:
                # Useful when tuning the phrase or threshold on a new mic.
                log.info("whisper wake heard (no match): %r", text)
        log.info("whisper wake worker exiting")
