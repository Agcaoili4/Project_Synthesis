"""Wake-word detection.

Two interchangeable backends:

* ``OpenWakeWordDetector`` — fast, low-CPU, requires a trained ``.onnx`` file.
* ``WhisperWakeDetector``  — no training, slightly higher CPU, drop-in.

Both expose ``feed(chunk) -> bool``, ``reset()``, and ``score_key``. Use
``build_wake_detector(settings, project_root)`` to pick the right one based
on configuration; callers do not need backend-specific code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np


log = logging.getLogger("synthesis.wake")


class WakeDetector(Protocol):
    """Common surface implemented by both wake backends."""

    @property
    def score_key(self) -> str: ...

    def feed(self, chunk: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


def build_wake_detector(settings, project_root: Path) -> WakeDetector:
    """Pick the wake backend that matches the user's configuration.

    Selection rules:
        * ``WAKE_ENGINE=whisper``      → WhisperWakeDetector
        * ``WAKE_ENGINE=openwakeword`` → OpenWakeWordDetector (default)
    """
    engine = (getattr(settings, "wake_engine", "openwakeword") or "openwakeword").strip().lower()

    if engine == "whisper":
        # Keep openWakeWord-only installs usable without faster-whisper.
        from app.infrastructure.wake.whisper_detector import WhisperWakeDetector

        log.info(
            "wake engine: whisper (model=%s, phrase=%r)",
            settings.wake_whisper_model,
            settings.wake_phrase,
        )
        return WhisperWakeDetector(
            phrase=settings.wake_phrase,
            model_name=settings.wake_whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            window_ms=settings.wake_whisper_window_ms,
            poll_ms=settings.wake_whisper_poll_ms,
        )

    if engine in {"openwakeword", "openww", "ww"}:
        from app.infrastructure.wake.openww_detector import OpenWakeWordDetector

        log.info("wake engine: openWakeWord (model=%s)", settings.wake_model)
        return OpenWakeWordDetector(
            wakeword=settings.wake_model,
            threshold=settings.wake_threshold,
            project_root=project_root,
        )

    raise ValueError(
        f"unsupported WAKE_ENGINE={engine!r}; expected 'openwakeword' or 'whisper'"
    )


__all__ = ["WakeDetector", "build_wake_detector"]
