"""Wake-word detector wrapping openWakeWord.

Accepts either:
  * a pretrained model name (alexa, hey_mycroft, hey_jarvis, hey_rhasspy), or
  * a path to a custom-trained ``.onnx`` file (relative paths resolve from the
    project root, absolute paths are used as-is).

See ``docs/wake_word_training.md`` for how to train a custom model in Colab.
"""

import re
from pathlib import Path

import numpy as np
from openwakeword.model import Model


PRETRAINED_WAKEWORDS: frozenset[str] = frozenset(
    {"alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy"}
)

DEFAULT_COOLDOWN_CHUNKS = 25  # about 2 seconds at 80 ms per chunk

_VERSION_SUFFIX = re.compile(r"_v\d+(?:\.\d+)*$")


def validate_threshold(threshold: float) -> None:
    """Raise ValueError unless threshold is in the closed interval [0, 1]."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"wake threshold must be in [0.0, 1.0], got {threshold!r}"
        )


def humanize_wake_phrase(wakeword: str) -> str:
    """Render a wake-word slug or path as the phrase the user actually says.

    Examples:
        hey_jarvis                       -> "Hey Jarvis"
        alexa                            -> "Alexa"
        models/wake/hey_synthesis.onnx   -> "Hey Synthesis"
        hey_jarvis_v0.1                  -> "Hey Jarvis"
    """
    if not wakeword:
        return ""
    stem = Path(wakeword).stem if wakeword.endswith(".onnx") else wakeword
    stem = _VERSION_SUFFIX.sub("", stem)
    words = [w for w in stem.replace("-", "_").split("_") if w]
    return " ".join(w.capitalize() for w in words)


def resolve_wakeword_source(
    wakeword: str,
    project_root: Path | None = None,
) -> tuple[str, str]:
    """Resolve a config value into (arg_for_openwakeword, score_lookup_hint).

    The score_lookup_hint is the *expected* key in openWakeWord's prediction
    dict. Callers should still verify against ``model.models.keys()`` after
    loading, since openWakeWord may rewrite keys for pretrained variants.
    """
    if not wakeword:
        raise ValueError("wakeword must be a non-empty string")

    if wakeword in PRETRAINED_WAKEWORDS:
        return wakeword, wakeword

    if not wakeword.endswith(".onnx"):
        raise ValueError(
            f"unknown wake word source: {wakeword!r}. "
            f"Use one of the pretrained names ({', '.join(sorted(PRETRAINED_WAKEWORDS))}) "
            f"or pass a path to a custom .onnx file (e.g. models/wake/hey_synthesis.onnx)."
        )

    path = Path(wakeword).expanduser()
    if not path.is_absolute():
        root = project_root or Path.cwd()
        path = (root / path).resolve()
    else:
        path = path.resolve()

    if path.suffix != ".onnx":
        raise ValueError(
            f"wake word model must be a .onnx file, got: {path.suffix} ({path})"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"wake word model file not found: {path}\n"
            f"Either set WAKE_MODEL to a pretrained name "
            f"({', '.join(sorted(PRETRAINED_WAKEWORDS))}) or train a custom "
            f"model and place the .onnx file at this path. See "
            f"docs/wake_word_training.md."
        )

    return str(path), path.stem


class OpenWakeWordDetector:
    """Real-time wake-word detector. Feed 80 ms float32 mono chunks at 16 kHz."""

    def __init__(
        self,
        wakeword: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_chunks: int = DEFAULT_COOLDOWN_CHUNKS,
        inference_framework: str = "onnx",
        project_root: Path | None = None,
    ) -> None:
        validate_threshold(threshold)
        if cooldown_chunks < 0:
            raise ValueError(
                f"cooldown_chunks must be non-negative, got {cooldown_chunks}"
            )

        model_arg, lookup_hint = resolve_wakeword_source(
            wakeword, project_root=project_root
        )

        try:
            self._model = Model(
                wakeword_models=[model_arg],
                inference_framework=inference_framework,
            )
        except Exception as e:
            raise RuntimeError(
                f"failed to load wake word model {wakeword!r}: {e}"
            ) from e

        loaded_keys = list(self._model.models.keys())
        if not loaded_keys:
            raise RuntimeError(
                f"openWakeWord loaded no models from {wakeword!r}"
            )
        # Pretrained models may load under versioned keys such as hey_jarvis_v0.1.
        self._score_key = (
            lookup_hint if lookup_hint in loaded_keys else loaded_keys[0]
        )
        self._threshold = threshold
        self._cooldown_default = cooldown_chunks
        self._cooldown_remaining = 0

    @property
    def score_key(self) -> str:
        """The key under which this detector's score appears in predictions."""
        return self._score_key

    def feed(self, chunk: np.ndarray) -> bool:
        """Return True when this chunk detects the wake phrase."""
        # openWakeWord wants int16 PCM, sounddevice gives us float32 in [-1, 1].
        int16_chunk = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
        scores = self._model.predict(int16_chunk)
        score = float(scores.get(self._score_key, 0.0))
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        if score >= self._threshold:
            self._cooldown_remaining = self._cooldown_default
            return True
        return False

    def reset(self) -> None:
        self._model.reset()
        self._cooldown_remaining = 0
