"""Unit tests for the wake-word detector's input validation and path resolution.

These tests intentionally do NOT load openWakeWord's actual ONNX model — that
work is covered in tests/integration/test_wake_detector.py. Here we only
exercise the resolver / validator so the suite stays fast.
"""

from pathlib import Path

import pytest

from app.infrastructure.wake.openww_detector import (
    PRETRAINED_WAKEWORDS,
    humanize_wake_phrase,
    resolve_wakeword_source,
)


def test_pretrained_name_passes_through_unchanged():
    arg, lookup_hint = resolve_wakeword_source("hey_jarvis")
    assert arg == "hey_jarvis"
    assert lookup_hint == "hey_jarvis"


def test_each_known_pretrained_name_resolves():
    for name in PRETRAINED_WAKEWORDS:
        arg, hint = resolve_wakeword_source(name)
        assert arg == name
        assert hint == name


def test_absolute_path_to_existing_onnx_resolves(tmp_path: Path):
    onnx = tmp_path / "hey_synthesis.onnx"
    onnx.write_bytes(b"fake-onnx-bytes")

    arg, hint = resolve_wakeword_source(str(onnx))

    assert arg == str(onnx.resolve())
    assert hint == "hey_synthesis"


def test_relative_path_resolves_from_project_root(tmp_path: Path):
    sub = tmp_path / "models" / "wake"
    sub.mkdir(parents=True)
    onnx = sub / "hey_synthesis.onnx"
    onnx.write_bytes(b"fake-onnx-bytes")

    arg, hint = resolve_wakeword_source(
        "models/wake/hey_synthesis.onnx", project_root=tmp_path
    )

    assert arg == str(onnx.resolve())
    assert hint == "hey_synthesis"


def test_missing_path_raises_filenotfound_with_helpful_message(tmp_path: Path):
    missing = tmp_path / "does_not_exist.onnx"
    with pytest.raises(FileNotFoundError) as exc:
        resolve_wakeword_source(str(missing))
    msg = str(exc.value)
    assert str(missing) in msg
    assert "hey_jarvis" in msg, "error should suggest pretrained fallbacks"


def test_wrong_extension_is_rejected(tmp_path: Path):
    not_onnx = tmp_path / "hey_synthesis.tflite"
    not_onnx.write_bytes(b"fake")
    with pytest.raises(ValueError) as exc:
        resolve_wakeword_source(str(not_onnx))
    assert ".onnx" in str(exc.value)


def test_unknown_name_without_extension_is_rejected():
    with pytest.raises(ValueError) as exc:
        resolve_wakeword_source("hey_synthesis")
    msg = str(exc.value)
    assert "hey_synthesis" in msg
    assert "pretrained" in msg.lower() or ".onnx" in msg


def test_empty_string_is_rejected():
    with pytest.raises(ValueError):
        resolve_wakeword_source("")


def test_threshold_validation_rejects_out_of_range_low():
    from app.infrastructure.wake.openww_detector import validate_threshold

    with pytest.raises(ValueError):
        validate_threshold(-0.1)


def test_threshold_validation_rejects_out_of_range_high():
    from app.infrastructure.wake.openww_detector import validate_threshold

    with pytest.raises(ValueError):
        validate_threshold(1.5)


def test_threshold_validation_accepts_boundary_values():
    from app.infrastructure.wake.openww_detector import validate_threshold

    validate_threshold(0.0)
    validate_threshold(1.0)


def test_humanize_pretrained_hey_jarvis():
    assert humanize_wake_phrase("hey_jarvis") == "Hey Jarvis"


def test_humanize_pretrained_alexa():
    assert humanize_wake_phrase("alexa") == "Alexa"


def test_humanize_custom_path_uses_basename(tmp_path: Path):
    onnx = tmp_path / "hey_synthesis.onnx"
    onnx.write_bytes(b"fake")
    assert humanize_wake_phrase(str(onnx)) == "Hey Synthesis"


def test_humanize_relative_path_uses_basename():
    assert humanize_wake_phrase("models/wake/hey_synthesis.onnx") == "Hey Synthesis"


def test_humanize_multi_word_basename():
    assert humanize_wake_phrase("ok_synthesis_buddy") == "Ok Synthesis Buddy"


def test_humanize_strips_trailing_version_suffix():
    # openWakeWord ships hey_jarvis as 'hey_jarvis_v0.1' in some builds.
    assert humanize_wake_phrase("hey_jarvis_v0.1") == "Hey Jarvis"


def test_humanize_empty_returns_empty():
    assert humanize_wake_phrase("") == ""
