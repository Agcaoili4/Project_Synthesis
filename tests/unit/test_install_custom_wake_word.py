"""Tests for the wake-word install script's pure-validation helpers.

We test the file-validation logic without invoking openWakeWord (the load
check is exercised by the integration test for the detector itself).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import install_custom_wake_word as installer  # noqa: E402


def test_validate_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(SystemExit):
        installer._validate_onnx_file(missing)


def test_validate_rejects_directory(tmp_path: Path):
    d = tmp_path / "fake.onnx"
    d.mkdir()
    with pytest.raises(SystemExit):
        installer._validate_onnx_file(d)


def test_validate_rejects_wrong_extension(tmp_path: Path):
    bad = tmp_path / "model.tflite"
    bad.write_bytes(b"x" * 2048)
    with pytest.raises(SystemExit):
        installer._validate_onnx_file(bad)


def test_validate_rejects_tiny_file(tmp_path: Path):
    tiny = tmp_path / "tiny.onnx"
    tiny.write_bytes(b"x" * 16)
    with pytest.raises(SystemExit):
        installer._validate_onnx_file(tiny)


def test_validate_accepts_plausible_onnx(tmp_path: Path):
    plausible = tmp_path / "ok.onnx"
    plausible.write_bytes(b"\x08\x07" + b"x" * 4096)
    installer._validate_onnx_file(plausible)


def test_upsert_env_value_creates_env_file(tmp_path: Path):
    env = tmp_path / ".env"
    installer._upsert_env_value(env, "WAKE_MODEL", "models/wake/hey_synthesis.onnx")
    assert env.read_text(encoding="utf-8") == (
        "WAKE_MODEL=models/wake/hey_synthesis.onnx\n"
    )


def test_upsert_env_value_replaces_existing_key(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "BRAIN_PORT=8000\nWAKE_MODEL=hey_jarvis\nWAKE_THRESHOLD=0.5\n",
        encoding="utf-8",
    )
    installer._upsert_env_value(env, "WAKE_MODEL", "models/wake/hey_synthesis.onnx")
    assert env.read_text(encoding="utf-8") == (
        "BRAIN_PORT=8000\n"
        "WAKE_MODEL=models/wake/hey_synthesis.onnx\n"
        "WAKE_THRESHOLD=0.5\n"
    )


def test_upsert_env_value_appends_missing_key(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("BRAIN_PORT=8000", encoding="utf-8")
    installer._upsert_env_value(env, "WAKE_MODEL", "models/wake/hey_synthesis.onnx")
    assert env.read_text(encoding="utf-8") == (
        "BRAIN_PORT=8000\nWAKE_MODEL=models/wake/hey_synthesis.onnx\n"
    )
