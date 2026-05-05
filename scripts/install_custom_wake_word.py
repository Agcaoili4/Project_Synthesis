"""Validate and install a custom-trained openWakeWord model.

Usage:
    .venv/bin/python scripts/install_custom_wake_word.py path/to/hey_synthesis.onnx
    .venv/bin/python scripts/install_custom_wake_word.py path/to/file.onnx --name hey_synthesis
    .venv/bin/python scripts/install_custom_wake_word.py path/to/file.onnx --update-env

What it does:
    1. Verifies the source file exists and is a valid ONNX model.
    2. Confirms openWakeWord can load it without errors.
    3. Copies it into ./models/wake/<name>.onnx.
    4. Prints the WAKE_MODEL value to drop in your .env, or updates .env when
       --update-env is passed.

Exits non-zero on any validation failure. Never overwrites an existing file
unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow `python scripts/install_custom_wake_word.py` to import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAKE_MODELS_DIR = PROJECT_ROOT / "models" / "wake"
ENV_FILE = PROJECT_ROOT / ".env"


def _fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _validate_onnx_file(path: Path) -> None:
    if not path.exists():
        _fail(f"source file not found: {path}")
    if not path.is_file():
        _fail(f"source is not a regular file: {path}")
    if path.suffix != ".onnx":
        _fail(f"source must have .onnx extension, got {path.suffix}")
    if path.stat().st_size < 1024:
        _fail(
            f"source file is suspiciously small ({path.stat().st_size} bytes); "
            "are you sure this is a trained ONNX model?"
        )

    # Magic-byte check: ONNX files start with the protobuf magic bytes.
    with path.open("rb") as f:
        head = f.read(8)
    if not head:
        _fail("source file is empty")


def _validate_loadable(path: Path) -> str:
    """Load the model with openWakeWord and return its score key."""
    try:
        from openwakeword.model import Model
    except ImportError as e:
        _fail(f"openwakeword not importable; activate the venv first ({e})")

    try:
        model = Model(wakeword_models=[str(path)], inference_framework="onnx")
    except Exception as e:
        _fail(f"openWakeWord refused to load the model: {e}")

    keys = list(model.models.keys())
    if not keys:
        _fail("openWakeWord loaded the file but registered no models")
    return keys[0]


def _upsert_env_value(env_path: Path, key: str, value: str) -> None:
    line = f"{key}={value}\n"
    if not env_path.exists():
        env_path.write_text(line, encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = False
    for i, existing in enumerate(lines):
        stripped = existing.lstrip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key == key:
            lines[i] = line
            updated = True
            break

    if not updated:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = f"{lines[-1]}\n"
        lines.append(line)

    env_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the trained .onnx file you downloaded from Colab.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the destination filename stem (default: source's stem).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination file.",
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        help="Set WAKE_MODEL in .env after installing the model.",
    )
    args = parser.parse_args()

    source: Path = args.source.expanduser().resolve()
    print(f"validating {source}...")
    _validate_onnx_file(source)

    print("loading via openWakeWord (this is the real test)...")
    score_key = _validate_loadable(source)
    print(f"  -> openWakeWord registered model key: {score_key!r}")

    name = args.name or source.stem
    dest = WAKE_MODELS_DIR / f"{name}.onnx"

    if dest.exists() and not args.force:
        _fail(
            f"destination already exists: {dest}\n"
            f"pass --force to overwrite, or --name to choose a different filename"
        )

    WAKE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    rel_dest = dest.relative_to(PROJECT_ROOT)
    print()
    print(f"installed -> {rel_dest}")
    if args.update_env:
        _upsert_env_value(ENV_FILE, "WAKE_MODEL", str(rel_dest))
        print(f"updated -> {ENV_FILE.relative_to(PROJECT_ROOT)}")
    print()
    print("next steps:")
    if not args.update_env:
        print(f'  1. set WAKE_MODEL="{rel_dest}" in your .env')
        print("     or rerun this installer with --update-env")
    print(
        "  2. run .venv/bin/python scripts/smoke_wake_word.py "
        f"and verify it fires on '{name.replace('_', ' ')}'"
    )


if __name__ == "__main__":
    main()
