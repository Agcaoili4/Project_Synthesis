# Training a custom "Hey Synthesis" wake word

openWakeWord ships only four pretrained wake phrases (`alexa`, `hey_mycroft`,
`hey_jarvis`, `hey_rhasspy`). To make Synthesis respond to literally
"Hey Synthesis," you train your own model.

This is a one-time, ~45-minute effort. Once trained, you drop the resulting
`.onnx` file into `models/wake/`, point `WAKE_MODEL` at it, and you're done
forever.

## How the training works (briefly)

You don't need to record yourself thousands of times. openWakeWord's training
pipeline:

1. Generates ~1,000 synthetic positive samples by piping the phrase
   "hey synthesis" through ~1,000 different Piper TTS voices (varied accents,
   pitches, speaking rates). This is your *positive* set — what should fire.
2. Mixes them with thousands of public-domain *negative* samples (people
   saying other things, ambient noise, music). This is what should *not* fire.
3. Fine-tunes a small ONNX discriminator on top of openWakeWord's shared
   embedding model.

The output is a ~1.3 MB `.onnx` file.

## Path A — Google Colab (recommended)

This is the path you should take unless you have specific reasons not to.
Free GPU, no environment setup, ~45 minutes wall-clock.

1. Open the official notebook:
   <https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb>
2. Click **"Open in Colab"** at the top.
3. **Runtime → Change runtime type → T4 GPU** (or any GPU; CPU is unusably slow).
4. Find the cell that sets `target_phrase` (it's near the top, in a config cell).
   Change it to:
   ```python
   target_phrase = "hey synthesis"
   ```
   Leave the other parameters at their defaults for v0.
5. **Runtime → Run all.** Walk away. It runs in stages:
   - Synthetic data generation (~10 min)
   - Negative-sample download (~5 min)
   - Training (~25 min on a T4)
6. When the last cell finishes, it writes a file in the Colab filesystem.
   Look for `hey_synthesis.onnx` (or similar — the notebook prints the path
   in its final output). In the Files panel on the left, right-click → Download.
7. Move the downloaded file somewhere local, then run:
   ```bash
   .venv/bin/python scripts/install_custom_wake_word.py ~/Downloads/hey_synthesis.onnx --update-env
   ```
   The script validates that openWakeWord can load it, copies it to
   `models/wake/hey_synthesis.onnx`, and writes the `WAKE_MODEL=` line to
   `.env`.
8. Confirm `.env` contains:
   ```bash
   WAKE_MODEL=models/wake/hey_synthesis.onnx
   ```
9. Verify with the smoke script before running the full daemon:
   ```bash
   .venv/bin/python scripts/smoke_wake_word.py
   ```
   Say "Hey Synthesis." You should see `WAKE!` printed.

If false fires are common, raise `WAKE_THRESHOLD` in `.env` (try 0.6, 0.7).
If real fires are missed, lower it (0.4, 0.3).

## Path B — Local training

Possible but painful on Apple Silicon. The training pipeline depends on Piper
TTS, which has the same broken arm64 wheel situation we hit during v0 setup.
If you really want local training:

1. Clone the repo: `git clone https://github.com/dscripka/openWakeWord.git`
2. `cd openWakeWord && pip install -e .[train]`
3. Resolve Piper-on-arm64 yourself (build from source, use Rosetta, or use
   Linux x86_64). Plan for 4–10 hours of debugging.
4. Run their training script with `target_phrase="hey synthesis"`.
5. Take the resulting `.onnx` and run it through
   `scripts/install_custom_wake_word.py` like in Path A.

Recommendation: don't.

## Verifying the installed model

After installation, before running the daemon end-to-end:

```bash
.venv/bin/python scripts/smoke_wake_word.py
```

This launches just the wake-word loop with your live mic and prints `WAKE!`
when it fires. Use it to:

- Confirm the model loads
- Tune `WAKE_THRESHOLD` (try a range; settle on the value where saying the
  phrase always fires but ambient speech does not)
- Verify it doesn't false-fire while you talk normally on calls or watch TV

Once you're happy with the threshold, restart the brain and the daemon and
have a real conversation.

## Roadmap

- v1: ship a *trained* `hey_synthesis.onnx` in the repo so other developers
  don't have to retrain. (We can't ship it now because we haven't trained it
  yet — chicken/egg, and the training data isn't pre-built.)
