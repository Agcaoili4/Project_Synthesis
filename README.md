# Project Synthesis

A locally-run, voice-driven AI assistant inspired by JARVIS — lives on your Mac, listens for a wake word, holds a real conversation, and never sends your audio to the cloud.

## Status

**v0 in progress.** First milestone: a fully-local voice loop — wake word → speech-to-text → LLM → text-to-speech — on Apple Silicon with zero recurring API cost.

## Vision

Long-term, Synthesis aims to be a personal AI in the JARVIS / FRIDAY mold: ambient, conversational, tool-using (email, calendar, web, local files), with persistent memory of you. We're getting there one subsystem at a time. **v0 is the voice spine; everything else plugs into it.**

## How it works (v0)

Two core processes on your machine, plus an automatically launched visual surface:

- **Daemon** (`app/daemon.py`) — owns the microphone, wake word, STT, and TTS. Always-on, real-time.
- **Brain** (`app/main.py`, FastAPI) — owns the LLM, conversation state, and the localhost app console. Restartable without losing the audio loop.
- **Visualizer** (`scripts/synthesis_visualizer.py`) — pygame window launched by the daemon by default; subscribes to a local event bus and renders state, audio energy, and DNA-style speaking animation.

They communicate over local HTTP. No cloud. No telemetry. Your audio never leaves the box.

```
  ┌──────────────────────────────┐         ┌──────────────────────────────┐
  │  DAEMON  (app/daemon.py)     │         │  BRAIN  (app/main.py)         │
  │                              │         │                              │
  │  mic ──► openWakeWord        │  HTTP   │  POST /converse              │
  │           │                  │ ──────► │       │                      │
  │           ▼ "hey synthesis"     │         │       ▼                      │
  │       Silero VAD + capture   │         │   Ollama (Qwen 2.5 7B)       │
  │           │                  │         │       │                      │
  │           ▼ audio buffer     │         │       ▼ reply text           │
  │       faster-whisper (STT)   │         │   append to session history  │
  │           │                  │         │       │                      │
  │           ▼ transcript       │         │       ▼                      │
  │       call BRAIN ────────────┼────────►│   return reply               │
  │           ▼ reply text       │ ◄────── │                              │
  │       local TTS ─────► speakers│       │  GET  /  (app console)       │
  │           │                  │         │  POST /dashboard/tts         │
  │           └── event bus :8765│         │                              │
  └──────────────────────────────┘         └──────────────────────────────┘
       ▲                                            ▲
       │                                            │ http://localhost:8000
       └── pygame visualizer                         └── browser / local app
```

## Stack

| Concern                  | Choice                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Wake word                | openWakeWord by default (`hey_jarvis`), with optional Whisper wake detection for "Hey Synthesis" without model training. |
| Voice activity detection | Silero VAD                                                                                                               |
| Speech-to-text           | faster-whisper (`base.en` by default; `small.en` for higher accuracy)                                                    |
| LLM                      | Qwen 2.5 7B Instruct via Ollama                                                                                          |
| Text-to-speech           | MLX-Audio Kokoro by default (`TTS_ENGINE=kokoro`) with macOS `say` fallback; TTS Studio in the app console.              |
| Audio I/O                | sounddevice + scipy resampling for cleaner Kokoro playback                                                               |
| Backend                  | FastAPI                                                                                                                  |
| App console              | Local server-rendered HTML using `assets/Synthesis.png` as the product mark                                              |
| Visualizer               | pygame, daemon event bus on `127.0.0.1:8765`, auto-started by `app.daemon`, DNA-like speaking animation                  |
| Long-term memory         | SQLite + `sqlite-vec`, embeddings via Ollama `nomic-embed-text`, semantic recall injected into the LLM prompt            |

Disk footprint: ~6 GB of models. Peak RAM: ~7 GB. Targets 16 GB Apple Silicon comfortably.

## Project layout

```
project-synthesis/
├── app/
│   ├── main.py                       # FastAPI entrypoint (the BRAIN process)
│   ├── daemon.py                     # audio loop entrypoint (the DAEMON process)
│   │
│   ├── core/
│   │   ├── config.py                 # Pydantic Settings: model paths, ports, wake word, voice
│   │   └── security.py               # loopback / bearer-token access guard
│   │
│   ├── domain/                       # Pure business logic. NO I/O. NO frameworks.
│   │   └── conversation.py           # Conversation, Message, Role
│   │
│   ├── application/                  # Use cases. Orchestrates domain + infrastructure.
│   │   └── converse.py               # ConverseUseCase: takes transcript, returns reply
│   │
│   ├── infrastructure/               # Adapters to the outside world.
│   │   ├── llm/ollama_client.py      # talks to Ollama
│   │   ├── stt/whisper_engine.py     # wraps faster-whisper
│   │   ├── events/visualizer_bus.py  # local NDJSON event bus for live UI
│   │   ├── tts/mlx_kokoro_engine.py  # wraps MLX-Audio Kokoro local TTS
│   │   ├── tts/say_engine.py         # wraps macOS `say` fallback
│   │   ├── wake/openww_detector.py   # wraps openWakeWord
│   │   ├── wake/whisper_detector.py  # optional Whisper wake phrase detector
│   │   ├── vad/silero_vad.py         # silence detection
│   │   ├── embeddings/ollama_embedder.py  # nomic-embed-text via Ollama
│   │   ├── memory/sqlite_vec_repository.py # SQLite + sqlite-vec store
│   │   └── audio/io.py               # sounddevice mic + speaker
│   │
│   ├── interfaces/                   # FastAPI routes (the brain's HTTP surface)
│   │   ├── converse_route.py         # POST /converse
│   │   ├── transcript_route.py       # GET  /transcript/{sid}
│   │   └── dashboard_route.py        # app console, manifest, logo, TTS Studio
│   │
│   └── schemas/                      # Pydantic request/response DTOs
│       └── converse.py               # ConverseRequest, ConverseResponse
│
├── tests/
│   ├── unit/                         # domain + application — no models loaded
│   └── integration/                  # spin up Ollama + faster-whisper, real I/O
│
├── assets/
│   └── Synthesis.png                 # main logo / visual identity
├── scripts/
│   ├── synthesis_visualizer.py       # pygame visualizer
│   ├── install_custom_wake_word.py   # custom openWakeWord installer
│   ├── memory.py                     # CLI: stats / list / search / forget / wipe
│   └── smoke_memory.py               # manual recall sanity check
├── data/                             # gitignored: synthesis_memory.db lives here
├── models/                           # downloaded model weights (gitignored)
├── .env                              # OLLAMA_URL, model names, ports
├── .env.example                      # documents required env vars
├── requirements.txt
└── README.md
```

**Architectural rule:** `domain/` never imports from `infrastructure/`. Adapters are swappable — swapping Ollama for Claude API is a one-file change in `infrastructure/llm/`.

## Running v0

From the repo root, use the project virtual environment. The brain must be
running before the daemon can talk and speak.

### 1. Prepare the environment

```bash
# Create the virtualenv if it does not already exist
uv venv --python 3.12

# Install runtime + test dependencies, including local MLX-Audio Kokoro TTS
uv pip install --python .venv/bin/python -e ".[test,kokoro]"
```

### 2. Start Ollama

Synthesis uses Ollama for the local LLM. Make sure Ollama is running and the
configured model exists:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### 3. Run the brain

Terminal 1:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check the text API:

```bash
curl -X POST http://127.0.0.1:8000/converse \
  -H "Content-Type: application/json" \
  -d '{"transcript":"hello","session_id":"manual-test"}'
```

The Synthesis app console is available at `http://127.0.0.1:8000`. It shows
local conversation sessions, refreshes the voice transcript, and includes a
text composer for talking to the local brain without using the microphone.
It also includes **TTS Studio**, a voice-only test surface that speaks typed
text through the configured local TTS engine and animates while audio is
playing. The app uses `assets/Synthesis.png` as its logo and installable web
app icon.

### 4. Run the voice daemon

Terminal 2:

```bash
.venv/bin/python -m app.daemon
```

Then say the wake word and start talking. The daemon handles microphone input,
speech-to-text, the brain request, and local text-to-speech output. It also
opens the pygame visualizer automatically by default, so you should see the
Synthesis window appear after startup.

For lower latency, the daemon warms the Whisper model at startup and logs per-turn
timings for capture, STT, brain, and TTS. Tune `WHISPER_MODEL`,
`WAKE_PRE_ROLL_MS`, `VAD_SILENCE_MS`, and `NO_SPEECH_TIMEOUT_S` in `.env` if
you want to trade speed against accuracy or cutoff tolerance. The wake chime is
off by default (`WAKE_CHIME_ENABLED=false`) so the wake-to-listen handoff is as
fast as possible; set it to `true` if you prefer an audible confirmation.

Kokoro is the default voice engine. To configure it explicitly, set:

```bash
TTS_ENGINE=kokoro
KOKORO_MODEL=mlx-community/Kokoro-82M-bf16
KOKORO_VOICE=af_heart
```

Kokoro runs locally through MLX-Audio and plays through `sounddevice`. If
MLX-Audio is missing or synthesis fails, the daemon logs the issue and falls
back to macOS `say` for that reply.

### 5. Visualizer

The daemon starts a local event bus on `127.0.0.1:8765` and launches the
pygame visualizer automatically by default. The visualizer runs in bus-only
mode when launched by the daemon, so it does not compete for the microphone.
During `SPEAKING`, it smoothly eases from the idle/listening wave into a
DNA-like animation, then eases back to the wave when Synthesis finishes.

Visualizer settings:

```bash
VISUALIZER_BUS_ENABLED=true
VISUALIZER_BUS_HOST=127.0.0.1
VISUALIZER_BUS_PORT=8765
VISUALIZER_AUTO_START=true
```

To stop the daemon from opening pygame automatically:

```bash
VISUALIZER_AUTO_START=false
```

Manual modes:

```bash
.venv/bin/python scripts/synthesis_visualizer.py --no-mic   # bus only
.venv/bin/python scripts/synthesis_visualizer.py --no-bus   # standalone mic demo
```

### 6. Wake word options

Default wake detection is openWakeWord:

```bash
WAKE_ENGINE=openwakeword
WAKE_MODEL=hey_jarvis
```

For "Hey Synthesis" without training a custom `.onnx`, use the Whisper wake
backend:

```bash
WAKE_ENGINE=whisper
WAKE_PHRASE=hey synthesis
WAKE_WHISPER_MODEL=tiny.en
WAKE_WHISPER_WINDOW_MS=1200
WAKE_WHISPER_POLL_MS=350
```

Whisper wake is more flexible, while openWakeWord is lighter and faster once
you have a matching model.

### 7. (Optional) Train a custom "Hey Synthesis" wake word

Out of the box the daemon listens for **"Hey Jarvis"** because that's the only
on-brand phrase openWakeWord ships pretrained. To make it respond to literally
"Hey Synthesis," train an openWakeWord `.onnx` model — see
[docs/wake_word_training.md](docs/wake_word_training.md). After training:

```bash
.venv/bin/python scripts/install_custom_wake_word.py ~/Downloads/hey_synthesis.onnx --update-env
```

The script validates the file with openWakeWord, copies it to
`models/wake/hey_synthesis.onnx`, and sets
`WAKE_MODEL=models/wake/hey_synthesis.onnx` in your `.env`.
`WAKE_MODEL` accepts either a
pretrained name or a path; the daemon validates and surfaces helpful errors
for either.

### 8. Long-term memory (semantic recall)

Synthesis can remember past turn-pairs and silently surface the relevant
ones the next time you bring up a related topic. The flow:

```
user transcript ──► nomic-embed-text (Ollama) ──► top-K vector search
                                                       │
                                                       ▼
                                       <memory> block injected as a
                                       SYSTEM message just before the
                                       user turn, then the LLM replies.
```

After the reply is sent, the (user, assistant) pair is embedded and
written to `data/synthesis_memory.db` in the background — your turn
isn't blocked on storage.

Off by default while iterating. To enable:

```bash
ollama pull nomic-embed-text                          # ~270 MB, one-time
.venv/bin/python scripts/smoke_memory.py              # sanity-check recall
# then set MEMORY_ENABLED=true in your .env
```

Inspect, search, or forget memories with the CLI:

```bash
.venv/bin/python -m scripts.memory stats
.venv/bin/python -m scripts.memory list --limit 20
.venv/bin/python -m scripts.memory search "favorite color" --top-k 5
.venv/bin/python -m scripts.memory show <id>
.venv/bin/python -m scripts.memory forget <id>
.venv/bin/python -m scripts.memory wipe --yes
```

Failure modes degrade gracefully: if Ollama or sqlite-vec is unavailable,
the brain logs a warning and serves turns without memory injection — the
voice loop never breaks because of memory.

Tunables in `.env`:

```bash
MEMORY_ENABLED=true
MEMORY_DB_PATH=data/synthesis_memory.db
MEMORY_EMBED_MODEL=nomic-embed-text
MEMORY_EMBED_DIM=768
MEMORY_RECALL_TOP_K=3
MEMORY_RECALL_THRESHOLD=0.65
```

### 9. Run tests

```bash
.venv/bin/python -m pytest tests/unit
```

## Security Notes

By default, the brain accepts requests only from loopback clients such as
`127.0.0.1`, `::1`, or `localhost`. If you intentionally expose it beyond your
machine, set `BRAIN_API_TOKEN` and send it as a bearer token:

```bash
curl -H "Authorization: Bearer $BRAIN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"transcript":"hello","session_id":"manual-test"}' \
  http://127.0.0.1:8000/converse
```

Conversation text is not written to daemon logs unless
`LOG_CONVERSATION_TEXT=true` is set. Keep that off for normal use; transcripts
can contain sensitive personal audio and assistant replies.

## Roadmap

- **v0** — Local voice loop, app console, TTS Studio, and pygame visualizer _(Done)_
- **v0.5** _(in progress)_ —
  - **Semantic memory** — SQLite + `sqlite-vec`, Ollama `nomic-embed-text` embeddings, always-on threshold-gated recall, CLI inspection _(Done)_
  - Streaming TTS — separate spec, not yet started
  - Interruption / barge-in — separate spec, depends on streaming TTS
  - Packaging polish — separate spec, last in the v0.5 sequence
- **v0.6** — App-console memory panel; auto-summarization layer over turn-pair recall
- **v1** — Tool use (calendar, email, web search); ship a pretrained `hey_synthesis.onnx` in the repo
- **v1.5** — Personality layer / system-prompt persona
- **v2** — Computer control (open apps, run scripts), vision input
- **v2.5** — Native macOS menu-bar surface

## License

See [LICENSE](LICENSE).
