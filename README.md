# Project Synthesis

A locally-run, voice-driven AI assistant inspired by JARVIS — lives on your Mac, listens for a wake word, holds a real conversation, and never sends your audio to the cloud.

## Status

**v0 in progress.** First milestone: a fully-local voice loop — wake word → speech-to-text → LLM → text-to-speech — on Apple Silicon with zero recurring API cost.

## Vision

Long-term, Synthesis aims to be a personal AI in the JARVIS / FRIDAY mold: ambient, conversational, tool-using (email, calendar, web, local files), with persistent memory of you. We're getting there one subsystem at a time. **v0 is the voice spine; everything else plugs into it.**

## How it works (v0)

Two cooperating processes on your machine:

- **Daemon** (`app/daemon.py`) — owns the microphone, wake word, STT, and TTS. Always-on, real-time.
- **Brain** (`app/main.py`, FastAPI) — owns the LLM, conversation state, and a localhost web dashboard. Restartable without losing the audio loop.

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
  │       local TTS ─────► speakers│       │  GET  /  (HTMX dashboard)    │
  └──────────────────────────────┘         └──────────────────────────────┘
                                                    ▲
                                                    │ http://localhost:8000
                                                    └─── browser
```

## Stack

| Concern                  | Choice                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Wake word                | openWakeWord. Default `hey_jarvis` (pretrained); custom `hey_synthesis` per [docs/wake_word_training.md](docs/wake_word_training.md) |
| Voice activity detection | Silero VAD                                                                                                                           |
| Speech-to-text           | faster-whisper (`base.en` by default; `small.en` for higher accuracy)                                                                |
| LLM                      | Qwen 2.5 7B Instruct via Ollama                                                                                                      |
| Text-to-speech           | MLX-Audio Kokoro by default (`TTS_ENGINE=kokoro`) with macOS `say` fallback.                                                          |
| Audio I/O                | sounddevice                                                                                                                          |
| Backend                  | FastAPI                                                                                                                              |
| Dashboard                | Server-rendered HTML + HTMX                                                                                                          |

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
│   │   └── logging.py                # structured logging
│   │
│   ├── domain/                       # Pure business logic. NO I/O. NO frameworks.
│   │   ├── conversation.py           # Conversation, Message, Role
│   │   └── transcript.py             # Transcript value object
│   │
│   ├── application/                  # Use cases. Orchestrates domain + infrastructure.
│   │   └── converse.py               # ConverseUseCase: takes transcript, returns reply
│   │
│   ├── infrastructure/               # Adapters to the outside world.
│   │   ├── llm/ollama_client.py      # talks to Ollama
│   │   ├── stt/whisper_engine.py     # wraps faster-whisper
│   │   ├── tts/mlx_kokoro_engine.py  # wraps MLX-Audio Kokoro local TTS
│   │   ├── tts/say_engine.py         # wraps macOS `say` fallback
│   │   ├── wake/openww_detector.py   # wraps openWakeWord
│   │   ├── vad/silero_vad.py         # silence detection
│   │   └── audio/io.py               # sounddevice mic + speaker
│   │
│   ├── interfaces/                   # FastAPI routes (the brain's HTTP surface)
│   │   ├── converse_route.py         # POST /converse
│   │   ├── transcript_route.py       # GET  /transcript/{sid}
│   │   └── dashboard_route.py        # GET  /  (HTML + HTMX)
│   │
│   └── schemas/                      # Pydantic request/response DTOs
│       └── converse.py               # ConverseRequest, ConverseResponse
│
├── tests/
│   ├── unit/                         # domain + application — no models loaded
│   └── integration/                  # spin up Ollama + faster-whisper, real I/O
│
├── models/                           # downloaded model weights (gitignored)
├── .env                              # OLLAMA_URL, model names, ports
├── .env.example                      # documents required env vars
├── requirements.txt
├── docker-compose.yml                # parked for now, used in v0.5
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

### 4. Run the voice daemon

Terminal 2:

```bash
.venv/bin/python -m app.daemon
```

Then say the wake word and start talking. The daemon handles microphone input,
speech-to-text, the brain request, and local text-to-speech output.

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

### 5. (Optional) Train a custom "Hey Synthesis" wake word

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

### 6. Run tests

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

- **v0** — Local voice loop _(in progress)_
- **v0.5** — Streaming TTS, interruption / barge-in, persistent memory (SQLite), Docker compose
- **v1** — Tool use (calendar, email, web search); ship a pretrained `hey_synthesis.onnx` in the repo
- **v1.5** — Personality layer / system-prompt persona
- **v2** — Computer control (open apps, run scripts), vision input
- **v2.5** — Native macOS menu-bar surface

## License

See [LICENSE](LICENSE).
