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
  │       say -v Moira ──► speakers│       │  GET  /  (HTMX dashboard)    │
  └──────────────────────────────┘         └──────────────────────────────┘
                                                    ▲
                                                    │ http://localhost:8000
                                                    └─── browser
```

## Stack

| Concern                  | Choice                                                                            |
| ------------------------ | --------------------------------------------------------------------------------- |
| Wake word                | openWakeWord (`hey_synthesis`)                                                    |
| Voice activity detection | Silero VAD                                                                        |
| Speech-to-text           | faster-whisper (`small.en`)                                                       |
| LLM                      | Qwen 2.5 7B Instruct via Ollama                                                   |
| Text-to-speech           | macOS `say` (`Moira` — Irish female, FRIDAY-adjacent). Piper/MLX upgrade in v0.5. |
| Audio I/O                | sounddevice                                                                       |
| Backend                  | FastAPI                                                                           |
| Dashboard                | Server-rendered HTML + HTMX                                                       |

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
│   │   ├── tts/say_engine.py         # wraps macOS `say` (Moira voice)
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

# Install runtime + test dependencies
uv pip install --python .venv/bin/python -e ".[test]"
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

The dashboard is available at `http://127.0.0.1:8000`.

### 4. Run the voice daemon

Terminal 2:

```bash
.venv/bin/python -m app.daemon
```

Then say the wake word and start talking. The daemon handles microphone input,
speech-to-text, the brain request, and macOS `say` text-to-speech output.

### 5. Run tests

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
- **v0.5** — Upgrade TTS to Piper or MLX-Audio Kokoro, streaming output, persistent memory (SQLite), Docker compose
- **v1** — Tool use (calendar, email, web search), custom `hey_synthesis` wake word
- **v1.5** — Personality layer / system-prompt persona
- **v2** — Computer control (open apps, run scripts), vision input
- **v2.5** — Native macOS menu-bar surface

## License

See [LICENSE](LICENSE).
