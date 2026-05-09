# Semantic Memory — Design Spec

**Project:** Project Synthesis
**Version target:** v0.5 (memory subsystem only; streaming TTS, barge-in, packaging
are separate specs)
**Date:** 2026-05-09
**Author:** Jansen Agcaoili (with Claude as collaborator)

## Why this exists

Synthesis v0 ships a fully-local voice loop on Apple Silicon: wake → STT →
LLM → TTS. Conversation context lives only in a rolling 20-message
in-memory window per session — when the brain restarts, that context is
gone, and there is no continuity between sessions.

Plain conversation persistence (write every turn to SQLite, restart-survive)
fixes the second problem but does not change what the LLM *knows* on any
given turn. The behavior shift Jansen actually wants — and the headline
v0.5 win — is **semantic recall of past turn-pairs**: when today's question
relates to a discussion from last Tuesday, Synthesis silently surfaces
that discussion to the LLM as additional context, and answers with
continuity.

This spec covers only the semantic-memory subsystem of v0.5. The other
three v0.5 items (streaming TTS, barge-in, packaging polish) each get
their own spec because they have different risk profiles, different
tests, and mostly different files.

## Decisions locked during brainstorming

| Decision | Choice | Why |
| --- | --- | --- |
| Granularity | Per turn-pair (user + assistant text in one row) | Simplest schema; no extra LLM summarization pass per session. |
| Embedder | Ollama `nomic-embed-text` (768-d) | Zero new pip deps; Apple-Silicon-tuned; no extra resident model in-process. |
| Vector store | SQLite + `sqlite-vec` extension | Single-file DB, no server, MIT-licensed, ships as wheel on macOS arm64. |
| Recall trigger | Always-on, similarity-threshold gated (default 0.65) | Always-on delivers the JARVIS-y "Synthesis just knows" feel; threshold avoids confabulating links from weak matches. |
| Inspection UX | CLI for v0.5; app-console panel deferred to v0.6 | Right blast radius for a new system; observe behavior before designing UI. |
| Configuration | New keys in `.env.example` only; never edit `.env` | Per repo convention; Jansen owns and edits `.env` himself. |
| Default state | `MEMORY_ENABLED=false` | Lets the spine merge cleanly; flip on after smoke test. |

## Out of scope (deferred)

- Streaming TTS, barge-in, packaging polish (the rest of v0.5).
- App-console memory panel (v0.6).
- Voice-driven forgetting ("Synthesis, forget that") — too easy to misfire pre-persona.
- Auto-summarization / session summaries.
- Conversation-store persistence across brain restart (different feature; turn-pairs survive but the rolling in-flight window does not).
- Auto-TTL / expiry.
- Cross-process write contention; multi-user.

## Architecture

The brain is clean DDD: `domain/` (pure) → `application/` (use cases +
ports) → `infrastructure/` (adapters) → `interfaces/` (FastAPI routes).
Memory slots in by adding **two new ports + two new adapters**, with
`ConverseUseCase` as the orchestration point.

The existing `InMemoryConversationStore` stays untouched. It owns the
rolling in-flight window. The new `MemoryRepository` is the long-term
store. They answer different questions ("what was just said this turn"
vs "what past turn-pairs are relevant to this query") and intentionally
do not share storage.

```
        ┌──────────────────────────────────────────────────────┐
        │ ConverseUseCase.execute(transcript, session_id)      │
        │                                                      │
        │  1. conv = store.get_or_create(session_id)           │
        │  2. conv.append_user(transcript)                     │
        │  3. q = await embedder.embed(transcript)             │   memory layer
        │  4. hits = await memory.recall(q, k, threshold)      ├────────┐
        │  5. if hits: chat_messages.insert(-1, <memory>)      │        │
        │  6. reply = await llm.chat(chat_messages)            │   ┌────▼──────────────┐
        │  7. conv.append_assistant(reply)                     │   │ Ollama embedder   │
        │  8. background.schedule(embed + remember(turn))──────┼──►│ POST /api/embed   │
        │  9. return reply                                     │   └───────────────────┘
        └──────────────────────────────────────────────────────┘   ┌───────────────────┐
                                                                   │ SQLite + vec0     │
                                                                   │ turn_pairs +      │
                                                                   │ vec_turn_pairs    │
                                                                   └───────────────────┘
```

### Components

#### Domain — `app/domain/memory.py`

Frozen `TurnPair` dataclass: `id` (uuid4 hex), `session_id`, `user_text`,
`assistant_text`, `created_at` (UTC). Pure, no I/O. Helper
`embed_text()` returns the canonical text embedded for storage:
`f"User: {user_text}\nAssistant: {assistant_text}"`.

#### Application — `app/application/memory.py`

Two `Protocol` ports:

- `Embedder.embed(text: str) -> list[float]` — async.
- `MemoryRepository` — async methods: `remember`, `recall`,
  `list_recent`, `get`, `forget`, `forget_before`, `wipe`, `stats`.

Three typed errors: `EmbedderError`, `MemoryUnavailable`,
`EmbedDimMismatch` — used by adapters to communicate degradation
opportunities cleanly to wiring.

`render_memory_block(hits)` — pure function rendering hits into the
SYSTEM-message body. Format:

```
<memory>
[YYYY-MM-DD] You said: "..."
I replied: "..."
---
[YYYY-MM-DD] You said: "..."
I replied: "..."
</memory>
```

#### Application — `app/application/background.py`

`BackgroundTaskRunner` — bounded fire-and-forget tracker for the
post-turn embed + store. Methods: `schedule(coro) -> bool` (drops with
warning if the queue exceeds `max_pending=32`), `drain(timeout=5.0)`
(awaits remaining tasks at shutdown). Wired into the FastAPI lifespan so
shutdown does not lose the most recent turn's storage.

#### Infrastructure — `app/infrastructure/embeddings/ollama_embedder.py`

`OllamaEmbedder` implements the `Embedder` Protocol. POSTs to
`{base_url}/api/embeddings` with `{"model": ..., "prompt": text}`.
Validates the returned vector length matches the configured dim and
raises `EmbedderError` with a useful message on any failure.

#### Infrastructure — `app/infrastructure/memory/sqlite_vec_repository.py`

`SqliteVecMemoryRepository` implements `MemoryRepository`. Three tables:

```sql
CREATE TABLE turn_pairs (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  user_text    TEXT NOT NULL,
  assistant_text TEXT NOT NULL,
  created_at   TEXT NOT NULL  -- ISO-8601 UTC
);
CREATE INDEX idx_turn_pairs_session ON turn_pairs(session_id);
CREATE INDEX idx_turn_pairs_created ON turn_pairs(created_at);

CREATE VIRTUAL TABLE vec_turn_pairs
  USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[768]);

CREATE TABLE memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
```

`memory_meta` records `embed_model` + `embed_dim` so we can detect a
schema mismatch at startup and refuse to mount rather than silently
mixing vector spaces.

**Cosine via L2:** sqlite-vec returns L2 distance for `FLOAT[N]`. We
L2-normalize embeddings on insert and on query — under unit length,
L2 ranking is identical to cosine ranking. The configured cosine
threshold *t* converts to an L2 distance bound
`L2_max = sqrt(2 * (1 - t))`.

**Async over sqlite:** all public methods are `async`; sync DB calls run
under `asyncio.to_thread`. A single shared connection per repo,
serialized by an `asyncio.Lock`. Connection opens with
`journal_mode=WAL` and `synchronous=NORMAL`.

### ConverseUseCase changes — `app/application/converse.py`

Dataclass gains five optional fields: `embedder`, `memory`,
`recall_top_k`, `recall_threshold`, `background_tasks`. With all five
absent, behavior is bit-identical to the v0 use case.

Per-turn flow:

1. `conv = store.get_or_create(session_id)`.
2. Append SYSTEM prompt if absent (existing behavior).
3. `conv.append_user(transcript)`.
4. `chat_messages = conv.to_chat_messages()`.
5. **Memory recall** — wrapped in try/except so any failure logs and
   returns `None`:
   - `q_emb = await embedder.embed(transcript)`
   - `hits = await memory.recall(q_emb, top_k, threshold)`
   - If hits: `chat_messages.insert(-1, {"role": "system", "content": render_memory_block(hits)})`.
6. `reply = await llm.chat(chat_messages)`.
7. `conv.append_assistant(reply)`; `store.save(conv)`.
8. **Memory store** (fire-and-forget via `background_tasks.schedule`):
   build `TurnPair`, embed `turn.embed_text()`, call
   `memory.remember(turn, embedding)`. Exceptions logged + swallowed.
9. Return `reply`.

**Critical invariant:** the `<memory>` block is injected into the
per-turn `chat_messages` list, NOT appended to the persistent
`Conversation`. `Conversation._trim()` preserves all SYSTEM messages
forever, so storing memory blocks there would accumulate them across
turns. The unit test
`test_memory_block_is_not_persisted_on_conversation` guards this.

### FastAPI wiring — `app/main.py` and `app/core/config.py`

Six new settings (`memory_enabled`, `memory_db_path`,
`memory_embed_model`, `memory_embed_dim`, `memory_recall_top_k`,
`memory_recall_threshold`).

`_build_memory_layer(s)`: if `memory_enabled`, builds embedder + repo +
runner; on `MemoryUnavailable` or `EmbedDimMismatch`, logs ERROR and
returns `(None, None, None)` so the brain still serves.

The lifespan handler awaits `runner.drain()` on shutdown.

### CLI — `scripts/memory.py`

```
python -m scripts.memory stats
python -m scripts.memory list [--since 7d|YYYY-MM-DD] [--session SID] [--limit N]
python -m scripts.memory search "query" [--top-k 5] [--threshold 0.5]
python -m scripts.memory show <id>
python -m scripts.memory forget <id>
python -m scripts.memory forget-before <when>
python -m scripts.memory wipe [--yes]
```

`forget-before` and `wipe` require typed `yes` confirmation unless
`--yes` is passed. Reads settings via `app.core.config` so DB path and
embed model match the running brain.

### Smoke — `scripts/smoke_memory.py`

Stores 5 hand-crafted turn-pairs across distinct topics into a temp DB
via the real Ollama embedding endpoint, runs 3 queries, and prints the
top-3 ranked hits with cosine similarity. Manual sanity check; not
wired into CI. Does not touch the production DB.

## Failure modes (all degrade gracefully)

| Failure | Detection | Behavior |
| --- | --- | --- |
| Ollama embed endpoint down | `httpx` exception in `OllamaEmbedder.embed` | Log WARNING; skip recall; skip store. Voice loop continues. |
| Returned vector dim ≠ configured dim | Length check after embed | Same as above + suggest `ollama pull` of correct model. |
| `sqlite-vec` extension fails to load | Exception during repo init | Log ERROR at startup with install hint; brain runs with memory disabled. |
| Existing DB built with different `embed_dim` | `memory_meta` mismatch on init | Refuse to mount; suggest `python -m scripts.memory wipe --yes`. Brain still serves. |
| DB locked / corrupt at runtime | sqlite exception | Log WARNING; that turn proceeds without recall/store. |
| Background queue saturated | `BackgroundTaskRunner.schedule` returns False | Log WARNING (turn not stored). |

**Contract:** memory is additive; voice loop correctness never depends
on memory being healthy. Every memory touchpoint inside
`ConverseUseCase` is wrapped in try/except.

## Settings (added to `.env.example` only)

```
MEMORY_ENABLED=false
MEMORY_DB_PATH=data/synthesis_memory.db
MEMORY_EMBED_MODEL=nomic-embed-text
MEMORY_EMBED_DIM=768
MEMORY_RECALL_TOP_K=3
MEMORY_RECALL_THRESHOLD=0.65
```

## Tests

**Unit — `tests/unit/test_memory_recall.py` (8 tests).**
Fakes only. Verify: no-memory path unchanged; injection placement
correct (SYSTEM message just before user turn); memory block does NOT
persist on the rolling Conversation; empty-hit case omits the block;
embedder failure does not break the turn; repo failure does not break
the turn; post-turn store schedules with canonical embed text once;
store skipped cleanly when no `BackgroundTaskRunner` is wired.

**Integration — `tests/integration/test_sqlite_vec_memory.py` (12 tests).**
Real SQLite + real `sqlite-vec`, deterministic 768-d unit vectors.
Verify: insert/get round-trip; recall ranks by similarity; threshold
filters far vectors; threshold passes near matches; top-k caps results;
forget removes both row and vector; forget-before cuts old rows;
wipe clears all; list_recent orders newest-first and filters by
session_id and `since`; stats reports real counts and metadata;
dim-mismatch refuses on reopen; persistence across reopen.

**Smoke — `scripts/smoke_memory.py`.** Manual; requires real Ollama.

## Files

**New (12):**
```
app/domain/memory.py
app/application/memory.py
app/application/background.py
app/infrastructure/embeddings/__init__.py
app/infrastructure/embeddings/ollama_embedder.py
app/infrastructure/memory/__init__.py
app/infrastructure/memory/sqlite_vec_repository.py
scripts/memory.py
scripts/smoke_memory.py
tests/unit/test_memory_recall.py
tests/integration/test_sqlite_vec_memory.py
data/.gitkeep
```

**Modified (7):**
```
app/application/converse.py
app/main.py
app/core/config.py
.env.example
.gitignore
requirements.txt
pyproject.toml
README.md
```

## Verification

End-to-end checklist run after implementation, before declaring v0.5
memory done:

1. **Dependency install** — `uv pip install --python .venv/bin/python sqlite-vec` succeeds; `ollama pull nomic-embed-text` succeeds; `curl -s http://127.0.0.1:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'` returns a 768-d vector.
2. **Unit tests** — `pytest tests/unit/test_memory_recall.py -v` all green (8/8).
3. **Integration tests** — `pytest tests/integration/test_sqlite_vec_memory.py -v` all green (12/12).
4. **No regressions** — `pytest tests/unit -v` all green (full suite).
5. **Smoke** — `python scripts/smoke_memory.py` prints top-1 hits matching the seeded topic for each of the three queries.
6. **Brain wiring (off)** — with `MEMORY_ENABLED=false`, brain starts and serves `/converse` exactly as before; no memory tables consulted.
7. **Brain wiring (on)** — with `MEMORY_ENABLED=true`, two-turn round-trip via `/converse` produces a `<memory>` block injection on the second turn (visible in INFO logs as `memory recalled N turn(s)`).
8. **Failure-mode drill** — pointing `MEMORY_EMBED_MODEL` at a non-existent Ollama model produces WARN logs but turns still complete.
9. **CLI** — `stats`, `list`, `search`, `forget`, `wipe` all behave per their help text.
10. **Voice loop** — full stack (Ollama + brain + daemon), say a memorable fact, restart brain, ask about the fact; reply demonstrates recall.

## Roadmap impact

README roadmap line for v0.5 is restructured to show memory as landed
and to list the remaining three v0.5 items as separate (still owed)
specs. v0.6 gains an "app-console memory panel" item, capturing the
inspection UX deferral.
