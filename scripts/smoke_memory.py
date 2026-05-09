"""Manual smoke check for the memory layer end-to-end.

Hits the real local Ollama embedding endpoint, stores 5 hand-crafted
turn-pairs about distinct topics into a temporary DB, runs 3 queries, and
prints the top-3 ranked recall hits with cosine similarity. Useful to
sanity-check that ``nomic-embed-text`` is behaving (i.e. queries about a
topic surface the matching turn-pair, not a random one).

This script does NOT touch the configured production DB — it uses an
in-tmpdir SQLite file so you can run it freely.

Prereqs:
    ollama serve
    ollama pull nomic-embed-text
"""

import asyncio
import math
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.domain.memory import TurnPair
from app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
from app.infrastructure.memory.sqlite_vec_repository import SqliteVecMemoryRepository

CORPUS = [
    (
        "what TTS engine are we using on apple silicon",
        "Kokoro via MLX-Audio is the default voice path; macOS say is the fallback.",
    ),
    (
        "how does the visualizer animate during speaking",
        "Front wave fades while strands split off the centerline and twist into a DNA helix; rungs join in around mix 0.42.",
    ),
    (
        "what wake word backend are we shipping",
        "Whisper streaming detector on tiny.en with 1.2s window — no custom .onnx training needed.",
    ),
    (
        "what's my favorite color",
        "You said it was teal.",
    ),
    (
        "what's the project layout for the brain",
        "FastAPI in app/main.py with clean DDD: domain, application, infrastructure, interfaces.",
    ),
]

QUERIES = [
    "remind me about teal",
    "how is voice synthesized?",
    "what does the visualizer look like when synthesis is talking",
]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _run() -> int:
    s = get_settings()
    embedder = OllamaEmbedder(
        base_url=s.ollama_url,
        model=s.memory_embed_model,
        dim=s.memory_embed_dim,
    )
    print(f"Using ollama at {s.ollama_url} with model {s.memory_embed_model} (dim={s.memory_embed_dim})")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "smoke.db")
        repo = await SqliteVecMemoryRepository.create(
            db_path=db_path,
            embed_model=s.memory_embed_model,
            embed_dim=s.memory_embed_dim,
        )
        print(f"\nSeeding {len(CORPUS)} turn-pairs into {db_path}")
        embeds: dict[str, list[float]] = {}
        for i, (u, a) in enumerate(CORPUS):
            turn = TurnPair(
                id=f"smoke-{i}",
                session_id="smoke",
                user_text=u,
                assistant_text=a,
                created_at=datetime.now(UTC),
            )
            emb = await embedder.embed(turn.embed_text())
            embeds[turn.id] = emb
            await repo.remember(turn, emb)
            print(f"  + {turn.id}: {u[:60]}")

        for q in QUERIES:
            print(f"\nQuery: {q!r}")
            qemb = await embedder.embed(q)
            hits = await repo.recall(qemb, k=3, threshold=-1.0)
            for rank, t in enumerate(hits, start=1):
                cos = _cos(qemb, embeds[t.id])
                print(f"  #{rank}  cos={cos:+.3f}  {t.id}")
                print(f"        U: {t.user_text}")
                print(f"        A: {t.assistant_text}")

        await repo.close()
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        print(f"\nsmoke failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        print(
            "Hint: is `ollama serve` running and have you done `ollama pull nomic-embed-text`?",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
