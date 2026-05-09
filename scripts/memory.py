"""CLI for inspecting and managing the long-term memory database.

Usage:
    python -m scripts.memory stats
    python -m scripts.memory list [--since 7d|YYYY-MM-DD] [--session SID] [--limit N]
    python -m scripts.memory search "query" [--top-k 5] [--threshold 0.6]
    python -m scripts.memory show <id>
    python -m scripts.memory forget <id>
    python -m scripts.memory forget-before YYYY-MM-DD
    python -m scripts.memory wipe [--yes]

Reads settings from app.core.config (so DB path + embed model match the
running brain). `search` requires Ollama to be reachable; everything else
operates purely on the local SQLite file.
"""

import argparse
import asyncio
import re
import sys
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
from app.infrastructure.memory.sqlite_vec_repository import SqliteVecMemoryRepository


def _parse_since(value: str) -> datetime:
    m = re.fullmatch(r"(\d+)([dhm])", value)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
        return datetime.now(UTC) - delta
    try:
        d = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"--since accepts '7d', '12h', '30m', or YYYY-MM-DD; got {value!r}") from exc
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _build_repo() -> SqliteVecMemoryRepository:
    s = get_settings()
    return asyncio.run(
        SqliteVecMemoryRepository.create(
            db_path=s.memory_db_path,
            embed_model=s.memory_embed_model,
            embed_dim=s.memory_embed_dim,
        )
    )


def _build_embedder() -> OllamaEmbedder:
    s = get_settings()
    return OllamaEmbedder(
        base_url=s.ollama_url,
        model=s.memory_embed_model,
        dim=s.memory_embed_dim,
    )


def _short(text: str, n: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def cmd_stats() -> None:
    repo = _build_repo()
    s = asyncio.run(repo.stats())
    for k, v in s.items():
        print(f"{k:14}: {v}")


def cmd_list(args: argparse.Namespace) -> None:
    repo = _build_repo()
    since = _parse_since(args.since) if args.since else None
    rows = asyncio.run(
        repo.list_recent(limit=args.limit, session_id=args.session, since=since)
    )
    if not rows:
        print("(no rows)")
        return
    for t in rows:
        ts = t.created_at.strftime("%Y-%m-%d %H:%M")
        print(f"{t.id}  {ts}  [{t.session_id}]")
        print(f"  U: {_short(t.user_text)}")
        print(f"  A: {_short(t.assistant_text)}")


def cmd_search(args: argparse.Namespace) -> None:
    repo = _build_repo()
    embedder = _build_embedder()
    emb = asyncio.run(embedder.embed(args.query))
    hits = asyncio.run(repo.recall(emb, k=args.top_k, threshold=args.threshold))
    if not hits:
        print("(no hits above threshold)")
        return
    for i, t in enumerate(hits, start=1):
        ts = t.created_at.strftime("%Y-%m-%d %H:%M")
        print(f"#{i}  {t.id}  {ts}  [{t.session_id}]")
        print(f"  U: {_short(t.user_text, 120)}")
        print(f"  A: {_short(t.assistant_text, 120)}")


def cmd_show(args: argparse.Namespace) -> None:
    repo = _build_repo()
    t = asyncio.run(repo.get(args.id))
    if t is None:
        raise SystemExit(f"no such turn: {args.id}")
    print(f"id         : {t.id}")
    print(f"session_id : {t.session_id}")
    print(f"created_at : {t.created_at.isoformat()}")
    print("user_text  :")
    print(t.user_text)
    print("assistant_text:")
    print(t.assistant_text)


def cmd_forget(args: argparse.Namespace) -> None:
    repo = _build_repo()
    ok = asyncio.run(repo.forget(args.id))
    print("removed" if ok else "no such turn")


def cmd_forget_before(args: argparse.Namespace) -> None:
    cutoff = _parse_since(args.cutoff)
    if not args.yes:
        ans = input(f"Forget all rows before {cutoff.isoformat()}? [type 'yes']: ")
        if ans.strip().lower() != "yes":
            print("aborted")
            return
    repo = _build_repo()
    n = asyncio.run(repo.forget_before(cutoff))
    print(f"removed {n} rows")


def cmd_wipe(args: argparse.Namespace) -> None:
    if not args.yes:
        ans = input("Wipe ALL memory rows? [type 'yes']: ")
        if ans.strip().lower() != "yes":
            print("aborted")
            return
    repo = _build_repo()
    n = asyncio.run(repo.wipe())
    print(f"wiped {n} rows")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.memory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="show row count, span, db size")

    p_list = sub.add_parser("list", help="list recent turn-pairs")
    p_list.add_argument("--since", default=None, help="e.g. 7d, 12h, 30m, or YYYY-MM-DD")
    p_list.add_argument("--session", default=None)
    p_list.add_argument("--limit", type=int, default=20)

    p_search = sub.add_parser("search", help="semantic search (requires Ollama)")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--threshold", type=float, default=0.5)

    p_show = sub.add_parser("show", help="print one turn-pair in full")
    p_show.add_argument("id")

    p_forget = sub.add_parser("forget", help="delete one turn-pair by id")
    p_forget.add_argument("id")

    p_fb = sub.add_parser("forget-before", help="delete rows older than cutoff")
    p_fb.add_argument("cutoff", help="e.g. 30d, 7d, or YYYY-MM-DD")
    p_fb.add_argument("--yes", action="store_true")

    p_wipe = sub.add_parser("wipe", help="delete every row")
    p_wipe.add_argument("--yes", action="store_true")

    args = p.parse_args(argv)
    {
        "stats": lambda: cmd_stats(),
        "list": lambda: cmd_list(args),
        "search": lambda: cmd_search(args),
        "show": lambda: cmd_show(args),
        "forget": lambda: cmd_forget(args),
        "forget-before": lambda: cmd_forget_before(args),
        "wipe": lambda: cmd_wipe(args),
    }[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
