"""SQLite + sqlite-vec backed MemoryRepository.

Two tables joined on id:
  * ``turn_pairs``      — the human-readable rows (text, session, timestamp)
  * ``vec_turn_pairs``  — sqlite-vec vec0 virtual table holding embeddings

A third table (``memory_meta``) records the embedding model + dim so we can
detect a mismatch on restart and refuse to mount, rather than silently
mixing vectors of incompatible spaces.

Embeddings are L2-normalized on insert and on query. With unit-length
vectors, L2 distance and cosine distance produce identical KNN ordering
(``L2^2 = 2 - 2*cos``), so we use L2 internally and convert the threshold:
``L2_max = sqrt(2 * (1 - threshold))``.
"""

import asyncio
import logging
import math
import os
import sqlite3
import struct
from datetime import datetime
from pathlib import Path

import sqlite_vec

from app.application.memory import EmbedDimMismatch, MemoryUnavailable
from app.domain.memory import TurnPair

log = logging.getLogger("synthesis.memory")


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _pack_floats(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _threshold_to_l2_max(threshold: float) -> float:
    """Convert cosine-similarity threshold (in [-1, 1]) to L2 distance bound.

    For unit vectors u, v: ||u - v||^2 = 2(1 - cos). So a hit at cosine >= t
    is the same row set as L2 <= sqrt(2 * (1 - t)).
    """
    t = max(-1.0, min(1.0, threshold))
    return math.sqrt(max(0.0, 2.0 * (1.0 - t)))


class SqliteVecMemoryRepository:
    """File-backed, sqlite-vec-indexed memory store. Implements MemoryRepository."""

    def __init__(
        self,
        db_path: str,
        embed_model: str,
        embed_dim: int,
    ) -> None:
        self._db_path = db_path
        self._embed_model = embed_model
        self._embed_dim = embed_dim
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    # ----- lifecycle -----

    @classmethod
    async def create(
        cls,
        db_path: str,
        embed_model: str,
        embed_dim: int,
    ) -> "SqliteVecMemoryRepository":
        repo = cls(db_path, embed_model, embed_dim)
        await asyncio.to_thread(repo._init_sync)
        return repo

    def _init_sync(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                self._db_path,
                isolation_level=None,  # autocommit; we manage txns manually
                check_same_thread=False,
            )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except sqlite3.Error as exc:
            raise MemoryUnavailable(
                f"sqlite-vec extension failed to load on {self._db_path}: {exc}"
            ) from exc

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_pairs (
                id             TEXT PRIMARY KEY,
                session_id     TEXT NOT NULL,
                user_text      TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turn_pairs_session ON turn_pairs(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turn_pairs_created ON turn_pairs(created_at)"
        )
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_turn_pairs
            USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[{self._embed_dim}])
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Dim/model gate.
        cur = conn.execute("SELECT key, value FROM memory_meta")
        meta = dict(cur.fetchall())
        if "embed_dim" in meta and int(meta["embed_dim"]) != self._embed_dim:
            raise EmbedDimMismatch(
                f"DB at {self._db_path} was built with embed_dim={meta['embed_dim']}, "
                f"but configured MEMORY_EMBED_DIM={self._embed_dim}. "
                f"Run `python -m scripts.memory wipe --yes` to rebuild."
            )
        if "embed_model" in meta and meta["embed_model"] != self._embed_model:
            log.warning(
                "embed model changed in config (db=%s, configured=%s); embeddings "
                "from different models share a vector space only if you trained that way. "
                "Consider `python -m scripts.memory wipe --yes` if recall feels off.",
                meta["embed_model"],
                self._embed_model,
            )
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta(key, value) VALUES (?, ?)",
            ("embed_model", self._embed_model),
        )
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta(key, value) VALUES (?, ?)",
            ("embed_dim", str(self._embed_dim)),
        )

        self._conn = conn

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None

    # ----- writes -----

    async def remember(self, turn: TurnPair, embedding: list[float]) -> None:
        if len(embedding) != self._embed_dim:
            raise EmbedDimMismatch(
                f"embedding dim {len(embedding)} != configured {self._embed_dim}"
            )
        normalized = _normalize(embedding)
        blob = _pack_floats(normalized)
        async with self._lock:
            await asyncio.to_thread(self._remember_sync, turn, blob)

    def _remember_sync(self, turn: TurnPair, blob: bytes) -> None:
        assert self._conn is not None
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "INSERT OR REPLACE INTO turn_pairs(id, session_id, user_text, assistant_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    turn.id,
                    turn.session_id,
                    turn.user_text,
                    turn.assistant_text,
                    turn.created_at.isoformat(),
                ),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_turn_pairs(id, embedding) VALUES (?, ?)",
                (turn.id, blob),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ----- reads -----

    async def recall(
        self, query_embedding: list[float], k: int, threshold: float
    ) -> list[TurnPair]:
        if len(query_embedding) != self._embed_dim:
            raise EmbedDimMismatch(
                f"query embedding dim {len(query_embedding)} != configured {self._embed_dim}"
            )
        normalized = _normalize(query_embedding)
        blob = _pack_floats(normalized)
        l2_max = _threshold_to_l2_max(threshold)
        async with self._lock:
            rows = await asyncio.to_thread(self._recall_sync, blob, k, l2_max)
        return [self._row_to_turn(r) for r in rows]

    def _recall_sync(
        self, blob: bytes, k: int, l2_max: float
    ) -> list[tuple[str, str, str, str, str]]:
        assert self._conn is not None
        sql = """
            SELECT t.id, t.session_id, t.user_text, t.assistant_text, t.created_at
            FROM vec_turn_pairs v
            JOIN turn_pairs t ON t.id = v.id
            WHERE v.embedding MATCH ?
              AND v.k = ?
              AND v.distance <= ?
            ORDER BY v.distance ASC
        """
        cur = self._conn.execute(sql, (blob, k, l2_max))
        return cur.fetchall()

    async def list_recent(
        self,
        limit: int = 50,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> list[TurnPair]:
        async with self._lock:
            rows = await asyncio.to_thread(
                self._list_recent_sync, limit, session_id, since
            )
        return [self._row_to_turn(r) for r in rows]

    def _list_recent_sync(
        self, limit: int, session_id: str | None, since: datetime | None
    ) -> list[tuple[str, str, str, str, str]]:
        assert self._conn is not None
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cur = self._conn.execute(
            f"SELECT id, session_id, user_text, assistant_text, created_at "
            f"FROM turn_pairs {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return cur.fetchall()

    async def get(self, turn_id: str) -> TurnPair | None:
        async with self._lock:
            row = await asyncio.to_thread(self._get_sync, turn_id)
        return self._row_to_turn(row) if row else None

    def _get_sync(
        self, turn_id: str
    ) -> tuple[str, str, str, str, str] | None:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, session_id, user_text, assistant_text, created_at "
            "FROM turn_pairs WHERE id = ?",
            (turn_id,),
        )
        return cur.fetchone()

    # ----- deletes -----

    async def forget(self, turn_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._forget_sync, turn_id)

    def _forget_sync(self, turn_id: str) -> bool:
        assert self._conn is not None
        try:
            self._conn.execute("BEGIN")
            cur = self._conn.execute("DELETE FROM turn_pairs WHERE id = ?", (turn_id,))
            removed = cur.rowcount > 0
            self._conn.execute("DELETE FROM vec_turn_pairs WHERE id = ?", (turn_id,))
            self._conn.execute("COMMIT")
            return removed
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    async def forget_before(self, cutoff: datetime) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._forget_before_sync, cutoff)

    def _forget_before_sync(self, cutoff: datetime) -> int:
        assert self._conn is not None
        try:
            self._conn.execute("BEGIN")
            cur = self._conn.execute(
                "SELECT id FROM turn_pairs WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            ids = [r[0] for r in cur.fetchall()]
            for tid in ids:
                self._conn.execute("DELETE FROM turn_pairs WHERE id = ?", (tid,))
                self._conn.execute("DELETE FROM vec_turn_pairs WHERE id = ?", (tid,))
            self._conn.execute("COMMIT")
            return len(ids)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    async def wipe(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._wipe_sync)

    def _wipe_sync(self) -> int:
        assert self._conn is not None
        try:
            self._conn.execute("BEGIN")
            cur = self._conn.execute("SELECT COUNT(*) FROM turn_pairs")
            n = cur.fetchone()[0]
            self._conn.execute("DELETE FROM turn_pairs")
            self._conn.execute("DELETE FROM vec_turn_pairs")
            self._conn.execute("COMMIT")
            return int(n)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ----- introspection -----

    async def stats(self) -> dict[str, object]:
        async with self._lock:
            return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, object]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM turn_pairs"
        )
        count, oldest, newest = cur.fetchone()
        try:
            db_bytes = os.path.getsize(self._db_path)
        except OSError:
            db_bytes = 0
        return {
            "row_count": int(count or 0),
            "oldest": oldest,
            "newest": newest,
            "db_bytes": int(db_bytes),
            "embed_model": self._embed_model,
            "embed_dim": self._embed_dim,
            "db_path": self._db_path,
        }

    # ----- helpers -----

    @staticmethod
    def _row_to_turn(row: tuple[str, str, str, str, str]) -> TurnPair:
        return TurnPair(
            id=row[0],
            session_id=row[1],
            user_text=row[2],
            assistant_text=row[3],
            created_at=datetime.fromisoformat(row[4]),
        )


__all__ = ["SqliteVecMemoryRepository"]
