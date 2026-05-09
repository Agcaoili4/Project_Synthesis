"""Integration tests for the real sqlite-vec backed memory repository.

Real SQLite + real sqlite-vec extension (CI-friendly: extension is a wheel,
no native build). Embeddings are deterministic 768-d unit vectors so we can
assert ranking exactly. No Ollama required.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.application.memory import EmbedDimMismatch
from app.domain.memory import TurnPair
from app.infrastructure.memory.sqlite_vec_repository import SqliteVecMemoryRepository

DIM = 768


def _onehot(idx: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


def _mix(idx_a: int, idx_b: int, w_a: float, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[idx_a % dim] = w_a
    v[idx_b % dim] = 1.0 - w_a
    return v


def _turn(i: int, session: str = "s1", user: str = "u", asst: str = "a") -> TurnPair:
    return TurnPair(
        id=f"t-{i}",
        session_id=session,
        user_text=f"{user}-{i}",
        assistant_text=f"{asst}-{i}",
        created_at=datetime(2026, 4, 1, tzinfo=UTC) + timedelta(minutes=i),
    )


@pytest.fixture
async def repo(tmp_path: Path) -> SqliteVecMemoryRepository:
    db = tmp_path / "mem.db"
    return await SqliteVecMemoryRepository.create(
        db_path=str(db),
        embed_model="fake-test",
        embed_dim=DIM,
    )


async def test_remember_and_get_roundtrip(repo):
    t = _turn(0)
    await repo.remember(t, _onehot(0))
    got = await repo.get(t.id)
    assert got is not None
    assert got.id == t.id
    assert got.user_text == t.user_text
    assert got.assistant_text == t.assistant_text
    assert got.session_id == t.session_id


async def test_recall_ranks_by_similarity(repo):
    # Three turns, each anchored to a distinct one-hot direction.
    for i in range(3):
        await repo.remember(_turn(i), _onehot(i))
    # Query close to direction 1.
    hits = await repo.recall(_onehot(1), k=3, threshold=-1.0)
    assert [h.id for h in hits] == ["t-1", "t-0", "t-2"] or hits[0].id == "t-1"
    assert hits[0].id == "t-1"


async def test_recall_threshold_filters_out_far_vectors(repo):
    await repo.remember(_turn(0), _onehot(0))
    await repo.remember(_turn(1), _onehot(1))
    # Query is dimension-7, orthogonal to both stored vectors (cos=0).
    # threshold=0.5 => requires cos >= 0.5 => orthogonal hits must be filtered.
    hits = await repo.recall(_onehot(7), k=5, threshold=0.5)
    assert hits == []


async def test_recall_threshold_lets_near_matches_through(repo):
    await repo.remember(_turn(0), _onehot(0))
    # Query is mostly aligned with stored vector (cos≈0.9).
    near = _mix(0, 100, 0.9)
    hits = await repo.recall(near, k=1, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].id == "t-0"


async def test_recall_top_k_caps_results(repo):
    for i in range(5):
        await repo.remember(_turn(i), _onehot(i))
    hits = await repo.recall(_onehot(0), k=2, threshold=-1.0)
    assert len(hits) == 2


async def test_forget_removes_row_and_vector(repo):
    await repo.remember(_turn(0), _onehot(0))
    assert await repo.forget("t-0") is True
    assert await repo.get("t-0") is None
    assert await repo.recall(_onehot(0), k=5, threshold=-1.0) == []


async def test_forget_before_cuts_old_rows(repo):
    for i in range(5):
        await repo.remember(_turn(i), _onehot(i))
    # All turns are at 2026-04-01 + i minutes. Cut at minute 3.
    cutoff = datetime(2026, 4, 1, 0, 3, tzinfo=UTC)
    n = await repo.forget_before(cutoff)
    assert n == 3  # turns 0, 1, 2 removed
    remaining = await repo.list_recent(limit=10)
    assert {t.id for t in remaining} == {"t-3", "t-4"}


async def test_wipe_clears_all(repo):
    for i in range(3):
        await repo.remember(_turn(i), _onehot(i))
    assert await repo.wipe() == 3
    assert await repo.list_recent(limit=10) == []
    assert await repo.recall(_onehot(0), k=5, threshold=-1.0) == []


async def test_list_recent_orders_newest_first_and_filters(repo):
    await repo.remember(_turn(0, session="alpha"), _onehot(0))
    await repo.remember(_turn(1, session="beta"), _onehot(1))
    await repo.remember(_turn(2, session="alpha"), _onehot(2))

    all_recent = await repo.list_recent(limit=10)
    assert [t.id for t in all_recent] == ["t-2", "t-1", "t-0"]

    only_alpha = await repo.list_recent(limit=10, session_id="alpha")
    assert {t.id for t in only_alpha} == {"t-0", "t-2"}

    cutoff = datetime(2026, 4, 1, 0, 0, 30, tzinfo=UTC)
    since = await repo.list_recent(limit=10, since=cutoff)
    assert {t.id for t in since} == {"t-1", "t-2"}


async def test_stats_reports_real_counts(repo):
    await repo.remember(_turn(0), _onehot(0))
    await repo.remember(_turn(1), _onehot(1))
    s = await repo.stats()
    assert s["row_count"] == 2
    assert s["embed_dim"] == DIM
    assert s["embed_model"] == "fake-test"
    assert s["oldest"] is not None and s["newest"] is not None


async def test_dim_mismatch_refuses_on_reopen(tmp_path: Path):
    db = tmp_path / "mem.db"
    repo = await SqliteVecMemoryRepository.create(
        db_path=str(db), embed_model="fake-test", embed_dim=DIM
    )
    await repo.remember(_turn(0), _onehot(0))
    await repo.close()

    with pytest.raises(EmbedDimMismatch):
        await SqliteVecMemoryRepository.create(
            db_path=str(db), embed_model="fake-test", embed_dim=384
        )


async def test_persistence_across_reopen(tmp_path: Path):
    db = tmp_path / "mem.db"
    repo = await SqliteVecMemoryRepository.create(
        db_path=str(db), embed_model="fake-test", embed_dim=DIM
    )
    await repo.remember(_turn(0), _onehot(0))
    await repo.close()

    repo2 = await SqliteVecMemoryRepository.create(
        db_path=str(db), embed_model="fake-test", embed_dim=DIM
    )
    hits = await repo2.recall(_onehot(0), k=1, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].id == "t-0"
    await repo2.close()
