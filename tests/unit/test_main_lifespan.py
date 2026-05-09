"""Regression tests for the brain's wiring of the memory layer.

The bug these tests guard against: prior to this change, `_build_memory_layer`
called `asyncio.run(SqliteVecMemoryRepository.create(...))` at module-import
time. Under `uvicorn app.main:app`, uvicorn already has an event loop running
by the time it imports the app module, so `asyncio.run()` raised
``RuntimeError: asyncio.run() cannot be called from a running event loop``.
The error was caught by a broad except and silently logged; memory was set
to None and the brain served turns without ever storing or recalling.

These tests reproduce that situation by running inside pytest-asyncio's
event loop and asserting the new code path tolerates it.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.application.background import BackgroundTaskRunner
from app.infrastructure.memory.sqlite_vec_repository import (
    SqliteVecMemoryRepository,
)
from app.main import _build_memory_sync, build_app


def _fake_settings(db_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        memory_enabled=True,
        ollama_url="http://127.0.0.1:11434",
        memory_db_path=db_path,
        memory_embed_model="nomic-embed-text",
        memory_embed_dim=768,
    )


async def test_build_memory_sync_does_not_call_asyncio_run(tmp_path: Path):
    """Regression: must not call asyncio.run() — that fails under uvicorn.

    pytest-asyncio runs this test inside an event loop, which is exactly the
    state uvicorn imposes at import time. If `_build_memory_sync` regressed
    to using `asyncio.run()` it would raise RuntimeError here.
    """
    db_path = str(tmp_path / "must-not-be-created.db")
    embedder, memory_init, runner = _build_memory_sync(_fake_settings(db_path))

    assert embedder is not None
    assert memory_init is not None
    assert runner is not None

    # Crucially: the DB must not have been opened yet. The repo factory is
    # deferred to lifespan startup, where a loop already exists.
    assert not os.path.exists(db_path), (
        "memory layer must not open the DB at sync init time — repo creation "
        "belongs in lifespan startup so an event loop is available"
    )


async def test_build_memory_sync_returns_none_when_disabled():
    s = SimpleNamespace(
        memory_enabled=False,
        ollama_url="http://127.0.0.1:11434",
        memory_db_path="/dev/null",
        memory_embed_model="x",
        memory_embed_dim=768,
    )
    embedder, memory_init, runner = _build_memory_sync(s)
    assert embedder is None
    assert memory_init is None
    assert runner is None


def test_build_app_lifespan_mounts_memory_repo(tmp_path: Path):
    """End-to-end: TestClient enters lifespan; memory_init must run + attach.

    Uses TestClient rather than an `async def` test so we exercise the same
    lifespan path uvicorn uses. If the bug came back, the DB would never get
    created here.
    """
    db_path = str(tmp_path / "lifespan_mount.db")

    async def mem_init() -> SqliteVecMemoryRepository:
        return await SqliteVecMemoryRepository.create(
            db_path=db_path, embed_model="fake", embed_dim=768,
        )

    class StubLLM:
        async def chat(self, messages):  # noqa: ARG002
            return "ok"

    app = build_app(
        llm=StubLLM(),
        memory_init=mem_init,
        background_tasks=BackgroundTaskRunner(),
    )

    assert not os.path.exists(db_path), "DB must not be opened before lifespan"

    with TestClient(app):
        # Entering TestClient runs the lifespan startup; memory_init must
        # have been awaited and the repo opened.
        assert os.path.exists(db_path), (
            "lifespan startup must await memory_init and open the DB; "
            "if this fails, the brain is silently running without memory"
        )


def test_build_app_lifespan_swallows_memory_init_failure(tmp_path: Path):
    """A broken memory_init must not crash the brain — log and serve."""

    async def bad_init():
        raise RuntimeError("simulated init failure")

    class StubLLM:
        async def chat(self, messages):  # noqa: ARG002
            return "ok"

    app = build_app(
        llm=StubLLM(),
        memory_init=bad_init,
        background_tasks=BackgroundTaskRunner(),
    )

    # Brain must still mount and accept the lifespan startup.
    with TestClient(app) as client:
        # /converse still works even though memory init blew up.
        resp = client.post(
            "/converse",
            json={"transcript": "hello", "session_id": "s1"},
        )
        assert resp.status_code == 200
        assert resp.json()["reply"] == "ok"


# Reference unused so pytest collects without ARG warnings.
_ = pytest
