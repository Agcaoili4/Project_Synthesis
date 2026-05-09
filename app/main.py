"""FastAPI entrypoint for the local Synthesis brain and dashboard."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.application.background import BackgroundTaskRunner
from app.application.converse import (
    ConversationStore,
    ConverseUseCase,
    InMemoryConversationStore,
    LLMClient,
)
from app.application.memory import (
    Embedder,
    EmbedDimMismatch,
    MemoryRepository,
    MemoryUnavailable,
)
from app.core.security import require_local_or_token
from app.interfaces.converse_route import build_router as build_converse_router
from app.interfaces.dashboard_route import (
    DashboardTTSEngine,
    build_router as build_dashboard_router,
)
from app.interfaces.transcript_route import build_router as build_transcript_router

log = logging.getLogger("synthesis.main")


def build_app(
    llm: LLMClient,
    store: ConversationStore | None = None,
    system_prompt: str | None = None,
    api_token: str | None = None,
    dashboard_tts_factory: Callable[[], DashboardTTSEngine] | None = None,
    embedder: Embedder | None = None,
    memory: MemoryRepository | None = None,
    memory_init: Callable[[], Awaitable[MemoryRepository]] | None = None,
    background_tasks: BackgroundTaskRunner | None = None,
    recall_top_k: int = 3,
    recall_threshold: float = 0.65,
) -> FastAPI:
    """Build the app with injectable dependencies for production and tests.

    Tests pass a ready ``memory`` instance directly. Production passes
    ``memory_init`` — an async factory that runs inside the FastAPI lifespan
    startup, where an event loop is already available. This avoids
    ``asyncio.run()`` at module-import time, which fails under uvicorn
    because uvicorn already has a loop running by then.
    """
    if store is None:
        store = InMemoryConversationStore()
    use_case = ConverseUseCase(
        llm=llm,
        store=store,
        system_prompt=system_prompt,
        embedder=embedder,
        memory=memory,
        recall_top_k=recall_top_k,
        recall_threshold=recall_threshold,
        background_tasks=background_tasks,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if memory_init is not None and use_case.memory is None:
            try:
                use_case.memory = await memory_init()
                log.info(
                    "memory enabled: top_k=%d threshold=%.2f",
                    use_case.recall_top_k,
                    use_case.recall_threshold,
                )
            except (MemoryUnavailable, EmbedDimMismatch) as exc:
                log.error("memory layer disabled: %s", exc)
                use_case.embedder = None
            except Exception:
                log.exception("memory layer disabled: unexpected init failure")
                use_case.embedder = None
        try:
            yield
        finally:
            if background_tasks is not None:
                await background_tasks.drain()

    app = FastAPI(title="Project Synthesis — Brain", version="0.0.1", lifespan=lifespan)

    @app.middleware("http")
    async def require_local_or_token_middleware(request: Request, call_next):
        require_local_or_token(request, api_token)
        return await call_next(request)

    app.include_router(build_converse_router(use_case))
    if isinstance(store, InMemoryConversationStore):
        app.include_router(build_transcript_router(store))
        app.include_router(
            build_dashboard_router(store, tts_engine_factory=dashboard_tts_factory)
        )
    return app


def _build_memory_sync(
    s,  # type: ignore[no-untyped-def]
) -> tuple[
    Embedder | None,
    Callable[[], Awaitable[MemoryRepository]] | None,
    BackgroundTaskRunner | None,
]:
    """Build the synchronous parts of the memory layer at import time.

    Returns ``(embedder, memory_init, runner)``. The ``memory_init``
    coroutine factory is awaited inside the FastAPI lifespan, where an
    event loop already exists. If ``MEMORY_ENABLED`` is false, returns
    all-None and the use case stays bit-identical to the v0 path.
    """
    if not s.memory_enabled:
        return None, None, None

    from app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
    from app.infrastructure.memory.sqlite_vec_repository import (
        SqliteVecMemoryRepository,
    )

    embedder = OllamaEmbedder(
        base_url=s.ollama_url,
        model=s.memory_embed_model,
        dim=s.memory_embed_dim,
    )

    async def _init() -> MemoryRepository:
        repo = await SqliteVecMemoryRepository.create(
            db_path=s.memory_db_path,
            embed_model=s.memory_embed_model,
            embed_dim=s.memory_embed_dim,
        )
        log.info(
            "memory layer mounted: db=%s model=%s dim=%d",
            s.memory_db_path,
            s.memory_embed_model,
            s.memory_embed_dim,
        )
        return repo

    runner = BackgroundTaskRunner()
    return embedder, _init, runner


def _configure_synthesis_logging() -> None:
    """Make `synthesis.*` loggers visible under uvicorn.

    uvicorn configures its own loggers but leaves application loggers
    untouched, so without this our INFO/WARNING lines (memory mount,
    recall hits, store failures) silently disappear into the void.
    """
    syn = logging.getLogger("synthesis")
    syn.setLevel(logging.INFO)
    if not syn.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        syn.addHandler(h)
        syn.propagate = False


def _build_production_app() -> FastAPI:
    _configure_synthesis_logging()
    from app.core.config import get_settings
    from app.infrastructure.llm.ollama_client import OllamaClient

    s = get_settings()
    llm = OllamaClient(base_url=s.ollama_url, model=s.ollama_model)
    embedder, memory_init, background_tasks = _build_memory_sync(s)
    return build_app(
        llm=llm,
        system_prompt=s.system_prompt,
        api_token=s.brain_api_token,
        embedder=embedder,
        memory_init=memory_init,
        background_tasks=background_tasks,
        recall_top_k=s.memory_recall_top_k,
        recall_threshold=s.memory_recall_threshold,
    )


app = _build_production_app()
