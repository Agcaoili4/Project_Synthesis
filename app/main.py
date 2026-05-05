"""The BRAIN for Project Synthesis — FastAPI app exposing /converse + dashboard."""

from typing import Callable

from fastapi import FastAPI, Request

from app.application.converse import (
    ConversationStore,
    ConverseUseCase,
    InMemoryConversationStore,
    LLMClient,
)
from app.core.security import require_local_or_token
from app.interfaces.converse_route import build_router as build_converse_router
from app.interfaces.dashboard_route import (
    DashboardTTSEngine,
    build_router as build_dashboard_router,
)
from app.interfaces.transcript_route import build_router as build_transcript_router


def build_app(
    llm: LLMClient,
    store: ConversationStore | None = None,
    system_prompt: str | None = None,
    api_token: str | None = None,
    dashboard_tts_factory: Callable[[], DashboardTTSEngine] | None = None,
) -> FastAPI:
    """Factory used by both production wiring and tests.

    Tests inject a FakeLLM + fresh InMemoryConversationStore.
    Production wiring at the bottom calls build_app(llm=OllamaClient(...)).
    """
    if store is None:
        store = InMemoryConversationStore()
    use_case = ConverseUseCase(llm=llm, store=store, system_prompt=system_prompt)

    app = FastAPI(title="Project Synthesis — Brain", version="0.0.1")

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


def _build_production_app() -> FastAPI:
    from app.core.config import get_settings
    from app.infrastructure.llm.ollama_client import OllamaClient

    s = get_settings()
    llm = OllamaClient(base_url=s.ollama_url, model=s.ollama_model)
    return build_app(
        llm=llm,
        system_prompt=s.system_prompt,
        api_token=s.brain_api_token,
    )


app = _build_production_app()
