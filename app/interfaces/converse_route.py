import logging

from fastapi import APIRouter, HTTPException

from app.application.converse import ConverseUseCase
from app.schemas.converse import ConverseRequest, ConverseResponse


log = logging.getLogger("synthesis.brain")


def build_router(use_case: ConverseUseCase) -> APIRouter:
    router = APIRouter()

    @router.post("/converse", response_model=ConverseResponse)
    async def converse(req: ConverseRequest) -> ConverseResponse:
        try:
            reply = await use_case.execute(
                transcript=req.transcript, session_id=req.session_id
            )
        except Exception:
            log.exception("converse failed while calling the LLM backend")
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM backend unavailable. Check that Ollama is running "
                    "and the configured model is installed."
                ),
            ) from None
        return ConverseResponse(reply=reply, session_id=req.session_id)

    return router
