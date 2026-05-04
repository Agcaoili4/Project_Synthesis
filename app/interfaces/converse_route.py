from fastapi import APIRouter

from app.application.converse import ConverseUseCase
from app.schemas.converse import ConverseRequest, ConverseResponse


def build_router(use_case: ConverseUseCase) -> APIRouter:
    router = APIRouter()

    @router.post("/converse", response_model=ConverseResponse)
    async def converse(req: ConverseRequest) -> ConverseResponse:
        reply = await use_case.execute(
            transcript=req.transcript, session_id=req.session_id
        )
        return ConverseResponse(reply=reply, session_id=req.session_id)

    return router
