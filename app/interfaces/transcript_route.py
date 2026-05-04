from fastapi import APIRouter, HTTPException

from app.application.converse import InMemoryConversationStore
from app.schemas.converse import TranscriptMessage, TranscriptResponse


def build_router(store: InMemoryConversationStore) -> APIRouter:
    router = APIRouter()

    @router.get("/transcript/{session_id}", response_model=TranscriptResponse)
    def get_transcript(session_id: str) -> TranscriptResponse:
        if session_id not in store.all_sessions():
            raise HTTPException(status_code=404, detail="session not found")
        conv = store.get_or_create(session_id)
        return TranscriptResponse(
            session_id=session_id,
            messages=[
                TranscriptMessage(
                    role=m.role.value,
                    content=m.content,
                    created_at=m.created_at.isoformat(),
                )
                for m in conv.messages
            ],
        )

    @router.get("/sessions")
    def list_sessions() -> dict[str, list[str]]:
        return {"sessions": store.all_sessions()}

    return router
