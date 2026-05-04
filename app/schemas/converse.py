from pydantic import BaseModel, Field


class ConverseRequest(BaseModel):
    transcript: str = Field(
        min_length=1,
        max_length=8000,
        description="What the user said.",
    )
    session_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Conversation key.",
    )


class ConverseResponse(BaseModel):
    reply: str
    session_id: str


class TranscriptMessage(BaseModel):
    role: str
    content: str
    created_at: str


class TranscriptResponse(BaseModel):
    session_id: str
    messages: list[TranscriptMessage]
