from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(frozen=True)
class TurnPair:
    """A completed user/assistant exchange — the unit of long-term memory."""

    id: str
    session_id: str
    user_text: str
    assistant_text: str
    created_at: datetime

    @classmethod
    def new(cls, session_id: str, user_text: str, assistant_text: str) -> "TurnPair":
        return cls(
            id=uuid4().hex,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            created_at=datetime.now(UTC),
        )

    def embed_text(self) -> str:
        """Canonical text used to embed this turn-pair."""
        return f"User: {self.user_text}\nAssistant: {self.assistant_text}"
