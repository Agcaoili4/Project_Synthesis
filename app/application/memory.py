"""Ports and errors for the long-term memory layer.

The memory layer answers a different question than ``ConversationStore``:
the store owns the rolling in-flight conversation window; the memory layer
owns the long-term, semantically searchable archive of completed turn pairs.
"""

from datetime import datetime
from typing import Protocol

from app.domain.memory import TurnPair


class EmbedderError(RuntimeError):
    """Raised when an embedding cannot be produced (transport, dim mismatch)."""


class MemoryUnavailable(RuntimeError):
    """Raised at startup when the memory backend cannot be mounted."""


class EmbedDimMismatch(RuntimeError):
    """Raised when a configured embedding dim does not match an existing DB."""


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class MemoryRepository(Protocol):
    async def remember(self, turn: TurnPair, embedding: list[float]) -> None: ...
    async def recall(
        self,
        query_embedding: list[float],
        k: int,
        threshold: float,
    ) -> list[TurnPair]: ...
    async def list_recent(
        self,
        limit: int = 50,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> list[TurnPair]: ...
    async def get(self, turn_id: str) -> TurnPair | None: ...
    async def forget(self, turn_id: str) -> bool: ...
    async def forget_before(self, cutoff: datetime) -> int: ...
    async def wipe(self) -> int: ...
    async def stats(self) -> dict[str, object]: ...


def render_memory_block(hits: list[TurnPair]) -> str:
    """Render recall hits into the SYSTEM-message body the LLM sees."""
    lines = ["<memory>"]
    for i, t in enumerate(hits):
        if i > 0:
            lines.append("---")
        date = t.created_at.strftime("%Y-%m-%d")
        lines.append(f'[{date}] You said: "{t.user_text}"')
        lines.append(f'I replied: "{t.assistant_text}"')
    lines.append("</memory>")
    return "\n".join(lines)
