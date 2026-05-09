import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.application.background import BackgroundTaskRunner
from app.application.memory import (
    Embedder,
    MemoryRepository,
    render_memory_block,
)
from app.domain.conversation import Conversation, Role
from app.domain.memory import TurnPair

log = logging.getLogger("synthesis.converse")


class LLMClient(Protocol):
    async def chat(self, messages: list[dict[str, str]]) -> str: ...


class ConversationStore(Protocol):
    def get_or_create(self, session_id: str) -> Conversation: ...
    def save(self, conversation: Conversation) -> None: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}

    def get_or_create(self, session_id: str) -> Conversation:
        if session_id not in self._store:
            self._store[session_id] = Conversation(session_id=session_id)
        return self._store[session_id]

    def save(self, conversation: Conversation) -> None:
        self._store[conversation.session_id] = conversation

    def all_sessions(self) -> list[str]:
        return list(self._store.keys())


@dataclass
class ConverseUseCase:
    llm: LLMClient
    store: ConversationStore
    system_prompt: str | None = None
    embedder: Embedder | None = None
    memory: MemoryRepository | None = None
    recall_top_k: int = 3
    recall_threshold: float = 0.65
    background_tasks: BackgroundTaskRunner | None = field(default=None)

    async def execute(self, transcript: str, session_id: str) -> str:
        conv = self.store.get_or_create(session_id)
        if self.system_prompt and not any(
            m.role is Role.SYSTEM for m in conv.messages
        ):
            conv.append(Role.SYSTEM, self.system_prompt)
        conv.append_user(transcript)

        chat_messages = conv.to_chat_messages()

        memory_block = await self._maybe_recall(transcript)
        if memory_block is not None:
            # Inject right before the just-appended user turn (last item in
            # chat_messages). Ephemeral: not stored on Conversation, so it
            # does not accumulate across turns.
            chat_messages.insert(-1, {"role": "system", "content": memory_block})

        reply = await self.llm.chat(chat_messages)
        conv.append_assistant(reply)
        self.store.save(conv)

        self._maybe_remember(transcript=transcript, reply=reply, session_id=session_id)
        return reply

    async def _maybe_recall(self, transcript: str) -> str | None:
        if self.embedder is None or self.memory is None:
            return None
        try:
            q_emb = await self.embedder.embed(transcript)
            hits = await self.memory.recall(
                q_emb, self.recall_top_k, self.recall_threshold
            )
        except Exception as exc:
            log.warning("memory recall skipped: %s", exc)
            return None
        if not hits:
            return None
        log.info("memory recalled %d turn(s)", len(hits))
        return render_memory_block(hits)

    def _maybe_remember(
        self, *, transcript: str, reply: str, session_id: str
    ) -> None:
        if (
            self.embedder is None
            or self.memory is None
            or self.background_tasks is None
        ):
            return
        turn = TurnPair.new(
            session_id=session_id,
            user_text=transcript,
            assistant_text=reply,
        )
        self.background_tasks.schedule(self._embed_and_store(turn))

    async def _embed_and_store(self, turn: TurnPair) -> None:
        assert self.embedder is not None and self.memory is not None
        try:
            embedding = await self.embedder.embed(turn.embed_text())
            await self.memory.remember(turn, embedding)
        except Exception as exc:
            log.warning("memory store skipped for turn %s: %s", turn.id, exc)
