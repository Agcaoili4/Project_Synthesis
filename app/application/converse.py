from dataclasses import dataclass
from typing import Protocol

from app.domain.conversation import Conversation, Role


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

    async def execute(self, transcript: str, session_id: str) -> str:
        conv = self.store.get_or_create(session_id)
        if self.system_prompt and not any(
            m.role is Role.SYSTEM for m in conv.messages
        ):
            conv.append(Role.SYSTEM, self.system_prompt)
        conv.append_user(transcript)
        reply = await self.llm.chat(conv.to_chat_messages())
        conv.append_assistant(reply)
        self.store.save(conv)
        return reply
