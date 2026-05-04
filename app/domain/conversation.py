from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    created_at: datetime


@dataclass
class Conversation:
    session_id: str
    max_history: int = 20
    messages: list[Message] = field(default_factory=list)

    def append(self, role: Role, content: str) -> Message:
        msg = Message(role=role, content=content, created_at=datetime.now(UTC))
        self.messages.append(msg)
        self._trim()
        return msg

    def append_user(self, content: str) -> Message:
        return self.append(Role.USER, content)

    def append_assistant(self, content: str) -> Message:
        return self.append(Role.ASSISTANT, content)

    def to_chat_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

    def _trim(self) -> None:
        system = [m for m in self.messages if m.role is Role.SYSTEM]
        rest = [m for m in self.messages if m.role is not Role.SYSTEM]
        if len(rest) > self.max_history:
            rest = rest[-self.max_history :]
        self.messages = system + rest
