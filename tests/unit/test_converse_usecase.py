import pytest

from app.application.converse import (
    ConverseUseCase,
    InMemoryConversationStore,
)
from app.domain.conversation import Role


class FakeLLM:
    """Records the messages it was called with; returns a canned reply."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


async def test_execute_returns_llm_reply(store):
    llm = FakeLLM(reply="Paris")
    uc = ConverseUseCase(llm=llm, store=store)
    reply = await uc.execute(transcript="capital of France?", session_id="s1")
    assert reply == "Paris"


async def test_execute_appends_user_transcript_before_calling_llm(store):
    llm = FakeLLM()
    uc = ConverseUseCase(llm=llm, store=store)
    await uc.execute(transcript="hello", session_id="s1")
    sent = llm.calls[0]
    assert sent[-1] == {"role": "user", "content": "hello"}


async def test_execute_appends_assistant_reply_to_history(store):
    llm = FakeLLM(reply="hi back")
    uc = ConverseUseCase(llm=llm, store=store)
    await uc.execute(transcript="hi", session_id="s1")
    conv = store.get_or_create("s1")
    assert conv.messages[-1].role is Role.ASSISTANT
    assert conv.messages[-1].content == "hi back"


async def test_execute_passes_full_history_to_llm_on_subsequent_turn(store):
    llm = FakeLLM(reply="ok")
    uc = ConverseUseCase(llm=llm, store=store)
    await uc.execute(transcript="first", session_id="s1")
    await uc.execute(transcript="second", session_id="s1")
    second_call = llm.calls[1]
    assert second_call == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]


async def test_execute_isolates_sessions(store):
    llm = FakeLLM()
    uc = ConverseUseCase(llm=llm, store=store)
    await uc.execute(transcript="alpha", session_id="a")
    await uc.execute(transcript="beta", session_id="b")
    a = store.get_or_create("a")
    b = store.get_or_create("b")
    assert [m.content for m in a.messages] == ["alpha", "ok"]
    assert [m.content for m in b.messages] == ["beta", "ok"]


async def test_execute_injects_system_prompt_on_first_turn_when_configured(store):
    llm = FakeLLM()
    uc = ConverseUseCase(
        llm=llm, store=store, system_prompt="You are Synthesis, a helpful AI."
    )
    await uc.execute(transcript="hi", session_id="s1")
    sent = llm.calls[0]
    assert sent[0] == {
        "role": "system",
        "content": "You are Synthesis, a helpful AI.",
    }


async def test_execute_does_not_duplicate_system_prompt_on_later_turns(store):
    llm = FakeLLM()
    uc = ConverseUseCase(llm=llm, store=store, system_prompt="be helpful")
    await uc.execute(transcript="one", session_id="s1")
    await uc.execute(transcript="two", session_id="s1")
    second_call = llm.calls[1]
    system_messages = [m for m in second_call if m["role"] == "system"]
    assert len(system_messages) == 1
