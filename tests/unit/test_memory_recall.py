"""Unit tests for the recall + remember flow inside ConverseUseCase.

No real embedder, no real DB — fakes only. Verifies that:
  * recall hits get rendered into a SYSTEM message and injected just before
    the user turn in the LLM-bound message list, but are NOT persisted on
    the rolling Conversation (so memory does not accumulate across turns).
  * recall failure (embedder raises, repo raises) does not bubble — the
    turn still completes with a reply.
  * the post-turn embed+store is scheduled exactly once per turn, fires
    the embedder + repo with the canonical embed text, and tolerates
    embedder failure silently.
"""

from datetime import UTC, datetime

import pytest

from app.application.background import BackgroundTaskRunner
from app.application.converse import ConverseUseCase, InMemoryConversationStore
from app.domain.conversation import Role
from app.domain.memory import TurnPair


class FakeLLM:
    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.reply


class FakeEmbedder:
    def __init__(self, vec: list[float] | None = None, raises: bool = False) -> None:
        self._vec = vec or [1.0, 0.0, 0.0]
        self._raises = raises
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self._raises:
            raise RuntimeError("embedder boom")
        return list(self._vec)


class FakeMemoryRepo:
    def __init__(self, hits: list[TurnPair] | None = None, raises: bool = False) -> None:
        self._hits = hits or []
        self._raises = raises
        self.recalled_with: list[tuple[list[float], int, float]] = []
        self.remembered: list[tuple[TurnPair, list[float]]] = []

    async def remember(self, turn: TurnPair, embedding: list[float]) -> None:
        self.remembered.append((turn, embedding))

    async def recall(
        self, query_embedding: list[float], k: int, threshold: float
    ) -> list[TurnPair]:
        self.recalled_with.append((query_embedding, k, threshold))
        if self._raises:
            raise RuntimeError("repo boom")
        return list(self._hits)

    # remaining protocol stubs — unused in these tests
    async def list_recent(self, limit=50, session_id=None, since=None):  # noqa: ARG002
        return []

    async def get(self, turn_id):  # noqa: ARG002
        return None

    async def forget(self, turn_id):  # noqa: ARG002
        return False

    async def forget_before(self, cutoff):  # noqa: ARG002
        return 0

    async def wipe(self) -> int:
        return 0

    async def stats(self) -> dict[str, object]:
        return {}


def _hit(text: str) -> TurnPair:
    return TurnPair(
        id="h-" + text[:6],
        session_id="s",
        user_text=f"q about {text}",
        assistant_text=f"a about {text}",
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


async def test_no_memory_layer_keeps_existing_behavior(store):
    llm = FakeLLM(reply="hi")
    uc = ConverseUseCase(llm=llm, store=store)
    reply = await uc.execute(transcript="hello", session_id="s1")
    assert reply == "hi"
    sent = llm.calls[0]
    assert sent == [{"role": "user", "content": "hello"}]


async def test_recall_injects_memory_block_before_user_turn(store):
    llm = FakeLLM(reply="ok")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[_hit("colors")])
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
        recall_top_k=3,
        recall_threshold=0.65,
    )
    await uc.execute(transcript="what's my favorite color", session_id="s1")
    sent = llm.calls[0]
    # last message must be the user transcript, prior message must be the
    # injected memory SYSTEM block.
    assert sent[-1] == {"role": "user", "content": "what's my favorite color"}
    assert sent[-2]["role"] == "system"
    assert "<memory>" in sent[-2]["content"]
    assert "colors" in sent[-2]["content"]
    # recall was invoked with configured k + threshold
    assert memory.recalled_with == [([1.0, 0.0, 0.0], 3, 0.65)]
    await runner.drain(timeout=1.0)


async def test_memory_block_is_not_persisted_on_conversation(store):
    """Crucial: memory injection must be ephemeral, not appended to conv."""
    llm = FakeLLM(reply="ok")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[_hit("teal")])
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
    )
    await uc.execute(transcript="t1", session_id="s1")
    await uc.execute(transcript="t2", session_id="s1")
    conv = store.get_or_create("s1")
    # No SYSTEM messages should accumulate from memory injection.
    system_msgs = [m for m in conv.messages if m.role is Role.SYSTEM]
    assert system_msgs == []
    # The second LLM call must contain only one memory block (the fresh one),
    # not a second one stacked on top of a persisted first one.
    sent2 = llm.calls[1]
    memory_blocks = [m for m in sent2 if m["role"] == "system" and "<memory>" in m["content"]]
    assert len(memory_blocks) == 1
    await runner.drain(timeout=1.0)


async def test_no_hits_means_no_memory_block(store):
    llm = FakeLLM(reply="ok")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[])
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
    )
    await uc.execute(transcript="weather?", session_id="s1")
    sent = llm.calls[0]
    assert all("<memory>" not in m.get("content", "") for m in sent)
    await runner.drain(timeout=1.0)


async def test_embedder_failure_does_not_break_turn(store):
    llm = FakeLLM(reply="still works")
    embedder = FakeEmbedder(raises=True)
    memory = FakeMemoryRepo(hits=[_hit("x")])
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
    )
    reply = await uc.execute(transcript="hello", session_id="s1")
    assert reply == "still works"
    # No memory block, but turn completed.
    assert all("<memory>" not in m.get("content", "") for m in llm.calls[0])
    await runner.drain(timeout=1.0)


async def test_repo_failure_does_not_break_turn(store):
    llm = FakeLLM(reply="still works")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[_hit("x")], raises=True)
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
    )
    reply = await uc.execute(transcript="hello", session_id="s1")
    assert reply == "still works"
    await runner.drain(timeout=1.0)


async def test_post_turn_remember_is_scheduled_with_canonical_text(store):
    llm = FakeLLM(reply="paris")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[])
    runner = BackgroundTaskRunner()
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=runner,
    )
    await uc.execute(transcript="capital of france", session_id="s1")
    await runner.drain(timeout=2.0)
    # First embedder call was the recall (transcript), second was the store
    # (canonical "User: ...\nAssistant: ...").
    assert embedder.calls[0] == "capital of france"
    assert embedder.calls[1] == "User: capital of france\nAssistant: paris"
    assert len(memory.remembered) == 1
    stored_turn, stored_emb = memory.remembered[0]
    assert stored_turn.session_id == "s1"
    assert stored_turn.user_text == "capital of france"
    assert stored_turn.assistant_text == "paris"
    assert stored_emb == [1.0, 0.0, 0.0]


async def test_remember_skipped_when_no_background_runner(store):
    """Memory configured but runner missing — recall still works, store skipped."""
    llm = FakeLLM(reply="ok")
    embedder = FakeEmbedder()
    memory = FakeMemoryRepo(hits=[])
    uc = ConverseUseCase(
        llm=llm,
        store=store,
        embedder=embedder,
        memory=memory,
        background_tasks=None,
    )
    await uc.execute(transcript="hi", session_id="s1")
    assert memory.remembered == []
