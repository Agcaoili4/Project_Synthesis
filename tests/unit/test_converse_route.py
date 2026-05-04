import pytest
from fastapi.testclient import TestClient

from app.application.converse import InMemoryConversationStore, LLMClient
from app.main import build_app


class FakeLLM:
    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


class FailingLLM:
    async def chat(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("backend down")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(reply="Paris")


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


@pytest.fixture
def client(fake_llm: LLMClient, store: InMemoryConversationStore) -> TestClient:
    app = build_app(llm=fake_llm, store=store, system_prompt=None)
    return TestClient(app)


def test_post_converse_returns_reply(client: TestClient, fake_llm: FakeLLM):
    response = client.post(
        "/converse",
        json={"transcript": "capital of France?", "session_id": "s1"},
    )
    assert response.status_code == 200
    assert response.json() == {"reply": "Paris", "session_id": "s1"}


def test_post_converse_returns_503_when_llm_is_unavailable():
    app = build_app(llm=FailingLLM(), store=InMemoryConversationStore())
    client = TestClient(app)

    response = client.post(
        "/converse",
        json={"transcript": "capital of France?", "session_id": "s1"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "LLM backend unavailable. Check that Ollama is running "
            "and the configured model is installed."
        )
    }


def test_post_converse_persists_history_across_calls(
    client: TestClient, store: InMemoryConversationStore
):
    client.post("/converse", json={"transcript": "first", "session_id": "s1"})
    client.post("/converse", json={"transcript": "second", "session_id": "s1"})
    conv = store.get_or_create("s1")
    assert [m.content for m in conv.messages] == ["first", "Paris", "second", "Paris"]


def test_post_converse_rejects_empty_transcript(client: TestClient):
    response = client.post(
        "/converse", json={"transcript": "", "session_id": "s1"}
    )
    assert response.status_code == 422


def test_post_converse_rejects_missing_session_id(client: TestClient):
    response = client.post("/converse", json={"transcript": "hi"})
    assert response.status_code == 422


def test_post_converse_rejects_oversized_transcript(client: TestClient):
    response = client.post(
        "/converse", json={"transcript": "x" * 8001, "session_id": "s1"}
    )
    assert response.status_code == 422


def test_post_converse_rejects_unsafe_session_id(client: TestClient):
    response = client.post(
        "/converse", json={"transcript": "hi", "session_id": "<script>"}
    )
    assert response.status_code == 422


def test_get_transcript_returns_session_messages(
    client: TestClient, store: InMemoryConversationStore
):
    client.post("/converse", json={"transcript": "hello", "session_id": "s1"})
    response = client.get("/transcript/s1")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s1"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello"
    assert body["messages"][1]["role"] == "assistant"


def test_get_dashboard_returns_html(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Synthesis" in response.text
    assert "https://unpkg.com" not in response.text


def test_dashboard_escapes_transcript_content(
    client: TestClient, store: InMemoryConversationStore
):
    client.post(
        "/converse",
        json={"transcript": "<b>xss</b>", "session_id": "s1"},
    )
    response = client.get("/dashboard/transcripts")
    assert response.status_code == 200
    assert "<b>xss</b>" not in response.text
    assert "&lt;b&gt;xss&lt;/b&gt;" in response.text


def test_transcript_and_dashboard_hide_system_prompt(fake_llm: FakeLLM):
    store = InMemoryConversationStore()
    app = build_app(llm=fake_llm, store=store, system_prompt="secret instructions")
    client = TestClient(app)

    client.post("/converse", json={"transcript": "hello", "session_id": "s1"})

    transcript = client.get("/transcript/s1").json()
    assert all(message["role"] != "system" for message in transcript["messages"])
    assert "secret instructions" not in client.get("/dashboard/transcripts").text
