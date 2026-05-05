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


class FakeTTS:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.spoken: list[str] = []
        self.warmed = False

    async def warm_up(self) -> None:
        self.warmed = True

    async def speak(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("speaker down")
        self.spoken.append(text)


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
    assert "manifest.webmanifest" in response.text
    assert "Voice loop monitor" in response.text
    assert "TTS Studio" in response.text
    assert 'id="wave"' in response.text
    assert "/dashboard/tts" in response.text
    assert "https://unpkg.com" not in response.text


def test_get_manifest_returns_installable_app_metadata(client: TestClient):
    response = client.get("/manifest.webmanifest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/manifest+json")
    body = response.json()
    assert body["name"] == "Project Synthesis"
    assert body["display"] == "standalone"
    assert body["start_url"] == "/"
    assert body["icons"][0]["src"] == "/synthesis-icon.svg"


def test_get_icon_returns_local_svg_asset(client: TestClient):
    response = client.get("/synthesis-icon.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text


def test_post_dashboard_tts_speaks_with_local_engine(fake_llm: FakeLLM):
    tts = FakeTTS()
    app = build_app(
        llm=fake_llm,
        store=InMemoryConversationStore(),
        dashboard_tts_factory=lambda: tts,
    )
    client = TestClient(app)

    response = client.post("/dashboard/tts", json={"text": "hello voice"})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert tts.warmed is True
    assert tts.spoken == ["hello voice"]


def test_post_dashboard_tts_rejects_blank_text(fake_llm: FakeLLM):
    app = build_app(
        llm=fake_llm,
        store=InMemoryConversationStore(),
        dashboard_tts_factory=lambda: FakeTTS(),
    )
    client = TestClient(app)

    response = client.post("/dashboard/tts", json={"text": "   "})

    assert response.status_code == 422


def test_post_dashboard_tts_returns_503_when_engine_fails(fake_llm: FakeLLM):
    app = build_app(
        llm=fake_llm,
        store=InMemoryConversationStore(),
        dashboard_tts_factory=lambda: FakeTTS(fail=True),
    )
    client = TestClient(app)

    response = client.post("/dashboard/tts", json={"text": "hello"})

    assert response.status_code == 503
    assert "local TTS unavailable" in response.json()["detail"]


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


def test_dashboard_escapes_session_id(client: TestClient):
    client.post(
        "/converse",
        json={"transcript": "hello", "session_id": "safe-session"},
    )
    response = client.get("/dashboard/transcripts")
    assert response.status_code == 200
    assert 'data-session-id="safe-session"' in response.text


def test_transcript_and_dashboard_hide_system_prompt(fake_llm: FakeLLM):
    store = InMemoryConversationStore()
    app = build_app(llm=fake_llm, store=store, system_prompt="secret instructions")
    client = TestClient(app)

    client.post("/converse", json={"transcript": "hello", "session_id": "s1"})

    transcript = client.get("/transcript/s1").json()
    assert all(message["role"] != "system" for message in transcript["messages"])
    assert "secret instructions" not in client.get("/dashboard/transcripts").text
