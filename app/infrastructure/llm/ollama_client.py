import httpx


class OllamaClient:
    """Adapter for the Ollama HTTP API. Implements the LLMClient Protocol."""

    def __init__(
        self,
        base_url: str,
        model: str,
        request_timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = request_timeout_s

    async def chat(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        return data["message"]["content"].strip()
