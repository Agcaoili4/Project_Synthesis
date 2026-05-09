"""Ollama-backed Embedder adapter.

POSTs to ``{ollama_url}/api/embeddings`` (the legacy endpoint, still supported
by current Ollama and what older client libraries use). Returns a single
fixed-dim vector per call.
"""

import httpx

from app.application.memory import EmbedderError


class OllamaEmbedder:
    """Adapter for Ollama's embedding endpoint. Implements the Embedder Protocol."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        request_timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._timeout = request_timeout_s

    async def embed(self, text: str) -> list[float]:
        if not text:
            raise EmbedderError("cannot embed empty text")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise EmbedderError(f"ollama embed transport failure: {exc}") from exc

        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise EmbedderError(f"ollama returned no embedding (response keys: {list(data)})")
        if len(vec) != self._dim:
            raise EmbedderError(
                f"embedding dim mismatch: model={self._model} returned {len(vec)} "
                f"but configured MEMORY_EMBED_DIM={self._dim}"
            )
        return [float(x) for x in vec]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model
