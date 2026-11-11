"""Async clients for the Services-BGE embedding (:9001) and rerank (:9002) APIs."""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx


class ServiceError(Exception):
    """A failure with a message an agent (and its user) can act on."""


@dataclass
class Embedding:
    dense: list[float]
    # Qdrant-shaped sparse vector ({"indices": [...], "values": [...]}); None on dense-only models.
    sparse: dict[str, list] | None = None


@dataclass
class EmbedResult:
    model: str
    items: list[Embedding] = field(default_factory=list)
    tokens: int = 0


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return str(detail)[:300]


class _ServiceClient:
    name = "service"
    hint = ""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        # Created lazily so it binds to the event loop that serves the requests.
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=_headers(self._api_key),
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def _post(self, path: str, body: dict) -> dict:
        try:
            response = await self._http().post(path, json=body)
        except httpx.TimeoutException as exc:
            raise ServiceError(f"The {self.name} at {self.base_url} timed out ({exc.__class__.__name__}).") from exc
        except httpx.TransportError as exc:
            raise ServiceError(
                f"The {self.name} at {self.base_url} is not reachable ({exc.__class__.__name__}). {self.hint}"
            ) from exc
        if response.status_code == 401:
            raise ServiceError(
                f"The {self.name} at {self.base_url} rejected the API key. "
                "Set SERVICE_API_KEY (or MCP_EMBEDDING_API_KEY / MCP_RERANKER_API_KEY) for the MCP server."
            )
        if response.status_code == 503:
            raise ServiceError(f"The {self.name} at {self.base_url} is still loading its model; retry in a few seconds.")
        if response.status_code >= 400:
            raise ServiceError(f"The {self.name} returned {response.status_code}: {_detail(response)}")
        return response.json()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class EmbeddingClient(_ServiceClient):
    name = "embedding service"
    hint = "Start it with ./scripts/service.sh start, check with ./scripts/service.sh status."

    def __init__(self, base_url: str, *, max_length: int = 1024, batch_size: int = 32, **kwargs) -> None:
        super().__init__(base_url, **kwargs)
        self.max_length = max_length
        self.batch_size = max(1, batch_size)

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: str = "document",
        sparse: bool | None = None,
    ) -> EmbedResult:
        """Embed texts in batches.

        `sparse=None` asks for sparse weights only where the backend has them
        (bge-m3); `True` requires them (a dense-only model then answers 400).
        """
        result = EmbedResult(model="")
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            body: dict = {
                "input": batch,
                "compact": True,
                "input_type": input_type,
                "max_length": self.max_length,
            }
            if sparse is not None:
                body["return_sparse"] = sparse
            payload = await self._post("/v1/embeddings", body)
            data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(batch):
                raise ServiceError(f"The embedding service returned {len(data)} vectors for {len(batch)} texts.")
            result.model = payload.get("model", result.model)
            result.tokens += int((payload.get("usage") or {}).get("total_tokens", 0) or 0)
            for item in data:
                vector = item.get("embedding")
                if not isinstance(vector, list):
                    raise ServiceError("The embedding service returned no dense vector (base64 or dense disabled?).")
                result.items.append(Embedding(dense=vector, sparse=item.get("sparse_qdrant") or None))
        return result


class RerankClient(_ServiceClient):
    name = "rerank service"
    hint = "Start it with ./scripts/service.sh start, or search with rerank=false."

    def __init__(self, base_url: str, *, max_length: int = 1024, **kwargs) -> None:
        super().__init__(base_url, **kwargs)
        self.max_length = max_length

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[tuple[int, float]]]:
        """Return (model, [(index into documents, relevance 0..1)]) best first."""
        body: dict = {
            "query": query,
            "documents": documents,
            "return_documents": False,
            "max_length": self.max_length,
        }
        if top_n is not None:
            body["top_n"] = top_n
        payload = await self._post("/v1/rerank", body)
        ranked = [(int(r["index"]), float(r["relevance_score"])) for r in payload.get("results", [])]
        return payload.get("model", ""), ranked
