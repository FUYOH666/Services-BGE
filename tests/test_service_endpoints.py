from __future__ import annotations

import base64
import importlib
import struct
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

DIM = 1024


class FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeTokenizer:
    """Counts whitespace tokens plus CLS/SEP, truncated like a real tokenizer."""

    def __call__(self, texts, truncation: bool = True, max_length: int = 512):
        ids = [[0] * min(len(t.split()) + 2, max_length) for t in texts]
        return {"input_ids": ids}


class FakeEmbeddingModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.calls: list[dict] = []

    def encode(
        self,
        texts,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ):
        self.calls.append({"texts": list(texts), "batch_size": batch_size, "max_length": max_length})
        result = {}
        if return_dense:
            result["dense_vecs"] = [FakeArray([float(i + 1)] * DIM) for i, _ in enumerate(texts)]
        if return_sparse:
            result["lexical_weights"] = [{"1": 0.8, "2": 0.2} for _ in texts]
        if return_colbert_vecs:
            result["colbert_vecs"] = [FakeArray([[0.1, 0.2], [0.3, 0.4]]) for _ in texts]
        return result

    def convert_id_to_token(self, lexical_weights):
        return [{"tok_1": item["1"], "tok_2": item["2"]} for item in lexical_weights]


class FakeRerankerModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.calls: list[dict] = []

    def compute_score(self, pairs, normalize: bool = True, batch_size: int | None = None, max_length: int | None = None):
        self.calls.append({"pairs": pairs, "batch_size": batch_size, "max_length": max_length})
        return [1.0 - (idx * 0.1) for idx, _ in enumerate(pairs)]


def _fake_flagembedding():
    return SimpleNamespace(BGEM3FlagModel=FakeEmbeddingModel, FlagReranker=FakeRerankerModel)


@pytest.fixture
def embedding_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", _fake_flagembedding())
    module = importlib.import_module("embedding_service.main")
    module._state.clear()
    monkeypatch.setattr(module.settings, "device", "cpu", raising=False)
    monkeypatch.setattr(module.settings, "api_key", None, raising=False)

    with TestClient(module.app) as client:
        yield client, module

    module._state.clear()


@pytest.fixture
def reranker_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", _fake_flagembedding())
    module = importlib.import_module("reranker_service.main")
    module._state.clear()
    monkeypatch.setattr(module.settings, "device", "cpu", raising=False)
    monkeypatch.setattr(module.settings, "api_key", None, raising=False)

    with TestClient(module.app) as client:
        yield client, module

    module._state.clear()


# --- Health ---------------------------------------------------------------


def test_embedding_readiness_and_health_aliases(embedding_client):
    client, module = embedding_client

    live = client.get("/livez")
    assert live.status_code == 200
    assert live.json()["ready"] is True

    module._state["ready"] = False

    live_after = client.get("/livez")
    assert live_after.status_code == 200
    assert live_after.json()["ready"] is False

    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "loading"
    assert ready.json()["ready"] is False

    health = client.get("/healthz")
    assert health.status_code == 503
    assert health.json() == ready.json()


def test_reranker_readiness_and_liveness(reranker_client):
    client, module = reranker_client

    live = client.get("/livez")
    assert live.status_code == 200
    assert live.json()["ready"] is True

    module._state["ready"] = False

    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "loading"
    assert ready.json()["ready"] is False

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json() == ready.json()


def test_warmup_runs_before_ready(embedding_client, reranker_client):
    _, emb = embedding_client
    _, rer = reranker_client
    assert emb._state["backend"].model.calls[0]["texts"] == ["warmup"]
    assert emb._state["dense_dim"] == DIM
    assert rer._state["backend"].model.calls[0]["pairs"] == [["warmup", "warmup"]]


# --- Embedding contract (fields used by downstream projects) ---------------


def test_embedding_default_contract_is_unchanged(embedding_client):
    """Downstream projects read these exact fields; they must stay in the default response."""
    client, _ = embedding_client

    response = client.post(
        "/v1/embeddings",
        json={
            "input": ["first", "second"],
            "return_dense": True,
            "return_sparse": True,
            "return_colbert": True,
            "max_length": 128,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "BAAI/bge-m3"
    assert len(body["data"]) == 2
    item = body["data"][0]
    assert item["index"] == 0
    assert item["object"] == "embedding"
    assert len(item["dense_embedding"]) == DIM
    assert item["embedding"] == item["dense_embedding"]
    assert item["sparse_qdrant"] == {"indices": [1, 2], "values": [0.8, 0.2]}
    assert item["sparse_embedding"] == {"tok_1": 0.8, "tok_2": 0.2}
    assert item["colbert_embedding"] == [[0.1, 0.2], [0.3, 0.4]]
    # Fields that are not requested are still present as null in the default shape.
    dense_only = client.post("/v1/embeddings", json={"input": ["x"], "return_sparse": False}).json()
    assert set(dense_only["data"][0]) >= {
        "index",
        "embedding",
        "dense_embedding",
        "sparse_embedding",
        "sparse_qdrant",
        "colbert_embedding",
    }
    assert dense_only["data"][0]["sparse_qdrant"] is None


def test_embedding_accepts_openai_string_input_and_model(embedding_client):
    client, _ = embedding_client

    response = client.post("/v1/embeddings", json={"input": "одна строка", "model": "anything", "dimensions": DIM})

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == DIM


def test_embedding_rejects_unsupported_dimensions(embedding_client):
    client, _ = embedding_client
    response = client.post("/v1/embeddings", json={"input": "x", "dimensions": 256})
    assert response.status_code == 400


def test_embedding_rejects_empty_input(embedding_client):
    client, _ = embedding_client
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 422


def test_embedding_usage_counts_tokens(embedding_client):
    client, _ = embedding_client

    body = client.post("/v1/embeddings", json={"input": ["a b c", "d"]}).json()

    # FakeTokenizer: words + 2 special tokens -> (3 + 2) + (1 + 2)
    assert body["usage"]["total_tokens"] == 8
    assert body["usage"]["prompt_tokens"] == 8


def test_embedding_base64_round_trip(embedding_client):
    client, _ = embedding_client

    body = client.post("/v1/embeddings", json={"input": ["a", "b"], "encoding_format": "base64"}).json()

    raw = base64.b64decode(body["data"][1]["embedding"])
    values = struct.unpack(f"<{DIM}f", raw)
    assert values[0] == pytest.approx(2.0)
    assert body["data"][1]["dense_embedding"] == body["data"][1]["embedding"]


def test_embedding_compact_drops_duplicates(embedding_client):
    client, _ = embedding_client

    body = client.post("/v1/embeddings", json={"input": ["a"], "compact": True}).json()

    item = body["data"][0]
    assert "dense_embedding" not in item
    assert "sparse_embedding" not in item
    assert "colbert_embedding" not in item
    assert len(item["embedding"]) == DIM
    assert item["sparse_qdrant"]["indices"] == [1, 2]


def test_embedding_max_length_defaults_to_settings(embedding_client, monkeypatch):
    client, module = embedding_client
    monkeypatch.setattr(module.settings, "max_length", 333)

    client.post("/v1/embeddings", json={"input": ["a"]})

    assert module._state["backend"].model.calls[-1]["max_length"] == 333


def test_embedding_limits(embedding_client, monkeypatch):
    client, module = embedding_client
    monkeypatch.setattr(module.settings, "max_items", 2)
    monkeypatch.setattr(module.settings, "max_text_chars", 10)
    monkeypatch.setattr(module.settings, "max_length_cap", 1024)

    assert client.post("/v1/embeddings", json={"input": ["a", "b", "c"]}).status_code == 413
    assert client.post("/v1/embeddings", json={"input": ["x" * 11]}).status_code == 413
    assert client.post("/v1/embeddings", json={"input": ["a"], "max_length": 2048}).status_code == 422
    assert client.post("/v1/embeddings", json={"input": ["a", "b"], "max_length": 1024}).status_code == 200


def test_embedding_api_key(embedding_client, monkeypatch):
    client, module = embedding_client
    monkeypatch.setattr(module.settings, "api_key", "s3cret")

    assert client.post("/v1/embeddings", json={"input": "a"}).status_code == 401
    assert client.post("/v1/embeddings", json={"input": "a"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/embeddings", json={"input": "a"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post("/v1/embeddings", json={"input": "a"}, headers={"X-API-Key": "s3cret"}).status_code == 200
    # Health endpoints stay open so launchd/status checks keep working.
    assert client.get("/readyz").status_code == 200


# --- Reranker contract -----------------------------------------------------


def test_reranker_default_contract_is_unchanged(reranker_client):
    client, _ = reranker_client

    response = client.post(
        "/v1/rerank",
        json={
            "query": "pipes",
            "documents": ["best match", "second match", "third match"],
            "normalize": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert body["usage"] == {"total_pairs": 3}
    scores = [item["relevance_score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["results"][0] == {"index": 0, "document": "best match", "relevance_score": 1.0}


def test_reranker_top_n(reranker_client):
    client, _ = reranker_client
    docs = ["a", "b", "c"]

    body = client.post("/v1/rerank", json={"query": "q", "documents": docs, "top_n": 2}).json()
    assert [r["index"] for r in body["results"]] == [0, 1]

    assert client.post("/v1/rerank", json={"query": "q", "documents": docs, "top_n": -1}).status_code == 422
    assert client.post("/v1/rerank", json={"query": "q", "documents": docs, "top_n": 0}).status_code == 422


def test_reranker_return_documents_false(reranker_client):
    client, _ = reranker_client

    body = client.post(
        "/v1/rerank", json={"query": "q", "documents": ["a", "b"], "return_documents": False}
    ).json()

    assert body["results"][0] == {"index": 0, "relevance_score": 1.0}


def test_reranker_passes_batch_size_and_max_length(reranker_client, monkeypatch):
    client, module = reranker_client
    monkeypatch.setattr(module._state["backend"], "batch_size", 7)

    client.post("/v1/rerank", json={"query": "q", "documents": ["a"], "max_length": 256})
    call = module._state["backend"].model.calls[-1]
    assert call["batch_size"] == 7
    assert call["max_length"] == 256

    client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert module._state["backend"].model.calls[-1]["max_length"] == module.settings.max_length


def test_reranker_limits(reranker_client, monkeypatch):
    client, module = reranker_client
    monkeypatch.setattr(module.settings, "max_documents", 2)
    monkeypatch.setattr(module.settings, "max_text_chars", 5)

    assert client.post("/v1/rerank", json={"query": "q", "documents": ["a", "b", "c"]}).status_code == 413
    assert client.post("/v1/rerank", json={"query": "q", "documents": ["toolong"]}).status_code == 413
    assert client.post("/v1/rerank", json={"query": "toolong", "documents": ["a"]}).status_code == 413


def test_reranker_api_key(reranker_client, monkeypatch):
    client, module = reranker_client
    monkeypatch.setattr(module.settings, "api_key", "k")

    assert client.post("/v1/rerank", json={"query": "q", "documents": ["a"]}).status_code == 401
    assert (
        client.post("/v1/rerank", json={"query": "q", "documents": ["a"]}, headers={"X-API-Key": "k"}).status_code
        == 200
    )


# --- Discovery ---------------------------------------------------------------


def test_models_endpoints_describe_backend(embedding_client, reranker_client):
    emb_client, _ = embedding_client
    rer_client, _ = reranker_client

    emb = emb_client.get("/v1/models").json()["data"][0]
    assert emb["id"] == "BAAI/bge-m3"
    assert emb["backend"] == "bge-m3"
    assert emb["capabilities"] == {"dense": True, "sparse": True, "colbert": True, "matryoshka": False, "prompts": False}
    assert emb["dimensions"] == DIM

    rer = rer_client.get("/v1/models").json()["data"][0]
    assert rer["backend"] == "flag-reranker"


def test_flag_reranker_rejects_instruction(reranker_client):
    client, _ = reranker_client
    response = client.post("/v1/rerank", json={"query": "q", "documents": ["a"], "instruction": "classify customs codes"})
    assert response.status_code == 400


# --- sentence-transformers embedding backend --------------------------------


class FakeSTModel:
    def __init__(self, model_name, device=None, trust_remote_code=False):
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeTokenizer()
        self.prompts = {"query": "Instruct: find\nQuery:"}
        self.max_seq_length = 512
        self.calls: list[dict] = []

    def half(self):
        return self

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False, truncate_dim=None, prompt=None, prompt_name=None):
        import numpy as np

        self.calls.append({"texts": list(texts), "prompt": prompt, "prompt_name": prompt_name, "truncate_dim": truncate_dim})
        dim = truncate_dim or 8
        out = np.ones((len(texts), dim), dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def st_embedding_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", _fake_flagembedding())
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSTModel))
    module = importlib.import_module("embedding_service.main")
    module._state.clear()
    for key, value in {
        "device": "cpu",
        "api_key": None,
        "backend": "sentence-transformers",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "query_prompt_name": "query",
        "matryoshka": True,
    }.items():
        monkeypatch.setattr(module.settings, key, value, raising=False)

    with TestClient(module.app) as client:
        yield client, module

    module._state.clear()


def test_st_backend_dense_only_and_prompts(st_embedding_client):
    client, module = st_embedding_client
    model = module._state["backend"].model

    body = client.post("/v1/embeddings", json={"input": ["труба"], "input_type": "query"}).json()
    item = body["data"][0]
    assert len(item["embedding"]) == 8
    assert item["sparse_qdrant"] is None
    assert model.calls[-1]["prompt_name"] == "query"
    # Query prompt tokens are counted: "Instruct: find\nQuery:труба" -> 3 words + 2
    assert body["usage"]["total_tokens"] == 5

    client.post("/v1/embeddings", json={"input": ["труба"]})
    assert model.calls[-1]["prompt_name"] is None

    assert client.get("/v1/models").json()["data"][0]["capabilities"]["sparse"] is False


def test_st_backend_rejects_explicit_sparse(st_embedding_client):
    client, _ = st_embedding_client
    assert client.post("/v1/embeddings", json={"input": ["a"], "return_sparse": True}).status_code == 400
    assert client.post("/v1/embeddings", json={"input": ["a"], "return_colbert": True}).status_code == 400


def test_st_backend_matryoshka_dimensions(st_embedding_client):
    client, module = st_embedding_client

    body = client.post("/v1/embeddings", json={"input": ["a"], "dimensions": 4}).json()
    assert len(body["data"][0]["embedding"]) == 4
    assert module._state["backend"].model.calls[-1]["truncate_dim"] == 4
    assert client.post("/v1/embeddings", json={"input": ["a"], "dimensions": 16}).status_code == 400


def test_unknown_backend_fails_fast(monkeypatch):
    from embedding_service.backends import load_backend as load_embedding
    from reranker_service.backends import load_backend as load_reranker

    with pytest.raises(ValueError, match="EMBEDDING_BACKEND"):
        load_embedding(SimpleNamespace(backend="nope"), "m")
    with pytest.raises(ValueError, match="RERANKER_BACKEND"):
        load_reranker(
            SimpleNamespace(backend="nope", device="cpu", use_fp16=False, batch_size=1, trust_remote_code=False, instruction=None),
            "m",
        )


def test_sigmoid_matches_normalized_scores():
    from reranker_service.backends import _sigmoid

    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(-800.0) == pytest.approx(0.0)
    assert _sigmoid(800.0) == pytest.approx(1.0)
