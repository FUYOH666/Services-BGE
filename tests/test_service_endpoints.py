from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeEmbeddingModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def encode(
        self,
        texts,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ):
        result = {}
        if return_dense:
            result["dense_vecs"] = [FakeArray([float(i + 1)] * 1024) for i, _ in enumerate(texts)]
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

    def compute_score(self, pairs, normalize: bool = True):
        return [1.0 - (idx * 0.1) for idx, _ in enumerate(pairs)]


@pytest.fixture
def embedding_client(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=FakeEmbeddingModel, FlagReranker=FakeRerankerModel),
    )
    module = importlib.import_module("embedding_service.main")
    module._state.clear()
    monkeypatch.setattr(module.settings, "device", "cpu", raising=False)

    with TestClient(module.app) as client:
        yield client, module

    module._state.clear()


@pytest.fixture
def reranker_client(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=FakeEmbeddingModel, FlagReranker=FakeRerankerModel),
    )
    module = importlib.import_module("reranker_service.main")
    module._state.clear()
    monkeypatch.setattr(module.settings, "device", "cpu", raising=False)

    with TestClient(module.app) as client:
        yield client, module

    module._state.clear()


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


def test_embedding_happy_path_response_shape(embedding_client):
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
    assert len(body["data"]) == 2
    assert len(body["data"][0]["dense_embedding"]) == 1024
    assert body["data"][0]["embedding"] == body["data"][0]["dense_embedding"]
    assert body["data"][0]["sparse_qdrant"]["indices"] == [1, 2]
    assert body["data"][0]["sparse_embedding"] == {"tok_1": 0.8, "tok_2": 0.2}
    assert body["data"][0]["colbert_embedding"] == [[0.1, 0.2], [0.3, 0.4]]


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


def test_reranker_happy_path_returns_sorted_scores(reranker_client):
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
    scores = [item["relevance_score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["results"][0]["document"] == "best match"
