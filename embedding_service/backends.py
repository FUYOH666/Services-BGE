"""Embedding backends.

A backend loads one model and turns texts into vectors. The HTTP layer in
`main.py` only talks to this interface, so switching models is a config change
(`EMBEDDING_BACKEND`, `EMBEDDING_MODEL`, prompts) instead of a code change.

- `bge-m3`: BAAI/bge-m3 through FlagEmbedding. Dense + sparse lexical weights +
  ColBERT vectors in one pass. The production default.
- `sentence-transformers`: any SentenceTransformer-compatible dense model
  (Qwen3-Embedding, multilingual-e5, harrier, ...). Dense only, with optional
  query/document prompts and Matryoshka truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from service_common.runtime import count_tokens

BACKENDS = ("bge-m3", "sentence-transformers")


@dataclass
class Capabilities:
    dense: bool = True
    sparse: bool = False
    colbert: bool = False
    # Matryoshka models accept a smaller `dimensions` than their native size.
    matryoshka: bool = False
    prompts: bool = False


@dataclass
class EncodeOutput:
    # Each entry is array-like with `.tolist()` (numpy arrays in production).
    dense: list[Any] | None = None
    # Token id (str) -> weight, as returned by FlagEmbedding.
    sparse: list[dict] | None = None
    colbert: list[Any] | None = None
    # Token -> weight, only for sparse backends that can decode ids.
    sparse_readable: list[dict] | None = None
    tokens: int = 0
    extra: dict = field(default_factory=dict)


class BGEM3Backend:
    name = "bge-m3"

    def __init__(self, model_path: str, *, device: str, use_fp16: bool, batch_size: int):
        from FlagEmbedding import BGEM3FlagModel

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device == "mps":
            kwargs["devices"] = ["mps"]
        self.model = BGEM3FlagModel(model_path, **kwargs)
        self.batch_size = batch_size
        self.capabilities = Capabilities(dense=True, sparse=True, colbert=True)

    @property
    def tokenizer(self):
        return getattr(self.model, "tokenizer", None)

    def encode(
        self,
        texts: list[str],
        *,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
        readable_sparse: bool,
        input_type: str | None = None,
        dimensions: int | None = None,
    ) -> EncodeOutput:
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )
        lexical = output.get("lexical_weights")
        if lexical is not None and not isinstance(lexical, list):
            lexical = [lexical]

        readable = None
        if lexical is not None and readable_sparse:
            readable = self.model.convert_id_to_token(lexical)
            if isinstance(readable, dict):
                readable = [readable]

        return EncodeOutput(
            dense=output.get("dense_vecs"),
            sparse=lexical,
            colbert=output.get("colbert_vecs"),
            sparse_readable=readable,
            tokens=count_tokens(self.tokenizer, texts, max_length),
        )


class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        use_fp16: bool,
        batch_size: int,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt: str | None = None,
        document_prompt_name: str | None = None,
        matryoshka: bool = False,
        trust_remote_code: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=trust_remote_code)
        if use_fp16:
            self.model.half()
        self.batch_size = batch_size
        self.query_prompt = query_prompt or None
        self.query_prompt_name = query_prompt_name or None
        self.document_prompt = document_prompt or None
        self.document_prompt_name = document_prompt_name or None
        has_prompts = bool(self.query_prompt or self.query_prompt_name or self.document_prompt or self.document_prompt_name)
        self.capabilities = Capabilities(dense=True, matryoshka=matryoshka, prompts=has_prompts)

    @property
    def tokenizer(self):
        return getattr(self.model, "tokenizer", None)

    def _prompt_kwargs(self, input_type: str | None) -> dict:
        if input_type == "query":
            if self.query_prompt:
                return {"prompt": self.query_prompt}
            if self.query_prompt_name:
                return {"prompt_name": self.query_prompt_name}
        else:
            if self.document_prompt:
                return {"prompt": self.document_prompt}
            if self.document_prompt_name:
                return {"prompt_name": self.document_prompt_name}
        return {}

    def encode(
        self,
        texts: list[str],
        *,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
        readable_sparse: bool,
        input_type: str | None = None,
        dimensions: int | None = None,
    ) -> EncodeOutput:
        self.model.max_seq_length = max_length
        kwargs = self._prompt_kwargs(input_type)
        dense = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            truncate_dim=dimensions if self.capabilities.matryoshka else None,
            **kwargs,
        )
        if dimensions is not None and self.capabilities.matryoshka:
            # Re-normalize after truncation so cosine == dot product.
            import numpy as np

            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            dense = dense / np.where(norms == 0, 1, norms)

        prompt = kwargs.get("prompt")
        if prompt is None and kwargs.get("prompt_name"):
            prompt = self.model.prompts.get(kwargs["prompt_name"], "")
        counted = [f"{prompt}{t}" for t in texts] if prompt else texts
        return EncodeOutput(
            dense=list(dense) if return_dense else None,
            tokens=count_tokens(self.tokenizer, counted, max_length),
        )


def load_backend(settings, model_path: str):
    backend = settings.backend
    if backend == "bge-m3":
        return BGEM3Backend(model_path, device=settings.device, use_fp16=settings.use_fp16, batch_size=settings.batch_size)
    if backend == "sentence-transformers":
        return SentenceTransformerBackend(
            model_path,
            device=settings.device,
            use_fp16=settings.use_fp16,
            batch_size=settings.batch_size,
            query_prompt=settings.query_prompt,
            query_prompt_name=settings.query_prompt_name,
            document_prompt=settings.document_prompt,
            document_prompt_name=settings.document_prompt_name,
            matryoshka=settings.matryoshka,
            trust_remote_code=settings.trust_remote_code,
        )
    raise ValueError(f"Unknown EMBEDDING_BACKEND {backend!r}; expected one of {', '.join(BACKENDS)}")
