"""Reranker backends.

A backend scores (query, document) pairs. `normalize=True` returns scores in
[0, 1] (sigmoid of the raw relevance logit), `normalize=False` returns the raw
logit, so all backends share one contract.

- `flag-reranker`: cross-encoders through FlagEmbedding (BAAI/bge-reranker-v2-m3,
  the production default).
- `cross-encoder`: any sentence-transformers CrossEncoder model.
- `qwen3-reranker`: Qwen3-Reranker 0.6B/4B/8B with the official yes/no-logit
  scoring and a task instruction.
"""
from __future__ import annotations

import math
from typing import Any

BACKENDS = ("flag-reranker", "cross-encoder", "qwen3-reranker")

QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _torch_dtype(use_fp16: bool, device: str):
    import torch

    if use_fp16 and device != "cpu":
        return torch.float16
    return torch.float32


class FlagRerankerBackend:
    name = "flag-reranker"
    supports_instruction = False

    def __init__(self, model_path: str, *, device: str, use_fp16: bool, batch_size: int, **_: Any):
        from FlagEmbedding import FlagReranker

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device == "mps":
            kwargs["devices"] = ["mps"]
        self.model = FlagReranker(model_path, **kwargs)
        self.batch_size = batch_size

    def score(self, pairs: list[list[str]], *, max_length: int, normalize: bool, instruction: str | None = None) -> list[float]:
        scores = self.model.compute_score(pairs, batch_size=self.batch_size, max_length=max_length, normalize=normalize)
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]


class CrossEncoderBackend:
    name = "cross-encoder"
    supports_instruction = False

    def __init__(self, model_path: str, *, device: str, use_fp16: bool, batch_size: int, trust_remote_code: bool = False, **_: Any):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_path, device=device, trust_remote_code=trust_remote_code)
        if use_fp16:
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, pairs: list[list[str]], *, max_length: int, normalize: bool, instruction: str | None = None) -> list[float]:
        import torch

        self.model.max_length = max_length
        # Identity activation returns raw logits; we apply the sigmoid ourselves in fp32
        # so normalize behaves the same for every backend.
        logits = self.model.predict(
            [tuple(p) for p in pairs],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = [float(x) for x in logits.reshape(len(pairs), -1)[:, 0]]
        return [_sigmoid(v) for v in values] if normalize else values


class Qwen3RerankerBackend:
    """Qwen3-Reranker scoring as documented on the model card.

    The model is a causal LM asked whether the document satisfies the query under an
    instruction; relevance = P("yes") = sigmoid(logit_yes - logit_no).
    """

    name = "qwen3-reranker"
    supports_instruction = True

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        use_fp16: bool,
        batch_size: int,
        instruction: str | None = None,
        trust_remote_code: bool = False,
        **_: Any,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.instruction = instruction or QWEN3_DEFAULT_INSTRUCTION
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=trust_remote_code)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=_torch_dtype(use_fp16, device), trust_remote_code=trust_remote_code
            )
            .to(device)
            .eval()
        )
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)

    def _format(self, instruction: str, query: str, doc: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def score(self, pairs: list[list[str]], *, max_length: int, normalize: bool, instruction: str | None = None) -> list[float]:
        torch = self.torch
        task = instruction or self.instruction
        texts = [self._format(task, q, d) for q, d in pairs]
        body_limit = max(16, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        encoded = self.tokenizer(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=body_limit)
        sequences = [self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]]

        order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
        raw = [0.0] * len(sequences)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = self.tokenizer.pad({"input_ids": [sequences[i] for i in idx]}, padding=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits[:, -1, :].float()
                diff = (logits[:, self.token_yes] - logits[:, self.token_no]).tolist()
                for i, value in zip(idx, diff):
                    raw[i] = float(value)
        return [_sigmoid(v) for v in raw] if normalize else raw


def load_backend(settings, model_path: str):
    common = {
        "device": settings.device,
        "use_fp16": settings.use_fp16,
        "batch_size": settings.batch_size,
        "trust_remote_code": settings.trust_remote_code,
        "instruction": settings.instruction,
    }
    if settings.backend == "flag-reranker":
        return FlagRerankerBackend(model_path, **common)
    if settings.backend == "cross-encoder":
        return CrossEncoderBackend(model_path, **common)
    if settings.backend == "qwen3-reranker":
        return Qwen3RerankerBackend(model_path, **common)
    raise ValueError(f"Unknown RERANKER_BACKEND {settings.backend!r}; expected one of {', '.join(BACKENDS)}")
