import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RerankerSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9002
    model: str = "BAAI/bge-reranker-v2-m3"
    # Optional Hugging Face revision (commit sha or tag) for reproducible restarts.
    model_revision: str | None = None
    device: str = "mps"
    use_fp16: bool = False
    batch_size: int = 32
    # Default token limit for query+document when a request does not send max_length.
    max_length: int = 512
    # Hard ceiling for per-request max_length.
    max_length_cap: int = 8192
    # Request limits; 0 disables a limit.
    max_documents: int = 1000
    max_text_chars: int = 100_000
    workers: int = 1
    log_level: str = "INFO"
    # Requests slower than this are logged at WARNING.
    slow_request_ms: int = 2000
    # Run one tiny inference before reporting ready.
    warmup: bool = True
    # 0 = clear the MPS cache after every request (historical behaviour);
    # >0 = clear only when the Metal driver holds more than this many MB.
    mps_cache_threshold_mb: int = 0
    # Optional shared secret. Empty = no auth (local-only default).
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANKER_API_KEY", "SERVICE_API_KEY"),
    )

    model_config = {"env_prefix": "RERANKER_", "env_file": ".env", "extra": "ignore"}


settings = RerankerSettings()
