import os

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class EmbeddingSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9001
    # Backend that loads the model: "bge-m3" (dense + sparse + ColBERT via
    # FlagEmbedding) or "sentence-transformers" (any dense SentenceTransformer).
    backend: str = "bge-m3"
    model: str = "BAAI/bge-m3"
    # Optional Hugging Face revision (commit sha or tag) for reproducible restarts.
    model_revision: str | None = None
    device: str = "mps"
    use_fp16: bool = False
    # Default token limit when a request does not send max_length.
    max_length: int = 512
    # Hard ceiling for per-request max_length (bge-m3 supports 8192).
    max_length_cap: int = 8192
    batch_size: int = 12
    # Request limits; 0 disables a limit.
    max_items: int = 256
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
    # sentence-transformers backend: prompts prepended by `input_type`.
    # Literal text wins over a prompt name defined in the model's config.
    # "\\n" in .env values is turned into a newline.
    query_prompt: str | None = None
    query_prompt_name: str | None = None
    document_prompt: str | None = None
    document_prompt_name: str | None = None
    # Model supports Matryoshka truncation, so smaller `dimensions` are allowed.
    matryoshka: bool = False
    trust_remote_code: bool = False
    # Optional shared secret. Empty = no auth (local-only default).
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "SERVICE_API_KEY"),
    )

    model_config = {"env_prefix": "EMBEDDING_", "env_file": ".env", "extra": "ignore"}

    @field_validator("query_prompt", "document_prompt", mode="before")
    @classmethod
    def _unescape_newlines(cls, value):
        if isinstance(value, str):
            return value.replace("\\n", "\n")
        return value


settings = EmbeddingSettings()
