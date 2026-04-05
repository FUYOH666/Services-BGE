import os

from pydantic_settings import BaseSettings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class EmbeddingSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9001
    model: str = "BAAI/bge-m3"
    device: str = "mps"
    use_fp16: bool = False
    max_length: int = 512
    batch_size: int = 12
    workers: int = 1
    log_level: str = "INFO"

    model_config = {"env_prefix": "EMBEDDING_", "env_file": ".env", "extra": "ignore"}


settings = EmbeddingSettings()
