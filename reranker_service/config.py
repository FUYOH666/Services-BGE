import os

from pydantic_settings import BaseSettings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RerankerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 9002
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "mps"
    use_fp16: bool = False
    batch_size: int = 32
    workers: int = 1
    log_level: str = "INFO"

    model_config = {"env_prefix": "RERANKER_", "env_file": ".env", "extra": "ignore"}


settings = RerankerSettings()
