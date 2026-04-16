from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    QWEN_API_KEY: str = Field(..., description="DashScope API key for Qwen models")
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-max"

    # Data directories
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    PAPERS_DIR: str = "./data/papers"

    def ensure_data_dirs(self) -> None:
        """Create required data directories if they do not exist."""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.PAPERS_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for app-wide reuse."""
    settings = Settings()
    settings.ensure_data_dirs()
    return settings
