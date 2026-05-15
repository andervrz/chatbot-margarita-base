"""
Configuración del proyecto usando Pydantic Settings.
Toda la config se carga desde variables de entorno o .env
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (LiteLLM + Groq) ---
    llm_model: str = Field(default="groq/llama-3.1-8b-instant", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=500, alias="LLM_MAX_TOKENS")
    groq_api_key: str = Field(alias="GROQ_API_KEY")

    # --- Embeddings ---
    embedding_model: str = Field(default="huggingface/sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")

    # --- Database ---
    database_path: Path = Field(default=Path("./data/margarita_bot.db"), alias="DATABASE_PATH")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


# Singleton global. Importar desde cualquier módulo.
settings = Settings()
