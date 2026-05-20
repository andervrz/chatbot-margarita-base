"""
Configuración del proyecto usando Pydantic Settings.
Toda la config se carga desde variables de entorno o .env.

Importante: el singleton se instancia en import-time. En tests,
setear env vars ANTES de cualquier import que toque este módulo.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default: project_root/data/margarita_bot.db
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "margarita_bot.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (LiteLLM + Groq) ---
    llm_model: str = Field(default="groq/llama-3.1-8b-instant", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0, alias="LLM_TEMPERATURE")
    # Aumentado a 1200 para mostrar 2-3 propiedades completas sin corte
    llm_max_tokens: int = Field(default=1200, alias="LLM_MAX_TOKENS")
    # Obligatorio. La app falla al arrancar si no está definida.
    groq_api_key: str = Field(alias="GROQ_API_KEY")

    # --- Embeddings ---
    embedding_model: str = Field(
        default="huggingface/sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")
    embedding_dimensions: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")

    # Fallback para embeddings vía OpenAI
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- Database ---
    database_path: Path = Field(default=_DEFAULT_DB_PATH, alias="DATABASE_PATH")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


_settings_instance: Settings | None = None

def get_settings() -> Settings:
    """Lazy singleton. Útil en tests donde se necesita reconfigurar."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings = get_settings()
