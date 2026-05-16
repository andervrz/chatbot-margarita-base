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
# Path(__file__) está en src/config.py → subir dos niveles → raíz del proyecto
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "margarita_bot.db"


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
    # Obligatorio. La app falla al arrancar si no está definida.
    groq_api_key: str = Field(alias="GROQ_API_KEY")

    # --- Embeddings ---
    embedding_model: str = Field(
        default="huggingface/sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    # Requerido cuando embedding_model empieza con 'huggingface/'
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")
    # Debe coincidir con las dimensiones del modelo de embeddings.
    # all-MiniLM-L6-v2 = 384, text-embedding-ada-002 = 1536, bge-large = 1024
    embedding_dimensions: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")

    # Fallback para embeddings vía OpenAI (si se usa LiteLLM con openai/)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- Database ---
    database_path: Path = Field(default=_DEFAULT_DB_PATH, alias="DATABASE_PATH")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


# Singleton global. Importar desde cualquier módulo.
# Nota: en tests, setear env vars antes de importar este módulo.
# Preferir get_settings() en nuevo código para permitir lazy init en el futuro.
_settings_instance: Settings | None = None

def get_settings() -> Settings:
    """Lazy singleton. Útil en tests donde se necesita reconfigurar."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# Variable global para compatibilidad con código existente.
settings = get_settings()
