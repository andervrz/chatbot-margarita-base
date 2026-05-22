"""
Configuración del proyecto usando Pydantic Settings.
Toda la config se carga desde variables de entorno o .env.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_max_tokens: int = Field(default=1200, alias="LLM_MAX_TOKENS")
    groq_api_key: str = Field(alias="GROQ_API_KEY")

    # --- Embeddings ---
    embedding_model: str = Field(
        default="huggingface/sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")
    embedding_dimensions: int = Field(default=384, alias="EMBEDDING_DIMENSIONS")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- Database ---
    database_path: Path = Field(default=_DEFAULT_DB_PATH, alias="DATABASE_PATH")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # NUEVO Fase 1: WhatsApp Business API
    whatsapp_access_token: str | None = Field(default=None, alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_business_phone_id: str | None = Field(default=None, alias="WHATSAPP_BUSINESS_PHONE_ID")
    whatsapp_verify_token: str | None = Field(default=None, alias="WHATSAPP_VERIFY_TOKEN")

    # NUEVO Fase 1: Datos de la agencia inmobiliaria
    agency_name: str = Field(default="Margarita Realty", alias="AGENCY_NAME")
    agency_whatsapp_number: str | None = Field(default=None, alias="AGENCY_WHATSAPP_NUMBER")
    agency_contact_email: str | None = Field(default=None, alias="AGENCY_CONTACT_EMAIL")

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


_settings_instance: Settings | None = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings = get_settings()