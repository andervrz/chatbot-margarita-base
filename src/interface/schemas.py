"""
Schemas Pydantic para la API REST.
Separan la interfaz HTTP de la lógica de dominio.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request: mensaje del usuario."""
    session_id: str = Field(
        ...,
        min_length=1,
        description="ID único de sesión. Permite retomar conversaciones.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Mensaje del usuario.",
    )
    user_id: str | None = Field(
        default=None,
        description="ID opcional del usuario para CRM futuro.",
    )


class ChatResponse(BaseModel):
    """Response: respuesta del asistente."""
    session_id: str
    response: str
    intent_type: str | None = None
    properties_found: int = 0
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check."""
    status: str
    database: str
    llm_provider: str
