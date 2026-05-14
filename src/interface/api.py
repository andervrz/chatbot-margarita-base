"""
API REST del bot inmobiliario.
Endpoints mínimos y necesarios: chat y health.
"""

import structlog
from fastapi import Depends, FastAPI, HTTPException

from src.domain.exceptions import DomainError, LLMError, NoDataError, PropertyNotFound
from src.interface.dependencies import lifespan, get_chat_service
from src.interface.schemas import ChatRequest, ChatResponse, HealthResponse
from src.application.chat import ChatService

logger = structlog.get_logger()

app = FastAPI(
    title="Margarita Realty Bot",
    description="Chatbot inmobiliario para Isla de Margarita",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Endpoint principal de conversación.
    Recibe mensaje del usuario y retorna respuesta del asistente.
    """
    try:
        response_text = await service.handle(
            session_id=request.session_id,
            user_message=request.message,
        )

        # Nota: para métricas simples, podríamos exponer más metadata,
        # pero mantenemos la respuesta limpia para el cliente.
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
        )

    except LLMError as exc:
        logger.error("chat_llm_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Error temporal con el servicio de lenguaje")

    except DomainError as exc:
        logger.error("chat_domain_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Error interno del bot")

    except Exception as exc:
        logger.error("chat_unexpected_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Error inesperado")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check simple."""
    return HealthResponse(
        status="ok",
        database="connected",
        llm_provider="groq",
    )
