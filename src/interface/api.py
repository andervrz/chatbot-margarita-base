"""
API REST del bot inmobiliario.
Endpoints mínimos: chat y health. Manejo de errores limpio.
"""

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request

from src.application.chat import ChatService
from src.domain.exceptions import DomainError, LLMError
from src.interface.dependencies import lifespan, get_chat_service
from src.interface.schemas import ChatRequest, ChatResponse, HealthResponse
from src.logging_config import configure_logging
from src.config import settings

# Configurar logging al importar el módulo
configure_logging(env=settings.app_env, log_level=settings.log_level)

logger = structlog.get_logger()

app = FastAPI(
    title="Margarita Realty Bot",
    description="Chatbot inmobiliario para Isla de Margarita",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Añade request_id a cada log para trazabilidad."""
    from structlog.contextvars import bind_contextvars
    import uuid

    request_id = str(uuid.uuid4())[:8]
    bind_contextvars(request_id=request_id)
    response = await call_next(request)
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Endpoint principal de conversación.
    """
    try:
        logger.info(
            "chat_request",
            session_id=request.session_id,
            user_id=request.user_id,
            msg_preview=request.message[:60],
        )

        response_text = await service.handle(
            session_id=request.session_id,
            user_message=request.message,
        )

        logger.info("chat_response", session_id=request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
        )

    except LLMError as exc:
        logger.error("chat_llm_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Servicio de lenguaje no disponible")

    except DomainError as exc:
        logger.error("chat_domain_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Error interno del bot")

    except Exception as exc:
        logger.error("chat_unexpected_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Error inesperado")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="ok",
        database="connected",
        llm_provider="groq",
    )
