"""
API REST del bot inmobiliario.
Endpoints: chat, health, webhook, admin. Manejo de errores limpio.
"""

import uuid

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.application.chat import ChatService
from src.domain.exceptions import DomainError, LLMError
from src.interface.dependencies import lifespan, get_chat_service, app_state, get_db
from src.interface.schemas import ChatRequest, ChatResponse, HealthResponse
from src.interface.webhook import router as webhook_router

logger = structlog.get_logger()

app = FastAPI(
    title="Margarita Realty Bot",
    description="Chatbot inmobiliario para Isla de Margarita",
    version="0.1.0",
    lifespan=lifespan,
)

# NUEVO: Servir archivos estáticos (interfaz web)
# Acceso: http://localhost:8000/static/chat_web.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Incluir router de webhook (Fase 1)
app.include_router(webhook_router)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    clear_contextvars()
    request_id = str(uuid.uuid4())[:8]
    bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
        return response
    finally:
        clear_contextvars()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
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
    db_status = "connected" if (app_state.db and app_state.db._connection) else "disconnected"
    
    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        llm_provider="groq",
    )


# ============================================
# ADMIN ENDPOINTS (NUEVO Fase 1)
# ============================================

@app.get("/admin/notifications-queue")
async def get_notifications_queue(
    db = Depends(get_db),
    limit: int = 20,
):
    """
    Retorna mensajes WhatsApp pendientes de la cola.
    Usado por el comando /queue en la interfaz web.
    """
    rows = await db.fetchall(
        """
        SELECT id, recipient_phone, message_text, message_type, status, created_at
        FROM notifications_queue
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [
        {
            "id": r["id"],
            "recipient_phone": r["recipient_phone"],
            "message_text": r["message_text"],
            "message_type": r["message_type"],
            "status": r["status"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@app.get("/admin/appointments")
async def get_appointments(
    db = Depends(get_db),
    status: str | None = None,
    limit: int = 50,
):
    """
    Dashboard de citas. Filtrable por status.
    """
    if status:
        rows = await db.fetchall(
            "SELECT * FROM appointments WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
    else:
        rows = await db.fetchall(
            "SELECT * FROM appointments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    
    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "property_id": r["property_id"],
            "lead_id": r["lead_id"],
            "requested_date": r["requested_date"],
            "requested_time": r["requested_time"],
            "status": r["status"],
            "notes": r["notes"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@app.post("/admin/appointments/{appointment_id}/confirm")
async def confirm_appointment(
    appointment_id: int,
    db = Depends(get_db),
):
    """
    Confirmar una cita manualmente (solo humanos).
    """
    await db.execute(
        "UPDATE appointments SET status = 'confirmed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (appointment_id,),
    )
    await db.commit()
    logger.info("appointment_confirmed_admin", appointment_id=appointment_id)
    return {"status": "confirmed", "appointment_id": appointment_id}


@app.post("/admin/appointments/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: int,
    db = Depends(get_db),
):
    """
    Cancelar una cita manualmente.
    """
    await db.execute(
        "UPDATE appointments SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (appointment_id,),
    )
    await db.commit()
    logger.info("appointment_cancelled_admin", appointment_id=appointment_id)
    return {"status": "cancelled", "appointment_id": appointment_id}