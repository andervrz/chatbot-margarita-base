"""
Endpoints FastAPI para webhook de WhatsApp Business API.
Stub: retorna 200 OK y loguea payload para debugging.
"""

from fastapi import APIRouter, Request, Response

import structlog

from src.infrastructure.whatsapp import WhatsAppClient

logger = structlog.get_logger()

router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: str,
    hub_verify_token: str,
    hub_challenge: str,
    whatsapp: WhatsAppClient,
) -> Response:
    """
    Verificación GET de Meta para webhook.
    Meta envía: ?hub.mode=subscribe&hub.verify_token=XXX&hub.challenge=YYY
    """
    result = await whatsapp.verify_webhook(hub_mode, hub_verify_token, hub_challenge)
    if result:
        return Response(content=result, media_type="text/plain")
    return Response(status_code=403)


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    whatsapp: WhatsAppClient,
) -> dict:
    """
    Recibe mensajes entrantes, delivery receipts, read receipts de Meta.
    Stub: loguea payload y retorna 200 OK.
    """
    try:
        payload = await request.json()
        logger.info("whatsapp_webhook_received", payload_type=type(payload).__name__)

        # Parsear mensaje entrante
        parsed = await whatsapp.parse_incoming_message(payload)
        if parsed:
            logger.info(
                "whatsapp_message_parsed",
                from_=parsed.get("from"),
                text_preview=parsed.get("text", "")[:60],
                msg_type=parsed.get("type"),
            )
        else:
            logger.debug("whatsapp_webhook_no_messages", payload_keys=list(payload.keys()))

        return {"status": "received"}

    except Exception as exc:
        logger.error("whatsapp_webhook_error", error=str(exc))
        return {"status": "error", "detail": str(exc)}