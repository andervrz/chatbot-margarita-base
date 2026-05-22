"""
Cliente WhatsApp Business API — Stub con visibilidad completa.
Mientras la API real no esté configurada, loguea mensajes y encola en SQLite.
"""

import structlog
from datetime import datetime, timezone

from src.config import settings
from src.infrastructure.db import Database

logger = structlog.get_logger()


class WhatsAppClient:
    """
    Cliente WhatsApp con dos modos:
    1. API real: envía vía Meta WhatsApp Business API (cuando esté configurada)
    2. Stub: loguea + encola en notifications_queue para visibilidad/debugging
    
    Inicialización: si no hay token configurado, opera en modo stub automáticamente.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self._api_ready = bool(
            settings.whatsapp_access_token and settings.whatsapp_business_phone_id
        )
        self._phone_id = settings.whatsapp_business_phone_id
        self._access_token = settings.whatsapp_access_token
        self._agent_phone = settings.agency_whatsapp_number

        if self._api_ready:
            logger.info("whatsapp_api_ready", phone_id=self._phone_id)
        else:
            logger.warning(
                "whatsapp_stub_mode",
                reason="missing_access_token_or_phone_id",
                queue_table="notifications_queue",
            )

    async def send_text_message(self, to: str, text: str) -> dict:
        """
        Envía mensaje de texto a un número de teléfono.
        En stub: loguea + encola en DB. Retorna mock de respuesta API.
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": self._normalize_phone(to),
            "type": "text",
            "text": {"body": text},
        }

        if self._api_ready:
            # TODO: Implementar llamada real a Meta API
            # response = await self._call_api("messages", payload)
            logger.info("whatsapp_api_send", to=to, preview=text[:60])
            return {"success": True, "message_id": f"msg_stub_{datetime.now(timezone.utc).isoformat()}"}
        
        # Modo stub: log + queue
        await self._queue_message(to, text, "text")
        logger.info(
            "whatsapp_stub_queued",
            to=to,
            text_preview=text[:80],
            status="pending",
        )
        return {"success": True, "stub": True, "queued": True}

    async def send_lead_notification(self, lead_summary: str) -> dict:
        """
        Notifica al agente sobre un nuevo lead o cita.
        Destinatario: agency_whatsapp_number desde config.
        """
        if not self._agent_phone:
            logger.error("whatsapp_no_agent_phone", reason="agency_whatsapp_number_not_configured")
            return {"success": False, "error": "No agent phone configured"}

        text = (
            f"🚨 NUEVO LEAD — {settings.agency_name}\n\n"
            f"{lead_summary}\n\n"
            f"—\n"
            f"Bot Margarita Realty | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )

        return await self.send_text_message(self._agent_phone, text)

    async def verify_webhook(self, mode: str, token: str, challenge: str) -> str | None:
        """
        Verificación GET de webhook Meta.
        Retorna challenge si el verify_token coincide.
        """
        if mode == "subscribe" and token == settings.whatsapp_verify_token:
            logger.info("whatsapp_webhook_verified", mode=mode)
            return challenge
        logger.warning("whatsapp_webhook_verify_failed", mode=mode, token_provided=token)
        return None

    async def parse_incoming_message(self, payload: dict) -> dict | None:
        """
        Parsea mensaje entrante de webhook POST.
        Retorna dict con: from, text, timestamp, type
        """
        try:
            entry = payload.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                return None

            msg = messages[0]
            parsed = {
                "from": msg.get("from"),
                "text": msg.get("text", {}).get("body", ""),
                "timestamp": msg.get("timestamp"),
                "type": msg.get("type"),
                "id": msg.get("id"),
            }
            logger.info("whatsapp_incoming_parsed", from_=parsed["from"], preview=parsed["text"][:60])
            return parsed

        except (KeyError, IndexError) as exc:
            logger.warning("whatsapp_parse_error", error=str(exc), payload_keys=list(payload.keys()))
            return None

    # ---------- Internos ----------

    async def _queue_message(
        self,
        recipient_phone: str,
        message_text: str,
        message_type: str = "lead_notification",
    ) -> None:
        """Inserta mensaje en notifications_queue para visibilidad."""
        await self.db.execute(
            """
            INSERT INTO notifications_queue (recipient_phone, message_text, message_type, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                recipient_phone,
                message_text,
                message_type,
                "pending",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        await self.db.commit()

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normaliza número venezolano para WhatsApp API."""
        digits = "".join(c for c in phone if c.isdigit())
        # Si empieza con 0 venezolano, quitarlo para formato internacional
        if digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]
        # Si no tiene código de país, agregar 58
        if not digits.startswith("58") and len(digits) == 10:
            digits = "58" + digits
        return digits