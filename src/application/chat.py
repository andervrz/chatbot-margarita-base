"""
Orquestador principal del chat.
Flujo de defensa en profundidad + state machine de calificación Fase 1.
"""

import structlog

from src.application.enrichment import PromptEnricher
from src.application.intent import ExtractedFilters, IntentDetector, IntentType, PartialLead
from src.application.booking import BookingService
from src.domain.models import Message, MessageRole
from src.domain.exceptions import NoDataError
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository
from src.infrastructure.whatsapp import WhatsAppClient

logger = structlog.get_logger()


class ChatService:
    """
    Servicio de aplicación que coordina toda la lógica de conversación.
    Principio: si la info no está en SQLite, el bot no la genera.

    Fase 1: State machine de calificación suave
    - Paso 1: Tipo de propiedad
    - Paso 2: Zona
    - Paso 3: Presupuesto
    - Paso 4: Preferencias (vista al mar, céntrico, económico)
    - Paso 5: Agendamiento (2 turnos: lead → horario)
    """

    def __init__(
        self,
        llm: LLMClient,
        property_repo: PropertyRepository,
        cache: CacheManager,
        conv_store: ConversationStore,
        enricher: PromptEnricher,
        intent_detector: IntentDetector,
        booking_service: BookingService,
    ) -> None:
        self.llm = llm
        self.property_repo = property_repo
        self.cache = cache
        self.conv_store = conv_store
        self.enricher = enricher
        self.intent = intent_detector
        self.booking = booking_service

    async def handle(self, session_id: str, user_message: str) -> str:
        """Procesa un mensaje del usuario y retorna la respuesta."""
        logger.info("chat_handle", session_id=session_id, msg_preview=user_message[:60])

        # 1. Detectar intención
        intent = self.intent.detect(user_message)

        # 2. FAQ o saludo → respuesta directa (0 LLM calls)
        if faq_response := self.intent.get_faq_response(intent):
            await self._persist(session_id, user_message, faq_response)
            return faq_response

        # 3. Captura de lead (detectado por regex)
        if intent == IntentType.CAPTURE_LEAD:
            response = await self._handle_capture_lead(session_id, user_message)
            await self._persist(session_id, user_message, response)
            return response

        # 4. Agendamiento (Fase 1) — 2 turnos
        if intent == IntentType.BOOK_APPOINTMENT:
            response = await self._handle_booking_flow(session_id, user_message)
            await self._persist(session_id, user_message, response)
            return response

        # 5. Cache lookup
        if len(user_message.split()) >= 4:
            if cached := await self.cache.get(user_message):
                await self._persist(session_id, user_message, cached)
                return cached
        else:
            from src.infrastructure.cache import _normalize
            import hashlib
            normalized = _normalize(user_message)
            hash_key = hashlib.sha256(normalized.encode()).hexdigest()
            row = await self.property_repo.db.fetchone(
                "SELECT response FROM cache_exact WHERE query_hash = ?",
                (hash_key,),
            )
            if row:
                await self._persist(session_id, user_message, row["response"])
                return row["response"]

        # 6. Recuperar historial
        history = await self.conv_store.get_history(session_id, limit=10)

        # 7. Búsqueda de propiedades
        response: str
        is_fallback = False
        try:
            response = await self._handle_property_search(
                session_id=session_id,
                user_message=user_message,
                intent=intent,
                history=history,
            )
        except NoDataError:
            response = self.enricher.build_fallback_message()
            is_fallback = True

        # 8. Persistir
        await self._persist(session_id, user_message, response)

        # 9. Guardar en cache (excepto fallbacks)
        if not is_fallback:
            await self.cache.set(user_message, response, intent.value)

        return response

    # ---------- Flujo de agendamiento (Fase 1, 2 turnos) ----------

    async def _handle_booking_flow(self, session_id: str, user_message: str) -> str:
        """
        State machine de agendamiento:
        Turno 1: Si no hay lead con teléfono → pedir datos
        Turno 2: Si hay lead → pedir horario y crear cita
        """
        # Verificar si ya existe lead con teléfono para esta sesión
        lead_row = await self.property_repo.db.fetchone(
            "SELECT id, name, phone FROM leads WHERE session_id = ? AND phone IS NOT NULL ORDER BY captured_at DESC LIMIT 1",
            (session_id,),
        )

        if not lead_row:
            # Turno 1: No tenemos teléfono → pedir nombre + teléfono
            # Extraer datos que el usuario ya haya dado en este mensaje
            lead_info = self.intent.extract_lead_info(user_message)
            if lead_info.name or lead_info.phone:
                await self._persist_lead(session_id, lead_info)
            
            return (
                "¡Perfecto! Para agendar una visita necesito tus datos de contacto.\n\n"
                "¿Me puedes decir tu nombre completo y un teléfono de contacto? "
                "Un asesor te llamará para confirmar la visita."
            )

        # Turno 2: Ya tenemos lead → verificar si ya pidió horario
        existing_appointment = await self.booking.get_appointment_by_session(session_id)
        if existing_appointment and existing_appointment.requested_date:
            # Ya tiene cita pendiente
            return (
                f"{lead_row['name'] or 'Hola'}, ya tienes una visita registrada para "
                f"{existing_appointment.requested_date} {existing_appointment.requested_time or ''}. "
                f"Tu asesor te confirmará la disponibilidad por WhatsApp en las próximas horas."
            )

        # Extraer preferencia de horario del mensaje (regex simple)
        # Si no hay horario explícito, pedirlo
        date_time = self._extract_datetime(user_message)
        
        if not date_time:
            return (
                f"Gracias {lead_row['name'] or ''}. ¿Qué día y horario te funcionaría "
                f"para la visita? Por ejemplo: 'mañana a las 3pm' o 'sábado en la mañana'."
            )

        # Crear cita
        # Obtener última propiedad mostrada (de metadata de conversación o usar primera de búsqueda reciente)
        last_property_id = await self._get_last_shown_property(session_id)
        
        appointment = await self.booking.create_appointment(
            session_id=session_id,
            property_id=last_property_id,
            lead_id=lead_row["id"],
            requested_date=date_time.get("date"),
            requested_time=date_time.get("time"),
            notes=f"Solicitado por usuario: {user_message[:100]}",
        )

        return (
            f"¡Listo {lead_row['name'] or ''}! 🎯\n\n"
            f"He registrado tu solicitud de visita"
            f"{f' para el {date_time['date']}' if date_time.get('date') else ''}"
            f"{f' a las {date_time['time']}' if date_time.get('time') else ''}.\n\n"
            f"Tu asesor te confirmará la disponibilidad exacta por WhatsApp en las próximas horas.\n\n"
            f"—\n"
            f"Referencia de cita: #{appointment.id}"
        )

    # ---------- Flujo de captura de lead ----------

    async def _handle_capture_lead(
        self,
        session_id: str,
        user_message: str,
    ) -> str:
        lead_info = self.intent.extract_lead_info(user_message)
        await self._persist_lead(session_id, lead_info)

        if lead_info.name and lead_info.phone:
            return (
                f"Gracias {lead_info.name}, he registrado tus datos "
                f"(teléfono: {lead_info.phone}). Un asesor te contactará "
                f"en las próximas 24-48 horas con opciones personalizadas."
            )
        elif lead_info.name:
            return (
                f"Gracias {lead_info.name}, he registrado tu interés. "
                f"¿Podrías compartirme tu teléfono para que un asesor te contacte?"
            )
        else:
            return (
                "Gracias por tu interés. Para que un asesor te contacte, "
                "¿podrías decirme tu nombre y teléfono?"
            )

    # ---------- Búsqueda de propiedades ----------

    async def _handle_property_search(
        self,
        session_id: str,
        user_message: str,
        intent: IntentType,
        history: list[Message],
    ) -> str:
        filters = self.intent.extract_filters(user_message)


        # Mapeo de filtros semánticos a columnas reales
        semantic_filters = {}
        if filters.has_ocean_view is not None:
            semantic_filters['has_ocean_view'] = filters.has_ocean_view
        # is_urban → busca en zonas céntricas (ya cubierto por zone filter)
        # is_budget_friendly → ya cubierto por max_price

        # Búsqueda SQL exacta con nuevos filtros booleanos
        properties = await self.property_repo.search_exact(
            municipality=filters.municipality,
            zone=filters.zone,
            property_type=filters.property_type,
            min_price=filters.min_price,
            max_price=filters.max_price,
            min_bedrooms=filters.min_bedrooms,
            **semantic_filters,
        )
        
        search_type = "sql_exact"
        context = None

        if not properties:
            logger.info("sql_no_results", session_id=session_id, filters=filters)

            if self._has_meaningful_filters(filters):
                embedding = await self.llm.embed(user_message)
                properties = await self.property_repo.search_vector(embedding, limit=5)

                if properties:
                    search_type = "vector_fallback"
                    context = self.enricher.build_vector_context(properties, user_message)

        if properties:
            # Guardar referencia de propiedades mostradas para agendamiento posterior
            await self._save_last_shown_properties(session_id, properties)

            if not context:
                context = self.enricher.build_search_context(properties, user_message)

            messages = self._build_messages(history, context, user_message)
            response = await self.llm.chat(messages)
            logger.info(
                "llm_call",
                session_id=session_id,
                search_type=search_type,
                results=len(properties),
            )
            return response

        raise NoDataError("No hay propiedades en SQL ni en vectorial")

    @staticmethod
    def _has_meaningful_filters(filters: ExtractedFilters) -> bool:
        return any([
            filters.municipality,
            filters.zone,
            filters.property_type,
            filters.min_price is not None,
            filters.max_price is not None,
            filters.min_bedrooms is not None,
            filters.has_ocean_view,
            filters.is_urban,
            filters.is_budget_friendly,
        ])

    def _build_messages(
        self,
        history: list[Message],
        context: str,
        user_message: str,
    ) -> list[dict[str, str]]:
        system_prompt = self.enricher.build_system_prompt()
        system_content = f"{system_prompt}\n\n---\n\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        for msg in history:
            messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    # ---------- Persistencia ----------

    async def _persist(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        await self.conv_store.append(session_id, MessageRole.USER, user_msg)
        await self.conv_store.append(session_id, MessageRole.ASSISTANT, assistant_msg)

    async def _persist_lead(self, session_id: str, lead_info: PartialLead) -> None:
        columns = ["session_id"]
        values = [session_id]
        placeholders = ["?"]

        if lead_info.name:
            columns.append("name")
            values.append(lead_info.name)
            placeholders.append("?")
        if lead_info.phone:
            columns.append("phone")
            values.append(lead_info.phone)
            placeholders.append("?")
        if lead_info.email:
            columns.append("email")
            values.append(lead_info.email)
            placeholders.append("?")

        sql = f"INSERT INTO leads ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        await self.property_repo.db.execute(sql, values)
        await self.property_repo.db.commit()

        logger.info(
            "lead_persisted",
            session_id=session_id,
            name=lead_info.name,
            phone=lead_info.phone is not None,
            email=lead_info.email is not None,
        )

    # ---------- Helpers de agendamiento ----------

    async def _save_last_shown_properties(self, session_id: str, properties: list) -> None:
        """Guarda IDs de propiedades mostradas en metadata de conversación."""
        prop_ids = [p.id for p in properties if p.id]
        await self.conv_store.append(
            session_id,
            MessageRole.SYSTEM,
            "",
            metadata={"last_shown_properties": prop_ids},
        )

    async def _get_last_shown_property(self, session_id: str) -> int | None:
        """Recupera primera propiedad de la última búsqueda."""
        rows = await self.property_repo.db.fetchall(
            """
            SELECT metadata FROM conversations 
            WHERE session_id = ? AND role = 'system' AND metadata != '{}' 
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id,),
        )
        if rows:
            import json
            try:
                meta = json.loads(rows[0]["metadata"])
                prop_ids = meta.get("last_shown_properties", [])
                return prop_ids[0] if prop_ids else None
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _extract_datetime(self, text: str) -> dict[str, str] | None:
        """
        Regex simple para extraer día/hora de mensaje coloquial.
        Ej: "mañana a las 3pm" → {"date": "mañana", "time": "3pm"}
        """
        import re
        text_lower = text.lower()

        # Patrones comunes
        patterns = [
            r"(hoy|mañana|pasado\s+mañana|lunes|martes|miércoles|jueves|viernes|sábado|domingo)(?:\s+a\s+las\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)?",
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "date": match.group(1) if len(match.groups()) > 0 else None,
                    "time": match.group(2) if len(match.groups()) > 1 and match.group(2) else None,
                }
        return None