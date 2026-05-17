"""
Orquestador principal del chat.
Flujo de defensa en profundidad para minimizar llamadas al LLM:
Intent → FAQ directa → Cache → SQL exacta → Vectorial (solo si hay filtros) → LLM (solo si hay datos) → Fallback controlado.
"""

import structlog

from src.application.enrichment import PromptEnricher
from src.application.intent import ExtractedFilters, IntentDetector, IntentType, PartialLead
from src.domain.models import Message, MessageRole
from src.domain.exceptions import NoDataError
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository

logger = structlog.get_logger()


class ChatService:
    """
    Servicio de aplicación que coordina toda la lógica de conversación.
    Principio: si la info no está en SQLite, el bot no la genera.
    """

    def __init__(
        self,
        llm: LLMClient,
        property_repo: PropertyRepository,
        cache: CacheManager,
        conv_store: ConversationStore,
        enricher: PromptEnricher,
        intent_detector: IntentDetector,
    ) -> None:
        self.llm = llm
        self.property_repo = property_repo
        self.cache = cache
        self.conv_store = conv_store
        self.enricher = enricher
        self.intent = intent_detector

    async def handle(self, session_id: str, user_message: str) -> str:
        """
        Procesa un mensaje del usuario y retorna la respuesta.
        """
        logger.info("chat_handle", session_id=session_id, msg_preview=user_message[:60])

        # 1. Detectar intención (sin LLM)
        intent = self.intent.detect(user_message)

        # 2. Si es FAQ o saludo → respuesta directa (0 LLM calls)
        if faq_response := self.intent.get_faq_response(intent):
            await self._persist(session_id, user_message, faq_response)
            return faq_response

        # 3. Captura de lead (detectado por regex, no FAQ)
        if intent == IntentType.CAPTURE_LEAD:
            response = await self._handle_capture_lead(session_id, user_message)
            await self._persist(session_id, user_message, response)
            return response

        # 4. Cache lookup (exacta + semántica)
        if cached := await self.cache.get(user_message):
            await self._persist(session_id, user_message, cached)
            return cached

        # 5. Recuperar historial de la sesión
        history = await self.conv_store.get_history(session_id, limit=10)

        # 6. Búsqueda de propiedades
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
            # 9. Fallback controlado (0 LLM calls)
            response = self.enricher.build_fallback_message()
            is_fallback = True

        # 10. Persistir conversación
        await self._persist(session_id, user_message, response)

        # 11. Guardar en cache para futuras consultas similares (excepto fallbacks)
        if not is_fallback:
            await self.cache.set(user_message, response, intent.value)

        return response

    # ---------- Flujo interno ----------

    async def _handle_capture_lead(
        self,
        session_id: str,
        user_message: str,
    ) -> str:
        """
        Procesa un mensaje de captura de lead.
        Extrae datos de contacto y persiste en la base de datos.
        """
        lead_info = self.intent.extract_lead_info(user_message)

        # Persistir lead en la base de datos
        await self._persist_lead(session_id, lead_info)

        # Respuesta personalizada según datos capturados
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

    async def _handle_property_search(
        self,
        session_id: str,
        user_message: str,
        intent: IntentType,
        history: list[Message],
    ) -> str:
        """
        Intenta resolver una búsqueda de propiedades.
        SQL exacta primero; vectorial como fallback SOLO si hay filtros extraídos.
        Si hay resultados, 1 LLM call para formatear.
        Si no hay, lanza NoDataError → fallback controlado.
        """

        # Extraer filtros del mensaje (regex, sin LLM)
        filters = self.intent.extract_filters(user_message)

        # 6a. Búsqueda SQL exacta
        properties = await self.property_repo.search_exact(
            municipality=filters.municipality,
            zone=filters.zone,
            property_type=filters.property_type,
            min_price=filters.min_price,
            max_price=filters.max_price,
            min_bedrooms=filters.min_bedrooms,
        )

        search_type = "sql_exact"
        context = None

        # 6b. Si no hay resultados SQL → fallback vectorial (solo si hay filtros semánticos)
        if not properties:
            logger.info("sql_no_results", session_id=session_id, filters=filters)

            # No gastar embedding si no hay nada que buscar semánticamente
            if self._has_meaningful_filters(filters):
                embedding = await self.llm.embed(user_message)
                properties = await self.property_repo.search_vector(embedding, limit=5)

                if properties:
                    search_type = "vector_fallback"
                    context = self.enricher.build_vector_context(properties, user_message)

        # 7. Si hay propiedades (de cualquier fuente) → enriquecer prompt y 1 LLM call
        if properties:
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

        # 8. Sin resultados en ningún lado
        raise NoDataError("No hay propiedades en SQL ni en vectorial")

    @staticmethod
    def _has_meaningful_filters(filters: ExtractedFilters) -> bool:
        """Determina si los filtros extraídos tienen valor semántico para búsqueda vectorial."""
        return any([
            filters.municipality,
            filters.zone,
            filters.property_type,
            filters.min_price is not None,
            filters.max_price is not None,
            filters.min_bedrooms is not None,
        ])

    def _build_messages(
        self,
        history: list[Message],
        context: str,
        user_message: str,
    ) -> list[dict[str, str]]:
        """Construye el payload de mensajes para LiteLLM."""
        system_prompt = self.enricher.build_system_prompt()

        # Un solo mensaje system para compatibilidad con Llama y otros modelos
        system_content = f"{system_prompt}\n\n---\n\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Historial reciente (últimos 10 mensajes)
        for msg in history:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Mensaje actual del usuario
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _persist(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Guarda el par mensaje/respuesta en la base de datos."""
        await self.conv_store.append(session_id, MessageRole.USER, user_msg)
        await self.conv_store.append(session_id, MessageRole.ASSISTANT, assistant_msg)

    async def _persist_lead(self, session_id: str, lead_info: PartialLead) -> None:
        """Guarda datos de lead en la tabla leads.

        Nota: En fase 0 se inserta un nuevo registro por mensaje.
        En fase 1 se puede implementar upsert por session_id.
        """
        # Construir INSERT dinámico según datos disponibles
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
