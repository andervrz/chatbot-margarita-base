"""
Orquestador principal del chat.
Flujo de defensa en profundidad para minimizar llamadas al LLM:
Intent → FAQ directa → Cache → SQL exacta → Vectorial (solo si hay filtros) → LLM (solo si hay datos) → Fallback controlado.
"""

import structlog

from src.application.enrichment import PromptEnricher
from src.application.intent import ExtractedFilters, IntentDetector, IntentType
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

        # 3. Cache lookup (exacta + semántica)
        if cached := await self.cache.get(user_message):
            await self._persist(session_id, user_message, cached)
            return cached

        # 4. Recuperar historial de la sesión
        history = await self.conv_store.get_history(session_id, limit=10)

        # 5. Búsqueda de propiedades
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
            # 8. Fallback controlado (0 LLM calls)
            response = self.enricher.build_fallback_message()
            is_fallback = True

        # 9. Persistir conversación
        await self._persist(session_id, user_message, response)

        # 10. Guardar en cache para futuras consultas similares (excepto fallbacks)
        if not is_fallback:
            await self.cache.set(user_message, response, intent.value)

        return response

    # ---------- Flujo interno ----------

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

        # 5a. Búsqueda SQL exacta
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

        # 5b. Si no hay resultados SQL → fallback vectorial (solo si hay filtros semánticos)
        if not properties:
            logger.info("sql_no_results", session_id=session_id, filters=filters)
            
            # No gastar embedding si no hay nada que buscar semánticamente
            if self._has_meaningful_filters(filters):
                embedding = await self.llm.embed(user_message)
                properties = await self.property_repo.search_vector(embedding, limit=5)

                if properties:
                    search_type = "vector_fallback"
                    context = self.enricher.build_vector_context(properties, user_message)

        # 6. Si hay propiedades (de cualquier fuente) → enriquecer prompt y 1 LLM call
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

        # 7. Sin resultados en ningún lado
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
