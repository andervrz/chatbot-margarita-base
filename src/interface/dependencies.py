"""
Inyección de dependencias simple.
Inicializa y conecta todos los módulos de infraestructura y aplicación.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from src.application.chat import ChatService
from src.application.enrichment import PromptEnricher
from src.application.intent import IntentDetector
from src.application.booking import BookingService
from src.config import settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository
from src.infrastructure.whatsapp import WhatsAppClient

logger = structlog.get_logger()


class AppState:
    def __init__(self):
        self.db: Database | None = None
        self.llm: LLMClient | None = None
        self.property_repo: PropertyRepository | None = None
        self.cache: CacheManager | None = None
        self.conv_store: ConversationStore | None = None
        self.enricher: PromptEnricher | None = None
        self.intent_detector: IntentDetector | None = None
        self.whatsapp: WhatsAppClient | None = None
        self.booking_service: BookingService | None = None
        self.chat_service: ChatService | None = None


app_state = AppState()


async def init_dependencies() -> None:
    logger.info("init_dependencies_start")

    # 1. LLM
    app_state.llm = LLMClient()

    # 2. Base de datos
    app_state.db = Database(
        settings.database_path,
        embedding_dim=app_state.llm.embedding_dim,
    )
    await app_state.db.connect()

    # 3. Repositorios
    app_state.property_repo = PropertyRepository(app_state.db)

    # 4. Cache
    app_state.cache = CacheManager(app_state.db, app_state.llm)

    # 5. Conversaciones
    app_state.conv_store = ConversationStore(app_state.db)

    # 6. WhatsApp client (stub o real)
    app_state.whatsapp = WhatsAppClient(app_state.db)

    # 7. Booking service
    app_state.booking_service = BookingService(app_state.db, app_state.whatsapp)

    # 8. IntentDetector con zone_map desde DB
    try:
        zone_index = await app_state.property_repo.get_zone_index()
        if zone_index:
            app_state.intent_detector = IntentDetector(zone_map=zone_index)
            logger.info("intent_detector_zone_map_from_db", zones=len(zone_index))
        else:
            app_state.intent_detector = IntentDetector()
            logger.info("intent_detector_zone_map_default", reason="db_empty")
    except Exception as exc:
        logger.warning("intent_detector_zone_map_fallback", error=str(exc))
        app_state.intent_detector = IntentDetector()

    # 9. Enricher
    app_state.enricher = PromptEnricher()

    # 10. ChatService (orquestador principal)
    app_state.chat_service = ChatService(
        llm=app_state.llm,
        property_repo=app_state.property_repo,
        cache=app_state.cache,
        conv_store=app_state.conv_store,
        enricher=app_state.enricher,
        intent_detector=app_state.intent_detector,
        booking_service=app_state.booking_service,
    )

    logger.info("init_dependencies_complete")


async def close_dependencies() -> None:
    if app_state.db:
        await app_state.db.close()
    logger.info("dependencies_closed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_dependencies()
    yield
    await close_dependencies()


# ---------- Funciones de inyección ----------

def get_chat_service(request: Request) -> ChatService:
    if not app_state.chat_service:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.chat_service


def get_db(request: Request) -> Database:
    if not app_state.db:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.db


def get_whatsapp(request: Request) -> WhatsAppClient:
    if not app_state.whatsapp:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.whatsapp


def get_booking_service(request: Request) -> BookingService:
    if not app_state.booking_service:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.booking_service