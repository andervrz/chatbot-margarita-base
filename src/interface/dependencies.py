"""
Inyección de dependencias simple.
Inicializa y conecta todos los módulos de infraestructura y aplicación.
No usamos framework de DI; solo funciones explícitas que FastAPI puede inyectar.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from src.application.chat import ChatService
from src.application.enrichment import PromptEnricher
from src.application.intent import IntentDetector
from src.config import settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository

logger = structlog.get_logger()


class AppState:
    """
    Contenedor de estado de la aplicación.
    Mantiene una sola instancia de cada dependencia durante el lifecycle.
    """
    def __init__(self):
        self.db: Database | None = None
        self.llm: LLMClient | None = None
        self.property_repo: PropertyRepository | None = None
        self.cache: CacheManager | None = None
        self.conv_store: ConversationStore | None = None
        self.enricher: PromptEnricher | None = None
        self.intent_detector: IntentDetector | None = None
        self.chat_service: ChatService | None = None


# Singleton global del estado
app_state = AppState()


async def init_dependencies() -> None:
    """
    Inicializa todas las dependencias en orden correcto.
    Llama a connect() en la DB antes de crear repositorios que dependen de ella.
    """
    logger.info("init_dependencies_start")

    # 1. LLM (necesario para saber embedding_dim de la DB)
    app_state.llm = LLMClient()

    # 2. Base de datos (fuente de verdad, con dimensión de embeddings del LLM)
    app_state.db = Database(
        settings.database_path,
        embedding_dim=app_state.llm.embedding_dim,
    )
    await app_state.db.connect()

    # 3. Repositorios (dependen de DB)
    app_state.property_repo = PropertyRepository(app_state.db)

    # 4. Cache (depende de DB + LLM)
    app_state.cache = CacheManager(app_state.db, app_state.llm)

    # 5. Almacenamiento de conversaciones (depende de DB)
    app_state.conv_store = ConversationStore(app_state.db)

    # 6. IntentDetector con zone_map desde la base de datos (Fuente Única de Verdad)
    # Si la BD no tiene datos aún, usa el default para no romper el arranque
    try:
        zone_index = await app_state.property_repo.get_zone_index()
        if zone_index:
            app_state.intent_detector = IntentDetector(zone_map=zone_index)
            logger.info("intent_detector_zone_map_from_db", zones=len(zone_index))
        else:
            app_state.intent_detector = IntentDetector()  # Usa default
            logger.info("intent_detector_zone_map_default", reason="db_empty")
    except Exception as exc:
        logger.warning("intent_detector_zone_map_fallback", error=str(exc))
        app_state.intent_detector = IntentDetector()  # Fallback seguro

    # 7. Enricher (sin dependencias externas)
    app_state.enricher = PromptEnricher()

    # 8. Servicio de aplicación (orquesta todo)
    app_state.chat_service = ChatService(
        llm=app_state.llm,
        property_repo=app_state.property_repo,
        cache=app_state.cache,
        conv_store=app_state.conv_store,
        enricher=app_state.enricher,
        intent_detector=app_state.intent_detector,
    )

    logger.info("init_dependencies_complete")


async def close_dependencies() -> None:
    """Limpieza ordenada al apagar."""
    if app_state.db:
        await app_state.db.close()
    # Nota: LLMClient no tiene recursos que liberar en fase 0.
    # Si se agrega pool de conexiones HTTP, cerrar aquí.
    logger.info("dependencies_closed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan de FastAPI: init al arrancar, cleanup al cerrar."""
    await init_dependencies()
    yield
    await close_dependencies()


# ---------- Funciones de inyección para FastAPI ----------

def get_chat_service(request: Request) -> ChatService:
    """Inyecta ChatService en los endpoints."""
    if not app_state.chat_service:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.chat_service


def get_db(request: Request) -> Database:
    """Inyecta Database para endpoints administrativos."""
    if not app_state.db:
        raise RuntimeError("Dependencias no inicializadas")
    return app_state.db
