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
    db: Database | None = None
    llm: LLMClient | None = None
    property_repo: PropertyRepository | None = None
    cache: CacheManager | None = None
    conv_store: ConversationStore | None = None
    enricher: PromptEnricher | None = None
    intent_detector: IntentDetector | None = None
    chat_service: ChatService | None = None


# Singleton global del estado
app_state = AppState()


async def init_dependencies() -> None:
    """
    Inicializa todas las dependencias en orden correcto.
    Llama a connect() en la DB antes de crear repositorios que dependen de ella.
    """
    logger.info("init_dependencies_start")

    # 1. Base de datos (fuente de verdad)
    app_state.db = Database(settings.database_path)
    await app_state.db.connect()

    # 2. LLM (LiteLLM + Groq para chat, OpenAI para embeddings)
    app_state.llm = LLMClient()

    # 3. Repositorios (dependen de DB)
    app_state.property_repo = PropertyRepository(app_state.db)

    # 4. Cache (depende de DB + LLM)
    app_state.cache = CacheManager(app_state.db, app_state.llm)

    # 5. Almacenamiento de conversaciones (depende de DB)
    app_state.conv_store = ConversationStore(app_state.db)

    # 6. Componentes sin dependencias externas
    app_state.enricher = PromptEnricher()
    app_state.intent_detector = IntentDetector()

    # 7. Servicio de aplicación (orquesta todo)
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
