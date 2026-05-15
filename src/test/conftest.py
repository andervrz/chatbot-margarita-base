"""
Fixtures de pytest para testing async.
Base de datos en memoria, LLM mockeado, servicios aislados.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.application.chat import ChatService
from src.application.enrichment import PromptEnricher
from src.application.intent import IntentDetector
from src.config import Settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository


@pytest.fixture(scope="session")
def event_loop():
    """Un solo event loop para toda la sesión de tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db() -> Database:
    """Base de datos SQLite en memoria, limpia para cada test."""
    db = Database(Path(":memory:"))
    await db.connect()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def property_repo(db: Database) -> PropertyRepository:
    return PropertyRepository(db)


@pytest_asyncio.fixture
def mock_llm() -> LLMClient:
    """LLMClient con métodos mockeados. Nunca llama a la red."""
    llm = LLMClient.__new__(LLMClient)
    llm.chat_model = "groq/llama-3.3-70b-versatile"
    llm.embedding_model = "openai/text-embedding-3-small"
    llm.default_temperature = 0.7
    llm.default_max_tokens = 500

    # Mock: chat siempre retorna una respuesta fija
    llm.chat = AsyncMock(return_value="Respuesta mockeada del asistente")

    # Mock: embed retorna un vector de 384 dimensiones (tamaño estándar)
    llm.embed = AsyncMock(return_value=[0.01] * 384)

    return llm


@pytest_asyncio.fixture
async def cache(db: Database, mock_llm: LLMClient) -> CacheManager:
    return CacheManager(db, mock_llm)


@pytest_asyncio.fixture
async def conv_store(db: Database) -> ConversationStore:
    return ConversationStore(db)


@pytest_asyncio.fixture
def enricher() -> PromptEnricher:
    return PromptEnricher()


@pytest_asyncio.fixture
def intent_detector() -> IntentDetector:
    return IntentDetector()


@pytest_asyncio.fixture
async def chat_service(
    mock_llm: LLMClient,
    property_repo: PropertyRepository,
    cache: CacheManager,
    conv_store: ConversationStore,
    enricher: PromptEnricher,
    intent_detector: IntentDetector,
) -> ChatService:
    """Servicio completo con LLM mockeado. Listo para testear flujos."""
    return ChatService(
        llm=mock_llm,
        property_repo=property_repo,
        cache=cache,
        conv_store=conv_store,
        enricher=enricher,
        intent_detector=intent_detector,
    )
