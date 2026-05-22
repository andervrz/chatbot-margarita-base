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
from src.application.booking import BookingService
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository
from src.infrastructure.whatsapp import WhatsAppClient

@pytest.fixture(scope="session")
def event_loop():
    """Un solo event loop para toda la sesión de tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db(mock_llm: LLMClient) -> Database:
    """Base de datos SQLite en memoria, limpia para cada test."""
    db = Database(Path(":memory:"), embedding_dim=mock_llm.embedding_dim)
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
    llm.embedding_model = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
    llm.default_temperature = 0.7
    llm.default_max_tokens = 500
    llm.embedding_dim = 384

    llm.chat = AsyncMock(return_value="Respuesta mockeada del asistente")
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
async def whatsapp_client(db: Database) -> WhatsAppClient:
    """WhatsAppClient en modo stub (sin tokens configurados)."""
    return WhatsAppClient(db)


@pytest_asyncio.fixture
async def booking_service(db: Database, whatsapp_client: WhatsAppClient) -> BookingService:
    """BookingService con WhatsApp stub."""
    return BookingService(db, whatsapp_client)


@pytest_asyncio.fixture
async def chat_service(
    mock_llm: LLMClient,
    property_repo: PropertyRepository,
    cache: CacheManager,
    conv_store: ConversationStore,
    enricher: PromptEnricher,
    intent_detector: IntentDetector,
    booking_service: BookingService,
) -> ChatService:
    """Servicio completo con LLM mockeado y BookingService."""
    return ChatService(
        llm=mock_llm,
        property_repo=property_repo,
        cache=cache,
        conv_store=conv_store,
        enricher=enricher,
        intent_detector=intent_detector,
        booking_service=booking_service,
    )