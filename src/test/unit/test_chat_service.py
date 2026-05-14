"""
Tests del ChatService.
Verifica el flujo de defensa en profundidad:
FAQ directa → Cache → SQL exacta → Vectorial → LLM → Fallback.
"""

import pytest

from src.domain.models import Municipality, Property, PropertyStatus, PropertyType
from src.application.intent import IntentType


class TestChatServiceFlow:
    """Tests de integración ligera del servicio con infraestructura mockeada."""

    async def test_faq_returns_zero_llm_calls(
        self, chat_service, mock_llm
    ):
        """Una FAQ debe responder directamente sin tocar el LLM."""
        response = await chat_service.handle(
            session_id="test-001",
            user_message="cuánto cuesta el metro cuadrado",
        )
        assert "Maneiro" in response
        assert "$1,000" in response
        mock_llm.chat.assert_not_called()

    async def test_greeting_returns_zero_llm_calls(
        self, chat_service, mock_llm
    ):
        response = await chat_service.handle(
            session_id="test-002",
            user_message="hola",
        )
        assert "Hola" in response
        mock_llm.chat.assert_not_called()

    async def test_cache_avoids_llm_on_repeat(
        self, chat_service, mock_llm, property_repo
    ):
        """Si el usuario repite la pregunta, la cache debe interceptar."""
        # Primera vez: seed una propiedad para que SQL retorne algo
        prop = Property(
            title="Casa Test",
            municipality=Municipality.MANEIRO,
            zone="Pampatar",
            type=PropertyType.CASA,
            price_usd=50000,
            bedrooms=2,
            bathrooms=1,
            area_m2=100,
            description="Casa de prueba en Pampatar",
            status=PropertyStatus.AVAILABLE,
        )
        await property_repo.create(prop)

        msg = "casa en pampatar"

        # Primera llamada: debe llegar al LLM
        r1 = await chat_service.handle(session_id="test-003", user_message=msg)
        assert mock_llm.chat.call_count == 1

        # Segunda llamada idéntica: debe usar cache, no LLM
        r2 = await chat_service.handle(session_id="test-004", user_message=msg)
        assert r1 == r2
        assert mock_llm.chat.call_count == 1  # No aumentó

    async def test_sql_exact_triggers_llm_when_results_exist(
        self, chat_service, mock_llm, property_repo
    ):
        """Si SQL encuentra propiedades, debe haber exactamente 1 LLM call."""
        prop = Property(
            title="Apartamento Test",
            municipality=Municipality.MARINO,
            zone="Porlamar",
            type=PropertyType.APARTAMENTO,
            price_usd=60000,
            bedrooms=1,
            bathrooms=1,
            area_m2=50,
            description="Apartamento céntrico en Porlamar",
            status=PropertyStatus.AVAILABLE,
        )
        await property_repo.create(prop)

        response = await chat_service.handle(
            session_id="test-005",
            user_message="apartamento en porlamar",
        )
        assert mock_llm.chat.call_count == 1
        assert "Respuesta mockeada" in response

    async def test_fallback_when_no_data_anywhere(
        self, chat_service, mock_llm
    ):
        """Si no hay propiedades en SQL ni vectorial, fallback controlado. 0 LLM calls."""
        response = await chat_service.handle(
            session_id="test-006",
            user_message="castillo en la luna",
        )
        assert "No tengo propiedades registradas" in response
        mock_llm.chat.assert_not_called()

    async def test_session_memory_persists(
        self, chat_service, conv_store
    ):
        """Los mensajes se guardan en la base de datos."""
        await chat_service.handle(
            session_id="test-007",
            user_message="hola",
        )
        history = await conv_store.get_history("test-007")
        assert len(history) == 2  # user + assistant
        assert history[0].role.value == "user"
        assert history[1].role.value == "assistant"

    async def test_vector_fallback_used_when_sql_empty(
        self, chat_service, mock_llm, property_repo, db
    ):
        """
        Si SQL no retorna nada pero hay embeddings vectoriales,
        debe usar búsqueda vectorial y luego 1 LLM call.
        """
        prop = Property(
            title="Cerca de Playa",
            municipality=Municipality.GOMEZ,
            zone="Playa Caribe",
            type=PropertyType.CASA,
            price_usd=40000,
            bedrooms=2,
            bathrooms=1,
            area_m2=80,
            description="Casa muy cerca de la playa con vista al mar",
            status=PropertyStatus.AVAILABLE,
        )
        created = await property_repo.create(prop)

        # Insertar embedding manualmente para que el fallback vectorial funcione
        embedding = await mock_llm.embed("cerca de la playa con vista al mar")
        await db.vec_insert(
            table="property_embeddings",
            vector=embedding,
            vector_column="description_embedding",
            property_id=created.id,
        )

        # Consulta que no matchea SQL exacto pero sí semánticamente
        response = await chat_service.handle(
            session_id="test-008",
            user_message="algo con vista al mar y cerca de la orilla",
        )
        assert mock_llm.chat.call_count == 1
        assert "Respuesta mockeada" in response
