"""
Tests del ChatService.
Verifica el flujo de defensa en profundidad + agendamiento Fase 1.
"""

import pytest

from src.domain.models import Municipality, Property, PropertyStatus, PropertyType
from src.application.intent import IntentType


class TestChatServiceFlow:
    """Tests de integración ligera del servicio con infraestructura mockeada."""

    async def test_faq_returns_zero_llm_calls(
        self, chat_service, mock_llm
    ):
        """Una FAQ debe responder directamente sin tocar el LLM ni embeddings."""
        response = await chat_service.handle(
            session_id="test-001",
            user_message="cuánto cuesta el metro cuadrado",
        )
        assert "Maneiro" in response
        assert "$1,000" in response
        mock_llm.chat.assert_not_called()
        mock_llm.embed.assert_not_called()

    async def test_greeting_returns_zero_llm_calls(
        self, chat_service, mock_llm
    ):
        response = await chat_service.handle(
            session_id="test-002",
            user_message="hola",
        )
        assert "Hola" in response
        mock_llm.chat.assert_not_called()
        mock_llm.embed.assert_not_called()

    async def test_cache_avoids_llm_on_repeat(
        self, chat_service, mock_llm, property_repo
    ):
        """Si el usuario repite la pregunta, la cache debe interceptar."""
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

        r1 = await chat_service.handle(session_id="test-003", user_message=msg)
        assert mock_llm.chat.call_count == 1

        r2 = await chat_service.handle(session_id="test-004", user_message=msg)
        assert r1 == r2
        assert mock_llm.chat.call_count == 1

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
        """Fallback cuando no hay datos ni filtros semánticos."""
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
        assert len(history) == 2
        assert history[0].role.value == "user"
        assert history[1].role.value == "assistant"

    async def test_vector_fallback_used_when_sql_empty(
        self, chat_service, mock_llm, property_repo, db
    ):
        """Si SQL no retorna nada pero hay embeddings vectoriales, usa fallback."""
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

        embedding = await mock_llm.embed("cerca de la playa con vista al mar")
        await db.vec_insert(
            table="property_embeddings",
            vector=embedding,
            vector_column="description_embedding",
            property_id=created.id,
        )
        await db.commit()

        response = await chat_service.handle(
            session_id="test-008",
            user_message="algo con vista al mar y cerca de la orilla",
        )
        assert mock_llm.chat.call_count == 1
        assert "Respuesta mockeada" in response


class TestChatServiceBookingFlow:
    """Tests del flujo de agendamiento Fase 1 (2 turnos)."""

    async def test_book_appointment_without_lead_asks_for_data(
        self, chat_service, mock_llm
    ):
        """
        Turno 1: Usuario dice "quiero verla" sin haber dado datos.
        Bot debe pedir nombre + teléfono.
        """
        response = await chat_service.handle(
            session_id="test-book-001",
            user_message="quiero verla",
        )
        assert "nombre completo" in response.lower() or "teléfono" in response.lower()
        mock_llm.chat.assert_not_called()

    async def test_book_appointment_with_lead_asks_for_schedule(
        self, chat_service, mock_llm, db
    ):
        """
        Turno 2: Usuario ya dio datos previamente.
        Bot debe pedir horario.
        """
        # Pre-crear lead
        await db.execute(
            "INSERT INTO leads (session_id, name, phone) VALUES (?, ?, ?)",
            ("test-book-002", "Juan Pérez", "0412-1234567")
        )
        await db.commit()

        response = await chat_service.handle(
            session_id="test-book-002",
            user_message="quiero verla",
        )
        assert "horario" in response.lower() or "día" in response.lower()
        mock_llm.chat.assert_not_called()

    async def test_book_appointment_creates_pending_appointment(
        self, chat_service, mock_llm, db, property_repo
    ):
        """
        Turno 2 completo: Usuario con lead pide horario.
        Se crea cita con status='pending'.
        """
        # Pre-crear lead y propiedad
        await db.execute(
            "INSERT INTO leads (session_id, name, phone) VALUES (?, ?, ?)",
            ("test-book-003", "María García", "0414-9876543")
        )
        lead_cursor = await db.execute("SELECT last_insert_rowid()")
        lead_row = await lead_cursor.fetchone()
        lead_id = lead_row[0]

        prop = Property(
            title="Casa Test Booking",
            municipality=Municipality.MANEIRO,
            zone="Pampatar",
            type=PropertyType.CASA,
            price_usd=95000,
            bedrooms=3,
            bathrooms=2,
            area_m2=120,
            description="Casa para test de agendamiento",
            status=PropertyStatus.AVAILABLE,
        )
        created = await property_repo.create(prop)

        # Guardar propiedad como "última mostrada"
        await db.execute(
            "INSERT INTO conversations (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            ("test-book-003", "system", "", '{"last_shown_properties": [' + str(created.id) + ']}')
        )
        await db.commit()

        response = await chat_service.handle(
            session_id="test-book-003",
            user_message="mañana a las 3pm",
        )
        # Verificar que se creó la cita
        row = await db.fetchone(
            "SELECT * FROM appointments WHERE session_id = ?",
            ("test-book-003",)
        )
        assert row is not None
        assert row["status"] == "pending"
        assert "confirmará" in response.lower()

    async def test_capture_lead_persists_data(
        self, chat_service, db
    ):
        """CAPTURE_LEAD persiste nombre y teléfono en tabla leads."""
        response = await chat_service.handle(
            session_id="test-lead-001",
            user_message="me llamo Carlos Rodríguez, mi teléfono es 0412-555-8877",
        )
        row = await db.fetchone(
            "SELECT * FROM leads WHERE session_id = ?",
            ("test-lead-001",)
        )
        assert row is not None
        assert row["name"] == "Carlos Rodríguez"
        assert row["phone"] == "04125558877"

    async def test_booking_with_property_reference(
        self, chat_service, mock_llm, db, property_repo
    ):
        """Agendamiento guarda referencia a propiedad específica."""
        # Crear propiedad
        prop = Property(
            title="Penthouse Test",
            municipality=Municipality.MANEIRO,
            zone="Playa El Ángel",
            type=PropertyType.PENTHOUSE,
            price_usd=290000,
            bedrooms=5,
            bathrooms=7,
            area_m2=435,
            description="Penthouse de lujo",
            status=PropertyStatus.AVAILABLE,
        )
        created = await property_repo.create(prop)

        # Crear lead
        await db.execute(
            "INSERT INTO leads (session_id, name, phone) VALUES (?, ?, ?)",
            ("test-book-004", "Ana López", "0416-9990000")
        )
        lead_cursor = await db.execute("SELECT last_insert_rowid()")
        lead_row = await lead_cursor.fetchone()
        lead_id = lead_row[0]

        # Simular que se mostró la propiedad
        await db.execute(
            "INSERT INTO conversations (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            ("test-book-004", "system", "", '{"last_shown_properties": [' + str(created.id) + ']}')
        )
        await db.commit()

        response = await chat_service.handle(
            session_id="test-book-004",
            user_message="quiero verla",
        )
        
        # La cita debe tener property_id
        row = await db.fetchone(
            "SELECT property_id FROM appointments WHERE session_id = ?",
            ("test-book-004",)
        )
        assert row is not None
        assert row["property_id"] == created.id