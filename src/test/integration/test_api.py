"""
Tests de integración de la API FastAPI.
Usan TestClient con base de datos en memoria inyectada.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from src.interface.api import app
from src.interface.dependencies import app_state, init_dependencies, close_dependencies


@pytest.fixture(scope="module")
async def client():
    """
    Cliente de test con lifespan completo.
    Inicializa dependencias reales pero con DB en memoria.
    """
    # Sobrescribir DB path a memoria antes de init
    from src.config import settings
    original_path = settings.database_path
    settings.database_path = Path(":memory:")

    await init_dependencies()

    with TestClient(app) as c:
        yield c

    await close_dependencies()
    settings.database_path = original_path


class TestChatEndpoint:
    def test_chat_faq(self, client):
        """POST /chat responde FAQ sin error."""
        response = client.post(
            "/chat",
            json={"session_id": "int-001", "message": "hola"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "int-001"
        assert "Hola" in data["response"]

    def test_chat_property_search(self, client):
        """POST /chat con búsqueda de propiedad."""
        response = client.post(
            "/chat",
            json={"session_id": "int-002", "message": "busco casa en pampatar"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data

    def test_chat_invalid_body(self, client):
        """Validación Pydantic rechaza body inválido."""
        response = client.post(
            "/chat",
            json={"session_id": "", "message": ""},
        )
        assert response.status_code == 422

    def test_health(self, client):
        """GET /health retorna ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["llm_provider"] == "groq"
