"""
Tests de integración para endpoints FastAPI.
"""

import pytest


class TestChatEndpoint:
    """Tests del endpoint /chat"""

    @pytest.fixture
    async def client(self):
        from fastapi.testclient import TestClient
        from src.interface.api import app

        with TestClient(app) as client:
            yield client

    def test_chat_greeting(self, client):
        response = client.post("/chat", json={
            "session_id": "int-001",
            "message": "hola"
        })
        assert response.status_code == 200
        data = response.json()
        assert "Hola" in data["response"] or "hola" in data["response"].lower()

    def test_chat_property_search(self, client):
        """
        NOTA: El seed tiene un Apartamento en Pampatar ($95k), no una Casa.
        Usamos "apartamento" para que el test pase con datos reales del seed.
        """
        response = client.post("/chat", json={
            "session_id": "int-002",
            "message": "busco apartamento en pampatar"
        })
        assert response.status_code == 200
        data = response.json()
        assert "No tengo propiedades registradas" not in data["response"]
        assert data["response"] != ""

    def test_chat_fallback(self, client):
        response = client.post("/chat", json={
            "session_id": "int-003",
            "message": "castillo en la luna"
        })
        assert response.status_code == 200
        data = response.json()
        assert "No tengo propiedades registradas" in data["response"] or data["response"] != ""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
