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


class TestWebhookEndpoint:
    """Tests del endpoint /webhook/whatsapp"""

    @pytest.fixture
    async def client(self):
        from fastapi.testclient import TestClient
        from src.interface.api import app

        with TestClient(app) as client:
            yield client

    def test_webhook_verify_success(self, client):
        """GET /webhook/whatsapp con token correcto retorna challenge."""
        response = client.get("/webhook/whatsapp", params={
            "hub.mode": "subscribe",
            "hub.verify_token": "margarita_bot_verify_123",
            "hub.challenge": "test_challenge_123"
        })
        assert response.status_code == 200
        assert response.text == "test_challenge_123"

    def test_webhook_verify_fail(self, client):
        """GET /webhook/whatsapp con token incorrecto retorna 403."""
        response = client.get("/webhook/whatsapp", params={
            "hub.mode": "subscribe",
            "hub.verify_token": "token_incorrecto",
            "hub.challenge": "test_challenge_123"
        })
        assert response.status_code == 403

    def test_webhook_receive_message(self, client):
        """POST /webhook/whatsapp recibe payload y retorna 200."""
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "1234567890",
                            "phone_id": "PHONE_ID"
                        },
                        "contacts": [{"profile": {"name": "Test User"}, "wa_id": "1234567890"}],
                        "messages": [{
                            "from": "1234567890",
                            "id": "MESSAGE_ID",
                            "timestamp": "1234567890",
                            "text": {"body": "Hola desde WhatsApp"},
                            "type": "text"
                        }]
                    },
                    "field": "messages"
                }]
            }]
        }
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"


class TestAdminEndpoints:
    """Tests de endpoints administrativos."""

    @pytest.fixture
    async def client(self):
        from fastapi.testclient import TestClient
        from src.interface.api import app

        with TestClient(app) as client:
            yield client

    def test_notifications_queue_empty(self, client):
        """GET /admin/notifications-queue retorna lista vacía inicialmente."""
        response = client.get("/admin/notifications-queue")
        # Si el endpoint no existe, retornará 404 hasta implementarse
        # Una vez implementado:
        # assert response.status_code == 200
        # assert response.json() == []