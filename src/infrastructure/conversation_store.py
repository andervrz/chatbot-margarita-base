"""
Almacenamiento de conversaciones en SQLite.
Permite retomar sesiones en cualquier momento.
"""

import json

import structlog

from src.domain.models import ConversationSession, Message, MessageRole
from src.infrastructure.db import Database

logger = structlog.get_logger()


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def append(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Guarda un mensaje en la sesión."""
        await self.db.execute(
            """
            INSERT INTO conversations (session_id, user_id, role, content, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                role.value,
                content,
                json.dumps(metadata or {}),
            ),
        )
        await self.db.commit()

    async def get_history(self, session_id: str, limit: int = 20) -> list[Message]:
        """
        Recupera los últimos N mensajes de una sesión, del más antiguo al más reciente.
        """
        rows = await self.db.fetchall(
            """
            SELECT role, content, metadata
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

        messages: list[Message] = []
        for row in reversed(rows):
            try:
                messages.append(
                    Message(
                        role=MessageRole(row["role"]),
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                )
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "conversation_row_corrupt",
                    session_id=session_id,
                    error=str(exc),
                    role_raw=row.get("role"),
                )
                continue

        return messages

    async def get_session(self, session_id: str) -> ConversationSession:
        """Recupera una sesión completa con su historial."""
        history = await self.get_history(session_id)
        return ConversationSession(
            session_id=session_id,
            messages=history,
        )
