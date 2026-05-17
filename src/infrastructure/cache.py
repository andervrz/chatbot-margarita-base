"""
Cache multinivel para reducir llamadas al LLM.
Nivel 1: Hash exacto (SHA-256) en SQLite.
Nivel 2: Similitud semántica (sqlite-vec) con embeddings.
"""

import hashlib
import re
from datetime import datetime, timezone, timedelta

import structlog

from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient

logger = structlog.get_logger()


def _normalize(text: str) -> str:
    """Limpia el texto para la cache exacta."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class CacheManager:
    """
    Gestiona cache exacta y semántica.
    Si el usuario repite o reformula ligeramente una pregunta, evitamos el LLM.
    """

    def __init__(self, db: Database, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm

    async def get(self, query: str) -> str | None:
        """
        Busca en cache exacta primero, luego en semántica.
        Retorna None si no hay coincidencia.
        """
        normalized = _normalize(query)
        hash_key = hashlib.sha256(normalized.encode()).hexdigest()

        # --- Nivel 1: Exacta ---
        row = await self.db.fetchone(
            """
            SELECT response, hit_count, ttl_hours, last_hit_at
            FROM cache_exact
            WHERE query_hash = ?
            """,
            (hash_key,),
        )
        if row:
            ttl_hours = row["ttl_hours"] if row["ttl_hours"] is not None else 168
            # Usar formato consistente con SQLite CURRENT_TIMESTAMP (sin T, sin timezone)
            cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
            cutoff = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

            if row["last_hit_at"] is None or row["last_hit_at"] > cutoff:
                await self.db.execute(
                    """
                    UPDATE cache_exact
                    SET hit_count = hit_count + 1, last_hit_at = ?
                    WHERE query_hash = ?
                    """,
                    (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), hash_key),
                )
                await self.db.commit()
                logger.debug(
                    "cache_exact_hit",
                    hash_prefix=hash_key[:8],
                    hits=row["hit_count"] + 1,
                )
                return row["response"]

        # --- Nivel 2: Semántica ---
        try:
            embedding = await self.llm.embed(query)
            rows = await self.db.vec_search(
                table="cache_semantic",
                vector=embedding,
                vector_column="query_embedding",
                limit=1,
                max_distance=0.40,
            )
            if rows:
                result = rows[0]
                logger.debug("cache_semantic_hit", distance=result["distance"])
                return result["response"]
        except Exception as exc:
            logger.warning("cache_semantic_lookup_error", error=str(exc))

        logger.debug("cache_miss")
        return None

    async def set(
        self,
        query: str,
        response: str,
        intent_type: str | None = None,
    ) -> None:
        """Almacena la respuesta en ambas caches."""
        normalized = _normalize(query)
        hash_key = hashlib.sha256(normalized.encode()).hexdigest()
        # Formato consistente con SQLite CURRENT_TIMESTAMP
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Guardar cache exacta
        await self.db.execute(
            """
            INSERT INTO cache_exact (query_hash, response, intent_type, last_hit_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(query_hash) DO UPDATE SET
                response = excluded.response,
                last_hit_at = excluded.last_hit_at,
                hit_count = hit_count + 1
            """,
            (hash_key, response, intent_type, now),
        )

        # Guardar cache semántica
        try:
            embedding = await self.llm.embed(query)
            await self.db.vec_insert(
                table="cache_semantic",
                vector=embedding,
                vector_column="query_embedding",
                response=response,
                intent_type=intent_type,
            )
        except Exception as exc:
            logger.error("cache_semantic_set_error", error=str(exc))

        await self.db.commit()
        logger.debug("cache_stored", hash_prefix=hash_key[:8], intent=intent_type)
