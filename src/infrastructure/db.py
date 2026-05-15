"""
Conexión SQLite async con extensión sqlite-vec cargada.
Única fuente de verdad: SQL estructurado + búsqueda vectorial en el mismo archivo .db
"""

from pathlib import Path

import aiosqlite
import sqlite_vec
import structlog

logger = structlog.get_logger()

SCHEMA_SQL = """
-- ============================================
-- PROPIEDADES (Fuente de verdad absoluta)
-- ============================================
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    municipality TEXT NOT NULL,
    zone TEXT NOT NULL,
    type TEXT NOT NULL,
    price_usd REAL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    area_m2 REAL,
    description TEXT NOT NULL DEFAULT '',
    features TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'available',
    contact_phone TEXT,
    contact_email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_properties_municipality ON properties(municipality);
CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(type);
CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status);
CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price_usd);

-- ============================================
-- CONVERSACIONES (Memoria persistente entre sesiones)
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT,
    role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, created_at);

-- ============================================
-- LEADS
-- ============================================
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    name TEXT,
    email TEXT,
    phone TEXT,
    interest_type TEXT,
    budget_usd REAL,
    preferred_zone TEXT,
    preferred_type TEXT,
    urgency TEXT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id);

-- ============================================
-- CACHE EXACTA
-- ============================================
CREATE TABLE IF NOT EXISTS cache_exact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE NOT NULL,
    response TEXT NOT NULL,
    intent_type TEXT,
    hit_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP,
    ttl_hours INTEGER DEFAULT 168
);

CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache_exact(query_hash);

-- ============================================
-- TABLAS VECTORIALES (sqlite-vec)
-- ============================================
CREATE VIRTUAL TABLE IF NOT EXISTS cache_semantic USING vec0(
    query_embedding float[384],
    +response TEXT,
    +intent_type TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS property_embeddings USING vec0(
    description_embedding float[384],
    +property_id INTEGER
);
"""


class Database:
    """Wrapper async sobre SQLite. Carga sqlite-vec automáticamente."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Cargar extensión C de sqlite-vec en la conexión subyacente
        await self._connection.enable_load_extension(True)
        sqlite_vec.load(self._connection._connection)
        await self._connection.enable_load_extension(False)

        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()
        logger.info("db_connected", path=str(self.db_path))

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("db_closed")

    async def execute(self, sql: str, parameters: tuple | list = ()) -> aiosqlite.Cursor:
        if not self._connection:
            raise RuntimeError("Database not connected. Call connect() first.")
        return await self._connection.execute(sql, parameters)

    async def fetchone(self, sql: str, parameters: tuple | list = ()) -> aiosqlite.Row | None:
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: tuple | list = ()) -> list[aiosqlite.Row]:
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchall()

    async def commit(self) -> None:
        if self._connection:
            await self._connection.commit()

    # ---------- sqlite-vec helpers ----------

    async def vec_insert(
        self,
        table: str,
        vector: list[float],
        vector_column: str = "embedding",
        **metadata,
    ) -> None:
        """Inserta un vector + metadatos en una tabla vec0."""
        vec_blob = sqlite_vec.serialize_float32(vector)
        cols = [vector_column] + list(metadata.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        params = [vec_blob] + list(metadata.values())
        await self.execute(sql, params)
        await self.commit()

    async def vec_search(
        self,
        table: str,
        vector: list[float],
        vector_column: str = "embedding",
        limit: int = 5,
        max_distance: float | None = None,
    ) -> list[dict]:
        """
        KNN search. Retorna filas con 'distance' (L2).
        Filtra por max_distance si se proporciona.
        """
        vec_blob = sqlite_vec.serialize_float32(vector)
        sql = f"""
        SELECT rowid, distance, *
        FROM {table}
        WHERE {vector_column} MATCH ?
        ORDER BY distance
        LIMIT ?
        """
        rows = await self.fetchall(sql, (vec_blob, limit))
        results = []
        for row in rows:
            dist = row["distance"]
            if max_distance is not None and dist > max_distance:
                continue
            results.append({**dict(row), "distance": dist})
        return results
