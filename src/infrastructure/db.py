"""
Conexión SQLite async con extensión sqlite-vec cargada.
"""

from pathlib import Path

import aiosqlite
import sqlite_vec
import structlog

logger = structlog.get_logger()


def _build_schema(embedding_dim: int) -> str:
    return f"""
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
    -- NUEVO Fase 1: Columnas booleanas para filtros (0=false, 1=true)
    has_ocean_view INTEGER DEFAULT 0,
    is_furnished INTEGER DEFAULT 0,
    has_pool INTEGER DEFAULT 0,
    has_parking INTEGER DEFAULT 0,
    has_security INTEGER DEFAULT 0,
    has_generator INTEGER DEFAULT 0,
    has_water_tank INTEGER DEFAULT 0,
    has_ac INTEGER DEFAULT 0,
    is_new_construction INTEGER DEFAULT 0,
    has_balcony INTEGER DEFAULT 0,
    is_gated_community INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_properties_municipality ON properties(municipality);
CREATE INDEX IF NOT EXISTS idx_properties_zone ON properties(zone);
CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(type);
CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status);
CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price_usd);
-- NUEVO Fase 1: Índices en columnas booleanas
CREATE INDEX IF NOT EXISTS idx_properties_ocean_view ON properties(has_ocean_view);
CREATE INDEX IF NOT EXISTS idx_properties_furnished ON properties(is_furnished);
CREATE INDEX IF NOT EXISTS idx_properties_pool ON properties(has_pool);
CREATE INDEX IF NOT EXISTS idx_properties_parking ON properties(has_parking);
CREATE INDEX IF NOT EXISTS idx_properties_security ON properties(has_security);
CREATE INDEX IF NOT EXISTS idx_properties_generator ON properties(has_generator);
CREATE INDEX IF NOT EXISTS idx_properties_water_tank ON properties(has_water_tank);
CREATE INDEX IF NOT EXISTS idx_properties_ac ON properties(has_ac);
CREATE INDEX IF NOT EXISTS idx_properties_new_construction ON properties(is_new_construction);
CREATE INDEX IF NOT EXISTS idx_properties_balcony ON properties(has_balcony);
CREATE INDEX IF NOT EXISTS idx_properties_gated ON properties(is_gated_community);

-- ============================================
-- CONVERSACIONES
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT,
    role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{{}}',
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
    has_rif INTEGER,
    visit_planned INTEGER,
    funding_source TEXT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id);

-- ============================================
-- CITAS (NUEVO Fase 1)
-- ============================================
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    property_id INTEGER,
    lead_id INTEGER,
    requested_date TEXT,
    requested_time TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_appointments_session ON appointments(session_id);
CREATE INDEX IF NOT EXISTS idx_appointments_status ON appointments(status);

-- ============================================
-- NOTIFICATIONS QUEUE (NUEVO Fase 1)
-- ============================================
CREATE TABLE IF NOT EXISTS notifications_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient_phone TEXT NOT NULL,
    message_text TEXT NOT NULL,
    message_type TEXT DEFAULT 'lead_notification',
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications_queue(status);
CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications_queue(created_at);

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
-- TABLAS VECTORIALES
-- ============================================
CREATE VIRTUAL TABLE IF NOT EXISTS cache_semantic USING vec0(
    query_embedding float[{embedding_dim}],
    +response TEXT,
    +intent_type TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS property_embeddings USING vec0(
    description_embedding float[{embedding_dim}],
    +property_id INTEGER
);
"""


class Database:
    """Wrapper async sobre SQLite. Carga sqlite-vec automáticamente."""

    def __init__(self, db_path: Path, embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._connection.enable_load_extension(True)
        await self._connection.load_extension(str(sqlite_vec.loadable_path()))
        await self._connection.enable_load_extension(False)

        schema = _build_schema(self.embedding_dim)
        await self._connection.executescript(schema)
        await self._connection.commit()
        logger.info("db_connected", path=str(self.db_path), embedding_dim=self.embedding_dim)

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
        vec_blob = sqlite_vec.serialize_float32(vector)
        cols = [vector_column] + list(metadata.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        params = [vec_blob] + list(metadata.values())
        await self.execute(sql, params)

    async def vec_search(
        self,
        table: str,
        vector: list[float],
        vector_column: str = "embedding",
        limit: int = 5,
        max_distance: float | None = None,
    ) -> list[dict]:
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