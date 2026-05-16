#!/usr/bin/env python3
"""
Script de carga inicial de propiedades en SQLite.
Datos representativos del mercado inmobiliario de Isla de Margarita 2025-2026.
Ejecutar una sola vez: python scripts/seed_db.py
"""

import asyncio
import json
from pathlib import Path

import structlog

# Ajustar path para imports desde scripts/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.domain.models import Municipality, Property, PropertyStatus, PropertyType
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository

logger = structlog.get_logger()

# ... SEED_PROPERTIES sin cambios ...

async def seed_properties() -> None:
    """Inserta propiedades y genera sus embeddings vectoriales."""
    llm = LLMClient()
    db = Database(settings.database_path, embedding_dim=llm.embedding_dim)
    await db.connect()

    repo = PropertyRepository(db)

    logger.info("seed_start", count=len(SEED_PROPERTIES))

    for data in SEED_PROPERTIES:
        prop = Property(**data)
        created = await repo.create(prop)
        logger.info("property_inserted", id=created.id, title=created.title)

        # Generar embedding de la descripción para búsqueda vectorial
        if created.id:
            embedding = await llm.embed(created.description)
            await db.vec_insert(
                table="property_embeddings",
                vector=embedding,
                vector_column="description_embedding",
                property_id=created.id,
            )
            logger.info("embedding_inserted", property_id=created.id)

    await db.commit()
    await db.close()
    logger.info("seed_complete")


if __name__ == "__main__":
    asyncio.run(seed_properties())
