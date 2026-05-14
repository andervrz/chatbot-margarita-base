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

# ---------- Datos de ejemplo basados en mercado real ----------

SEED_PROPERTIES: list[dict] = [
    {
        "title": "Apartamento Vista al Mar Pampatar",
        "municipality": Municipality.MANEIRO,
        "zone": "Pampatar",
        "type": PropertyType.APARTAMENTO,
        "price_usd": 95000,
        "bedrooms": 2,
        "bathrooms": 2,
        "area_m2": 85,
        "description": "Moderno apartamento con vista directa al mar, piscina comunitaria, seguridad 24h, a 5 min de Playa El Ángel. Ideal para alquiler vacacional.",
        "features": ["vista_al_mar", "piscina", "seguridad_24h", "estacionamiento"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Casa Colonial en La Asunción",
        "municipality": Municipality.ARISMENDI,
        "zone": "La Asunción Centro",
        "type": PropertyType.CASA,
        "price_usd": 45000,
        "bedrooms": 3,
        "bathrooms": 2,
        "area_m2": 120,
        "description": "Casa colonial restaurada en el centro histórico. Cerca de la Catedral y mercado municipal. Patio interior amplio.",
        "features": ["patio", "centro_historico", "cerca_iglesia"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Penthouse de Lujo Playa El Agua",
        "municipality": Municipality.ANTOLIN_DEL_CAMPO,
        "zone": "Playa El Agua",
        "type": PropertyType.PENTHOUSE,
        "price_usd": 180000,
        "bedrooms": 3,
        "bathrooms": 3,
        "area_m2": 200,
        "description": "Espectacular penthouse con terraza privada, jacuzzi y vista panorámica. Edificio con acceso directo a la playa. Alta demanda de alquiler vacacional.",
        "features": ["terraza", "jacuzzi", "vista_panoramica", "acceso_playa", "seguridad"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Terreno Playa Caribe",
        "municipality": Municipality.GOMEZ,
        "zone": "Playa Caribe",
        "type": PropertyType.TERRENO,
        "price_usd": 25000,
        "bedrooms": None,
        "bathrooms": None,
        "area_m2": 500,
        "description": "Terreno plano a 200 metros de la playa. Zona emergente con crecimiento turístico. Ideal para construcción de posada o casa vacacional.",
        "features": ["cerca_playa", "plano", "zona_emergente"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Apartamento Ejecutivo Porlamar",
        "municipality": Municipality.MARINO,
        "zone": "Porlamar Centro",
        "type": PropertyType.APARTAMENTO,
        "price_usd": 65000,
        "bedrooms": 2,
        "bathrooms": 2,
        "area_m2": 75,
        "description": "Apartamento en zona comercial de Porlamar. Cerca de Sambil Margarita, bancos y restaurantes. Perfecto para residencia o inversión de alquiler largo plazo.",
        "features": ["cerca_centro_comercial", "transporte_publico", "urbanizacion_cerrada"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Casa Familiar Juan Griego",
        "municipality": Municipality.MARINO,
        "zone": "Juan Griego",
        "type": PropertyType.CASA,
        "price_usd": 55000,
        "bedrooms": 3,
        "bathrooms": 2,
        "area_m2": 140,
        "description": "Casa amplia en zona residencial tranquila de Juan Griego. Jardín frontal, cochera para 2 vehículos. Cerca de la bahía y restaurantes de pescado.",
        "features": ["jardin", "cochera", "zona_residencial"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Local Comercial Pampatar",
        "municipality": Municipality.MANEIRO,
        "zone": "Pampatar Pueblo",
        "type": PropertyType.LOCAL,
        "price_usd": 80000,
        "bedrooms": None,
        "bathrooms": 2,
        "area_m2": 90,
        "description": "Local comercial en pleno corazón de Pampatar. Alto flujo peatonal turístico. Apto para restaurante, boutique o oficina de turismo.",
        "features": ["alto_trafico", "cerca_playa", "area_comercial"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Apartamento Estudio Playa Parguito",
        "municipality": Municipality.ANTOLIN_DEL_CAMPO,
        "zone": "Playa Parguito",
        "type": PropertyType.APARTAMENTO,
        "price_usd": 35000,
        "bedrooms": 1,
        "bathrooms": 1,
        "area_m2": 45,
        "description": "Estudio acogedor a 100m de Playa Parguito. Ideal para surfistas y turismo de aventura. Bajo mantenimiento, alto retorno en alquiler vacacional.",
        "features": ["cerca_playa", "bajo_mantenimiento", "rentabilidad_alta"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Casa Quinta El Tirano",
        "municipality": Municipality.ANTOLIN_DEL_CAMPO,
        "zone": "El Tirano",
        "type": PropertyType.CASA,
        "price_usd": 110000,
        "bedrooms": 4,
        "bathrooms": 3,
        "area_m2": 250,
        "description": "Espaciosa casa quinta con piscina privada, BBQ y jardín tropical. Zona muy solicitada por turistas europeos. Excelente para alquiler vacacional por temporadas.",
        "features": ["piscina_privada", "bbq", "jardin_tropical", "renta_temporada"],
        "status": PropertyStatus.AVAILABLE,
    },
    {
        "title": "Terreno con Vista La Guardia",
        "municipality": Municipality.ARISMENDI,
        "zone": "La Guardia",
        "type": PropertyType.TERRENO,
        "price_usd": 18000,
        "bedrooms": None,
        "bathrooms": None,
        "area_m2": 300,
        "description": "Terreno con vista a la laguna. Zona residencial en crecimiento. Servicios básicos disponibles. Precio accesible para primer inversión.",
        "features": ["vista_laguna", "servicios", "precio_accesible"],
        "status": PropertyStatus.AVAILABLE,
    },
]


async def seed_properties() -> None:
    """Inserta propiedades y genera sus embeddings vectoriales."""
    db = Database(settings.database_path)
    await db.connect()

    repo = PropertyRepository(db)
    llm = LLMClient()

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

    await db.close()
    logger.info("seed_complete")


if __name__ == "__main__":
    asyncio.run(seed_properties())
