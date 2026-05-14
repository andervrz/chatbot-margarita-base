#!/usr/bin/env python3
"""
Pre-pobla la cache exacta y semántica con las preguntas más frecuentes.
Reduce a cero las llamadas al LLM para consultas comunes desde el día 1.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.application.intent import IntentDetector

# Pares (consulta_variación, respuesta_faq, intent_type)
WARM_ENTRIES = [
    # Precios m²
    ("cuánto cuesta el metro cuadrado en pampatar", "faq_price_m2"),
    ("precio por m2 en margarita", "faq_price_m2"),
    ("valor del metro cuadrado en porlamar", "faq_price_m2"),
    ("cuánto sale el m² en playa el agua", "faq_price_m2"),
    # Extranjeros
    ("puedo comprar si soy extranjero", "faq_foreign_buy"),
    ("comprar propiedad en margarita siendo extranjero", "faq_foreign_buy"),
    ("requisitos para comprar si no soy venezolano", "faq_foreign_buy"),
    # Trámites
    ("qué documentos necesito para comprar", "faq_procedure"),
    ("cómo es el trámite de compra", "faq_procedure"),
    ("pasos para comprar una casa", "faq_procedure"),
    # Rentabilidad
    ("es rentable alquilar por airbnb", "faq_rental_roi"),
    ("cuánto se gana con alquiler vacacional", "faq_rental_roi"),
    ("retorno de inversión en margarita", "faq_rental_roi"),
]


async def warm() -> None:
    db = Database(settings.database_path)
    await db.connect()

    llm = LLMClient()
    cache = CacheManager(db, llm)
    detector = IntentDetector()

    print("Warming cache...")

    for query, intent_key in WARM_ENTRIES:
        intent = IntentType(intent_key)
        response = detector.get_faq_response(intent)
        if response:
            await cache.set(query, response, intent.value)
            print(f"  Cached: {query[:50]}...")
        else:
            print(f"  Skip (no response): {query[:50]}...")

    await db.close()
    print("Cache warmed.")


if __name__ == "__main__":
    # Importar aquí para evitar circular si se importa como módulo
    from src.application.intent import IntentType
    asyncio.run(warm())
