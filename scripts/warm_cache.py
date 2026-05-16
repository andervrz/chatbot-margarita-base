#!/usr/bin/env python3
"""
Pre-pobla la cache exacta y semántica con las preguntas más frecuentes.
Reduce a cero las llamadas al LLM para consultas comunes desde el día 1.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.intent import IntentDetector, IntentType
from src.config import settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient

# ============================================
# PREGUNTAS FRECUENTES PARA PRE-CACHE
# ============================================
WARM_ENTRIES: list[tuple[str, str]] = [
    # Saludos
    ("hola", "greeting"),
    ("buenos días", "greeting"),
    ("buenas tardes", "greeting"),
    
    # Precios por m²
    ("precio por metro cuadrado", "faq_price_m2"),
    ("cuánto cuesta el m2", "faq_price_m2"),
    ("valor del metro cuadrado en pampatar", "faq_price_m2"),
    ("precio por m² en playa el agua", "faq_price_m2"),
    ("cuánto cuesta el metro cuadrado en porlamar", "faq_price_m2"),
    
    # Extranjeros
    ("extranjero puede comprar", "faq_foreign_buy"),
    ("no soy venezolano puedo comprar", "faq_foreign_buy"),
    ("cómo compra un extranjero en margarita", "faq_foreign_buy"),
    ("visa de inversionista", "faq_foreign_buy"),
    
    # Rentabilidad
    ("rentabilidad alquiler vacacional", "faq_rental_roi"),
    ("roi de inversión en margarita", "faq_rental_roi"),
    ("cuánto se gana con alquiler vacacional", "faq_rental_roi"),
    
    # Trámites
    ("documentos necesarios para comprar", "faq_procedure"),
    ("trámites de compra", "faq_procedure"),
    ("pasos para comprar una casa", "faq_procedure"),
    ("escritura y registro", "faq_procedure"),
    
    # Despedidas
    ("gracias", "goodbye"),
    ("adiós", "goodbye"),
    ("hasta luego", "goodbye"),
]


async def warm() -> None:
    llm = LLMClient()
    db = Database(settings.database_path, embedding_dim=llm.embedding_dim)
    await db.connect()

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
    asyncio.run(warm())
