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

# ... WARM_ENTRIES sin cambios ...

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
