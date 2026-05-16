#!/usr/bin/env python3
"""
CLI interactivo para probar el Margarita Realty Bot.
Conecta directamente con ChatService, sin pasar por HTTP.
Útil para desarrollo, debugging y demos locales.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ajustar path para imports desde scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from src.application.chat import ChatService
from src.application.enrichment import PromptEnricher
from src.application.intent import IntentDetector
from src.config import settings
from src.infrastructure.cache import CacheManager
from src.infrastructure.conversation_store import ConversationStore
from src.infrastructure.db import Database
from src.infrastructure.llm import LLMClient
from src.infrastructure.repositories import PropertyRepository
from src.logging_config import configure_logging

configure_logging(env="development", log_level="WARNING")
logger = structlog.get_logger()

console = Console()

# ... Colors, print_banner, print_help, show_history sin cambios ...

async def init_services(db_path: Path) -> ChatService:
    """Inicializa dependencias reales (no mockeadas)."""
    llm = LLMClient()
    db = Database(db_path, embedding_dim=llm.embedding_dim)
    await db.connect()

    property_repo = PropertyRepository(db)
    cache = CacheManager(db, llm)
    conv_store = ConversationStore(db)
    enricher = PromptEnricher()
    intent_detector = IntentDetector()

    return ChatService(
        llm=llm,
        property_repo=property_repo,
        cache=cache,
        conv_store=conv_store,
        enricher=enricher,
        intent_detector=intent_detector,
    )

# ... main() sin cambios estructurales, solo init_services envuelto en try/except ...

async def main() -> None:
    parser = argparse.ArgumentParser(description="CLI del Margarita Realty Bot")
    # ... args sin cambios ...

    # Inicializar servicios
    try:
        with console.status("[bold green]Conectando a base de datos y LLM..."):
            service = await init_services(Path(args.db))
    except FileNotFoundError as exc:
        console.print(f"[bright_red]❌ Error de configuración: {exc}[/bright_red]")
        console.print("[dim]Verifica que existe la carpeta 'prompts/' con los templates.[/dim]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bright_red]❌ Error al inicializar: {exc}[/bright_red]")
        sys.exit(1)

    # ... resto sin cambios ...
