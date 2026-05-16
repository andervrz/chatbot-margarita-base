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

class Colors:
    """Colores consistentes para la UI de terminal."""
    BOT = "bright_cyan"
    USER = "bright_green"
    SYSTEM = "dim"
    ERROR = "bright_red"
    METRIC = "bright_yellow"


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

def print_banner() -> None:
    """Banner de bienvenida."""
    banner = """
    ┌─────────────────────────────────────────┐
    │   🏝️  MARGARITA REALTY BOT  🤖         │
    │   Asesor Inmobiliario de Margarita      │
    │   CLI de desarrollo v0.1.0              │
    └─────────────────────────────────────────┘
    """
    console.print(Panel(banner, style=Colors.BOT, border_style=Colors.BOT))


def print_help() -> None:
    """Comandos especiales disponibles."""
    help_text = """
**Comandos especiales:**
• `/help`      — Mostrar esta ayuda
• `/history`   — Ver historial de la sesión actual
• `/new`       — Iniciar nueva sesión (olvidar contexto)
• `/session`   — Mostrar ID de sesión actual
• `/raw`       — Mostrar respuesta cruda sin formatear
• `/quit`      — Salir
    """
    console.print(Markdown(help_text))


async def show_history(service: ChatService, session_id: str, conv_store: ConversationStore) -> None:
    """Muestra el historial persistente de la sesión."""
    history = await conv_store.get_history(session_id, limit=50)

    if not history:
        console.print("[dim]No hay mensajes en esta sesión todavía.[/dim]")
        return

    console.print(Rule(f"Historial de sesión: {session_id}", style=Colors.SYSTEM))
    for msg in history:
        role_color = Colors.USER if msg.role.value == "user" else Colors.BOT
        icon = "👤" if msg.role.value == "user" else "🤖"
        console.print(f"[{role_color}]{icon} {msg.role.value}:[/{role_color}] {msg.content[:200]}")
    console.print(Rule(style=Colors.SYSTEM))


async def main() -> None:
    parser = argparse.ArgumentParser(description="CLI del Margarita Realty Bot")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="ID de sesión para retomar conversación",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(settings.database_path),
        help="Path a la base de datos SQLite",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Omitir banner de bienvenida",
    )
    args = parser.parse_args()

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
        
    # Session ID
    session_id = args.session or f"cli-{int(time.time())}"
    console.print(f"[{Colors.SYSTEM}]✓ Sesión:[/{Colors.SYSTEM}] {session_id}\n")

    if not args.no_banner:
        print_banner()

    console.print(
        f"[{Colors.SYSTEM}]Escribe tu mensaje o /help para ver comandos. Ctrl+C para salir.\n[/{Colors.SYSTEM}]"
    )

    show_raw = False

    try:
        while True:
            # Input del usuario
            user_input = Prompt.ask(f"[{Colors.USER}]👤 Tú[/{Colors.USER}]")

            # Comandos especiales
            if user_input.startswith("/"):
                cmd = user_input.strip().lower()

                if cmd == "/quit":
                    break
                elif cmd == "/help":
                    print_help()
                    continue
                elif cmd == "/history":
                    await show_history(service, session_id, service.conv_store)
                    continue
                elif cmd == "/new":
                    session_id = f"cli-{int(time.time())}"
                    console.print(f"\n[{Colors.SYSTEM}]🆕 Nueva sesión: {session_id}[/{Colors.SYSTEM}]\n")
                    continue
                elif cmd == "/session":
                    console.print(f"[{Colors.SYSTEM}]Sesión actual: {session_id}[/{Colors.SYSTEM}]")
                    continue
                elif cmd == "/raw":
                    show_raw = not show_raw
                    state = "activado" if show_raw else "desactivado"
                    console.print(f"[{Colors.SYSTEM}]Modo raw {state}[/{Colors.SYSTEM}]")
                    continue
                else:
                    console.print(f"[{Colors.ERROR}]Comando desconocido: {cmd}[/{Colors.ERROR}]")
                    continue

            # Procesar mensaje
            start = time.perf_counter()
            try:
                response = await service.handle(
                    session_id=session_id,
                    user_message=user_input,
                )
                elapsed = (time.perf_counter() - start) * 1000

                # Mostrar respuesta
                if show_raw:
                    console.print(f"\n[{Colors.BOT}]🤖 Bot (raw):[/{Colors.BOT}]")
                    console.print(response)
                else:
                    console.print(f"\n[{Colors.BOT}]🤖 Bot:[/{Colors.BOT}]")
                    console.print(Markdown(response))

                # Métricas discretas
                console.print(
                    f"[{Colors.METRIC}]⏱ {elapsed:.0f}ms | sesión: {session_id[:20]}...[/{Colors.METRIC}]\n"
                )

            except Exception as exc:
                console.print(f"[{Colors.ERROR}]❌ Error: {exc}[/{Colors.ERROR}]")

    except KeyboardInterrupt:
        console.print(f"\n[{Colors.SYSTEM}]👋 Saliendo...[/{Colors.SYSTEM}]")

    finally:
        # Cleanup
        await service.cache.db.close()
        console.print(f"[{Colors.SYSTEM}]✓ Desconectado.[/{Colors.SYSTEM}]")


if __name__ == "__main__":
    asyncio.run(main())
