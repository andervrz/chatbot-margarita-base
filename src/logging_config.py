"""
Configuración centralizada de logging estructurado.
Usa structlog para logs JSON en producción y coloreados en desarrollo.
"""

import logging
import sys

import structlog


def configure_logging(env: str = "development", log_level: str = "INFO") -> None:
    """Configura structlog y logging estándar de Python."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if env == "development":
        # Consola legible con colores
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Producción: JSON para ingestión
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configurar también el logging estándar de Python (uvicorn, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
