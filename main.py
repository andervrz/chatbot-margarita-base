"""
Entry point de la aplicación.
Ejecutar en desarrollo:  uvicorn main:app --reload
Ejecutar en producción:  python main.py
"""

import uvicorn

from src.config import settings
from src.logging_config import configure_logging

if __name__ == "__main__":
    # Configurar logging ANTES de que uvicorn arranque, para capturar errores tempranos
    configure_logging(settings.app_env, settings.log_level)

    uvicorn.run(
        "src.interface.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_dev,           # Solo recarga en desarrollo
        log_level=settings.log_level.lower(),
    )
