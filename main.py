"""
Entry point de la aplicación.
Ejecutar: uvicorn main:app --reload
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.interface.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
