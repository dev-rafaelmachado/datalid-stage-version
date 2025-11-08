"""
🚀 Script para Iniciar a API
"""

import sys
from pathlib import Path

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    import uvicorn

    from src.api.config import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
