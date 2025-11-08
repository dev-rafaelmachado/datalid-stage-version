"""
🚀 Script para Iniciar a API
"""

import sys
from pathlib import Path

# Adicionar src ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

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
