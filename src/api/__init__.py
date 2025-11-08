"""
🚀 API REST para Detecção de Datas de Validade
Sistema modular e escalável usando FastAPI.
"""

__version__ = "1.0.0"
__all__ = ["DatalidClient", "get_settings", "APISettings"]

# Imports principais
try:
    from .config import APISettings, get_settings
except ImportError:
    pass

# Cliente Python (para uso externo)
try:
    import sys
    from pathlib import Path
    scripts_path = Path(__file__).parent.parent.parent / "scripts" / "api"
    sys.path.insert(0, str(scripts_path))
    from scripts.api.client import DatalidClient
except ImportError:
    pass
