"""
🚀 Script para iniciar o servidor da API
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
root_dir = Path(__file__).parent.parent.parent  # Go up to project root
sys.path.insert(0, str(root_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Iniciar Datalid API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modo desenvolvimento
  python scripts/api/start_server.py --dev
  
  # Modo produção
  python scripts/api/start_server.py --host 0.0.0.0 --port 8000 --workers 4
  
  # Com GPU
  python scripts/api/start_server.py --device cuda:0
  
  # Debug
  python scripts/api/start_server.py --debug --reload
        """
    )
    
    # Argumentos do servidor
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host (padrão: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Porta (padrão: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers (padrão: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload em mudanças de código"
    )
    
    # Configurações de modelo
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device para modelos (cuda:0, cpu, etc)"
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default=None,
        help="Engine de OCR padrão"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confiança mínima padrão"
    )
    
    # Modo
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Modo desenvolvimento (debug + reload)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug"
    )
    
    # Autenticação
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Habilitar autenticação"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        action="append",
        help="Adicionar API key"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Nível de logging"
    )
    
    args = parser.parse_args()
    
    # Configurar environment
    import os
    
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.workers:
        os.environ["WORKERS"] = str(args.workers)
    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    if args.ocr_engine:
        os.environ["DEFAULT_OCR_ENGINE"] = args.ocr_engine
    if args.confidence:
        os.environ["DEFAULT_CONFIDENCE"] = str(args.confidence)
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    
    # Modo desenvolvimento
    if args.dev:
        args.debug = True
        args.reload = True
        os.environ["ENVIRONMENT"] = "development"
    
    if args.debug:
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    if args.reload:
        os.environ["RELOAD"] = "true"
    
    # Autenticação
    if args.auth:
        os.environ["AUTH_ENABLED"] = "true"
        if args.api_key:
            import json
            os.environ["API_KEYS"] = json.dumps(args.api_key)
    
    # Importar após configurar environment
    from src.api.config import get_settings
    
    settings = get_settings()
    
    # Banner
    print("=" * 80)
    print(f"🚀 {settings.api_name} v{settings.api_version}")
    print("=" * 80)
    print(f"📍 URL: http://{settings.host}:{settings.port}")
    print(f"📚 Docs: http://{settings.host}:{settings.port}{settings.docs_url}")
    print(f"🔧 Environment: {settings.environment}")
    print(f"🐛 Debug: {settings.debug}")
    print(f"🔄 Reload: {settings.reload}")
    print(f"👥 Workers: {settings.workers}")
    print(f"🤖 Device: {settings.model_device}")
    print(f"🔍 OCR: {settings.default_ocr_engine}")
    print(f"🔐 Auth: {settings.auth_enabled}")
    print("=" * 80)
    
    # Iniciar servidor
    import uvicorn
    
    uvicorn_config = {
        "app": "src.api.main:app",
        "host": settings.host,
        "port": settings.port,
        "workers": settings.workers if not settings.reload else 1,
        "reload": settings.reload,
        "log_level": settings.log_level.lower(),
        "access_log": settings.log_requests,
    }
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n👋 Servidor encerrado")


if __name__ == "__main__":
    main()
