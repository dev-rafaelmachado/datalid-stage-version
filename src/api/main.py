"""
🚀 Datalid API - Main Application
API REST para detecção e extração de datas de validade.
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

# Adicionar src ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.api.config import get_settings
from src.api.middleware import setup_middleware
from src.api.routes import router
from src.api.routes_v2 import router_v2
from src.api.schemas import ErrorDetail, ErrorResponse
from src.api.service import get_service

# ========================================
# CONFIGURAÇÃO DE LOGGING
# ========================================

def setup_logging():
    """Configura sistema de logging."""
    settings = get_settings()
    
    # Remover handler padrão
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler (se configurado)
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            settings.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    logger.info("📝 Logging configurado")


# ========================================
# EXCEPTION HANDLERS
# ========================================

def setup_exception_handlers(app: FastAPI):
    """Configura handlers de exceção."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handler para erros de validação."""
        request_id = getattr(request.state, "request_id", None)
        
        logger.warning(f"Erro de validação: {exc.errors()}")
        
        error_response = ErrorResponse(
            status="error",
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Erro de validação dos dados de entrada",
                details={"errors": exc.errors()}
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handler para exceções gerais."""
        request_id = getattr(request.state, "request_id", None)
        
        logger.exception(f"Exceção não tratada: {exc}")
        
        error_response = ErrorResponse(
            status="error",
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="Erro interno do servidor",
                details={"exception": str(exc)} if get_settings().debug else None
            ),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode='json')
        )
    
    logger.info("🛡️ Exception handlers configurados")


# ========================================
# STARTUP & SHUTDOWN
# ========================================

async def startup_event():
    """Executado ao iniciar a aplicação."""
    settings = get_settings()
    
    logger.info("=" * 80)
    logger.info(f"🚀 {settings.api_name} v{settings.api_version}")
    logger.info("=" * 80)
    logger.info(f"📝 Ambiente: {settings.environment}")
    logger.info(f"🔧 Debug: {settings.debug}")
    logger.info(f"🌐 Host: {settings.host}:{settings.port}")
    logger.info(f"📚 Documentação: {settings.docs_url}")
    logger.info("=" * 80)
    
    # Criar diretórios necessários
    settings.get_upload_path().mkdir(parents=True, exist_ok=True)
    settings.get_results_path().mkdir(parents=True, exist_ok=True)
    logger.info("📁 Diretórios criados")
    
    # Inicializar serviço de processamento
    try:
        service = get_service()
        service.initialize()
        logger.success("✅ Serviço de processamento inicializado")
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar serviço: {e}")
        logger.warning("⚠️ API rodando mas processamento pode falhar")
    
    logger.success("✅ API pronta para receber requisições!")


async def shutdown_event():
    """Executado ao desligar a aplicação."""
    logger.info("🛑 Encerrando API...")
    
    # Cleanup se necessário
    settings = get_settings()
    if settings.cleanup_uploads:
        upload_dir = settings.get_upload_path()
        if upload_dir.exists():
            import shutil
            try:
                shutil.rmtree(upload_dir)
                logger.info("🧹 Uploads temporários limpos")
            except Exception as e:
                logger.warning(f"Erro ao limpar uploads: {e}")
    
    logger.info("👋 API encerrada")


# ========================================
# APP FACTORY
# ========================================

def create_app() -> FastAPI:
    """
    Cria e configura a aplicação FastAPI.
    
    Returns:
        Aplicação FastAPI configurada
    """
    settings = get_settings()
    
    # Configurar logging
    setup_logging()
    
    # Criar app
    app = FastAPI(
        title=settings.api_name,
        version=settings.api_version,
        description=settings.api_description,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        debug=settings.debug,
    )
    
    # Configurar middleware
    setup_middleware(app)
    
    # Configurar exception handlers
    setup_exception_handlers(app)
    
    # Registrar rotas
    app.include_router(router, tags=["API v1"])
    app.include_router(router_v2, tags=["API v2"])
    
    # Eventos de lifecycle
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)
    
    return app


# ========================================
# APP INSTANCE
# ========================================

app = create_app()


# ========================================
# CLI PARA RODAR O SERVIDOR
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    # Configuração do Uvicorn
    uvicorn_config = {
        "app": "src.api.main:app",
        "host": settings.host,
        "port": settings.port,
        "workers": settings.workers,
        "reload": settings.reload,
        "log_level": settings.log_level.lower(),
        "access_log": settings.log_requests,
    }
    
    # Rodar servidor
    logger.info(f"🚀 Iniciando servidor em {settings.host}:{settings.port}")
    uvicorn.run(**uvicorn_config)
