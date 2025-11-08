"""
🛡️ Middleware da API
Componentes de middleware para logging, CORS, rate limiting, etc.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import ErrorDetail, ErrorResponse

# ========================================
# REQUEST ID MIDDLEWARE
# ========================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Adiciona ID único a cada requisição."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


# ========================================
# LOGGING MIDDLEWARE
# ========================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Loga todas as requisições."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        settings = get_settings()
        
        if not settings.log_requests:
            return await call_next(request)
        
        # Dados da requisição
        request_id = getattr(request.state, "request_id", "unknown")
        method = request.method
        url = str(request.url)
        client = request.client.host if request.client else "unknown"
        
        # Tempo de início
        start_time = time.time()
        
        # Log de entrada
        logger.info(
            f"→ [{request_id}] {method} {url} from {client}"
        )
        
        # Processar requisição
        try:
            response = await call_next(request)
            
            # Tempo de processamento
            process_time = time.time() - start_time
            
            # Log de saída
            logger.info(
                f"← [{request_id}] {method} {url} "
                f"→ {response.status_code} ({process_time:.3f}s)"
            )
            
            # Adicionar header de tempo
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"✗ [{request_id}] {method} {url} "
                f"→ ERROR ({process_time:.3f}s): {str(e)}"
            )
            raise


# ========================================
# ERROR HANDLER MIDDLEWARE
# ========================================

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Captura e formata erros."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        try:
            return await call_next(request)
            
        except Exception as e:
            request_id = getattr(request.state, "request_id", None)
            
            # Log do erro
            logger.exception(f"Erro não tratado na requisição {request_id}")
            
            # Criar response de erro
            error_response = ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message="Erro interno do servidor",
                    details={"exception": str(e)} if get_settings().debug else None
                ),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump(mode='json')
            )


# ========================================
# RATE LIMITING MIDDLEWARE
# ========================================

from collections import defaultdict
from datetime import datetime, timedelta


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting simples baseado em IP."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        """
        Args:
            app: FastAPI app
            requests_per_minute: Limite de requisições por minuto
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: defaultdict = defaultdict(list)
        self.enabled = get_settings().rate_limit_enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        if not self.enabled:
            return await call_next(request)
        
        # Ignorar health check
        if request.url.path in ["/health", "/", "/metrics"]:
            return await call_next(request)
        
        # Obter IP do cliente
        client_ip = request.client.host if request.client else "unknown"
        
        # Limpar requisições antigas
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > cutoff
        ]
        
        # Verificar limite
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit excedido para IP {client_ip}")
            
            error_response = ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message=f"Limite de {self.requests_per_minute} requisições por minuto excedido",
                    details={
                        "limit": self.requests_per_minute,
                        "window": "1 minute",
                        "retry_after": 60
                    }
                ),
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response.model_dump(mode='json'),
                headers={"Retry-After": "60"}
            )
        
        # Adicionar requisição
        self.requests[client_ip].append(now)
        
        # Processar
        response = await call_next(request)
        
        # Adicionar headers de rate limit
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_ip])
        )
        
        return response


# ========================================
# AUTHENTICATION MIDDLEWARE
# ========================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware de autenticação por API Key."""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.enabled = self.settings.auth_enabled
        self.api_keys = set(self.settings.api_keys)
        self.header_name = self.settings.api_key_header
        
        # Rotas públicas (não requerem autenticação)
        self.public_routes = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        if not self.enabled:
            return await call_next(request)
        
        # Verificar se é rota pública
        if request.url.path in self.public_routes:
            return await call_next(request)
        
        # Obter API key do header
        api_key = request.headers.get(self.header_name)
        
        # Validar
        if not api_key or api_key not in self.api_keys:
            logger.warning(
                f"Tentativa de acesso não autorizado: {request.client.host if request.client else 'unknown'}"
            )
            
            error_response = ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="UNAUTHORIZED",
                    message="API key inválida ou ausente",
                    details={
                        "header": self.header_name
                    }
                ),
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=error_response.model_dump(mode='json')
            )
        
        # Autorizado
        return await call_next(request)


# ========================================
# TIMEOUT MIDDLEWARE
# ========================================

import asyncio


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware para timeout de requisições."""
    
    def __init__(self, app, timeout: int = 300):
        """
        Args:
            app: FastAPI app
            timeout: Timeout em segundos
        """
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Processa a requisição."""
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout na requisição {request.url}")
            
            error_response = ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="REQUEST_TIMEOUT",
                    message=f"Requisição excedeu o timeout de {self.timeout}s",
                    details={"timeout": self.timeout}
                ),
                request_id=getattr(request.state, "request_id", None)
            )
            
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=error_response.model_dump(mode='json')
            )


# ========================================
# HELPER FUNCTIONS
# ========================================

def setup_cors(app) -> None:
    """Configura CORS no app."""
    settings = get_settings()
    
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            **settings.get_cors_config()
        )
        logger.info("✅ CORS configurado")


def setup_middleware(app) -> None:
    """Configura todos os middlewares."""
    settings = get_settings()
    
    # Ordem importa! Do mais externo para o mais interno
    
    # 1. CORS (mais externo)
    setup_cors(app)
    
    # 2. Error Handler
    app.add_middleware(ErrorHandlerMiddleware)
    
    # 3. Timeout
    app.add_middleware(TimeoutMiddleware, timeout=settings.request_timeout)
    
    # 4. Rate Limiting
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_requests
        )
    
    # 5. Authentication
    if settings.auth_enabled:
        app.add_middleware(AuthenticationMiddleware)
    
    # 6. Logging
    if settings.log_requests:
        app.add_middleware(LoggingMiddleware)
    
    # 7. Request ID (mais interno)
    app.add_middleware(RequestIDMiddleware)
    
    logger.info("✅ Middlewares configurados")


__all__ = [
    "RequestIDMiddleware",
    "LoggingMiddleware",
    "ErrorHandlerMiddleware",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "TimeoutMiddleware",
    "setup_cors",
    "setup_middleware",
]
