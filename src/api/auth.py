"""
🔐 Autenticação e Autorização
Sistema de API Keys e JWT tokens.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (APIKeyHeader, HTTPAuthorizationCredentials,
                              HTTPBearer)
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import get_settings

# ========================================
# CONFIGURAÇÃO
# ========================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# ========================================
# SCHEMAS
# ========================================

class TokenData(BaseModel):
    """Dados do token JWT."""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """Modelo de usuário."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = []


# ========================================
# FUNÇÕES DE AUTENTICAÇÃO
# ========================================

def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verifica se a API key é válida.
    
    Args:
        api_key: API key do header
        
    Returns:
        True se válida
        
    Raises:
        HTTPException: Se inválida ou não fornecida
    """
    settings = get_settings()
    
    # Se autenticação não habilitada, permitir acesso
    if not settings.auth_enabled:
        return True
    
    # Verificar se API key foi fornecida
    if not api_key:
        logger.warning("API key não fornecida")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key não fornecida",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Verificar se API key é válida
    if api_key not in settings.api_keys:
        logger.warning(f"API key inválida: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    logger.debug(f"API key válida: {api_key[:8]}...")
    return True


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Cria um token JWT.
    
    Args:
        data: Dados para incluir no token
        expires_delta: Tempo de expiração
        
    Returns:
        Token JWT
    """
    settings = get_settings()
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> TokenData:
    """
    Verifica e decodifica um token JWT.
    
    Args:
        credentials: Credenciais do header Authorization
        
    Returns:
        Dados do token
        
    Raises:
        HTTPException: Se token inválido ou expirado
    """
    settings = get_settings()
    
    # Se autenticação não habilitada, permitir acesso
    if not settings.auth_enabled:
        return TokenData()
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token não fornecido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = TokenData(username=username, scopes=scopes)
        return token_data
        
    except JWTError as e:
        logger.warning(f"Erro ao verificar token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token_data: TokenData = Depends(verify_token)) -> User:
    """
    Obtém o usuário atual a partir do token.
    
    Args:
        token_data: Dados do token
        
    Returns:
        Usuário atual
    """
    # TODO: Buscar usuário do banco de dados
    # Por enquanto, retornar um usuário mock
    user = User(
        username=token_data.username or "anonymous",
        scopes=token_data.scopes
    )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuário desativado"
        )
    
    return user


def require_scope(required_scope: str):
    """
    Decorator para exigir um escopo específico.
    
    Args:
        required_scope: Escopo requerido
        
    Returns:
        Dependency function
    """
    def check_scope(user: User = Depends(get_current_user)) -> User:
        if required_scope not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permissão insuficiente. Requer escopo: {required_scope}"
            )
        return user
    
    return check_scope


# ========================================
# HELPERS
# ========================================

def hash_password(password: str) -> str:
    """Hash de senha."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica senha."""
    return pwd_context.verify(plain_password, hashed_password)


__all__ = [
    "verify_api_key",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "require_scope",
    "hash_password",
    "verify_password",
    "User",
    "TokenData",
]
