"""
⚙️ Configuração da API
Gerenciamento de configurações usando Pydantic Settings.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Configurações principais da API."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="API_",  # Usar prefixo API_ para evitar conflitos
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================================
    # INFORMAÇÕES DA API
    # ========================================
    api_name: str = Field(
        default="Datalid API",
        description="Nome da API"
    )
    api_version: str = Field(
        default="1.0.0",
        description="Versão da API"
    )
    api_description: str = Field(
        default="API REST para detecção e extração de datas de validade",
        description="Descrição da API"
    )
    
    # ========================================
    # SERVIDOR
    # ========================================
    host: str = Field(
        default="0.0.0.0",
        description="Host do servidor"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Porta do servidor"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Número de workers Uvicorn"
    )
    reload: bool = Field(
        default=False,
        description="Auto-reload (apenas desenvolvimento)"
    )
    
    # ========================================
    # CORS
    # ========================================
    cors_enabled: bool = Field(
        default=True,
        description="Habilitar CORS"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Origins permitidas para CORS"
    )
    cors_credentials: bool = Field(
        default=True,
        description="Permitir credentials no CORS"
    )
    cors_methods: List[str] = Field(
        default=["*"],
        description="Métodos HTTP permitidos"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        description="Headers permitidos"
    )
    
    # ========================================
    # LIMITES E SEGURANÇA
    # ========================================
    max_file_size_mb: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Tamanho máximo de arquivo (MB)"
    )
    max_batch_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Número máximo de imagens em batch"
    )
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        description="Extensões de imagem permitidas"
    )
    
    # Rate limiting (requisições por minuto)
    rate_limit_enabled: bool = Field(
        default=True,
        description="Habilitar rate limiting"
    )
    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Requisições permitidas por minuto"
    )
    
    # Timeout
    request_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout de requisição (segundos)"
    )
    
    # ========================================
    # AUTENTICAÇÃO (OPCIONAL)
    # ========================================
    auth_enabled: bool = Field(
        default=False,
        description="Habilitar autenticação"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Nome do header de autenticação"
    )
    api_keys: List[str] = Field(
        default=[],
        description="API keys válidas"
    )
    
    # ========================================
    # JWT E AUTENTICAÇÃO AVANÇADA
    # ========================================
    jwt_secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-USE-RANDOM-SECRET",
        description="Chave secreta para JWT (MUDE EM PRODUÇÃO!)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algoritmo JWT"
    )
    jwt_expire_minutes: int = Field(
        default=1440,  # 24 horas
        ge=1,
        description="Tempo de expiração do token (minutos)"
    )
    
    # ========================================
    # MODELOS
    # ========================================
    default_yolo_model: str = Field(
        default="experiments/1104/yolov8m-seg/weights/best.",
        description="Caminho do modelo YOLO padrão"
    )
    model_device: str = Field(
        default="0",
        description="Device para modelos (cuda:0, cpu, etc)"
    )
    
    # ========================================
    # OCR
    # ========================================
    default_ocr_engine: str = Field(
        default="openocr",
        description="Engine de OCR padrão"
    )
    default_ocr_config: str = Field(
        default="config/ocr/openocr.yaml",
        description="Configuração de OCR padrão"
    )
    default_preprocessing_config: str = Field(
        default="config/preprocessing/ppro-openocr-medium.yaml",
        description="Configuração de pré-processamento padrão"
    )
    
    # ========================================
    # DETECÇÃO
    # ========================================
    default_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confiança mínima padrão"
    )
    default_iou: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="IoU threshold padrão"
    )
    
    # ========================================
    # CACHE
    # ========================================
    cache_enabled: bool = Field(
        default=True,
        description="Habilitar cache de resultados"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="TTL do cache (segundos)"
    )
    cache_max_size: int = Field(
        default=100,
        ge=10,
        description="Tamanho máximo do cache"
    )
    
    # ========================================
    # STORAGE
    # ========================================
    upload_dir: str = Field(
        default="uploads",
        description="Diretório para uploads temporários"
    )
    results_dir: str = Field(
        default="outputs/api_results",
        description="Diretório para resultados salvos"
    )
    cleanup_uploads: bool = Field(
        default=True,
        description="Limpar uploads após processamento"
    )
    save_results_by_default: bool = Field(
        default=False,
        description="Salvar resultados por padrão"
    )
    
    # ========================================
    # LOGGING
    # ========================================
    log_level: str = Field(
        default="INFO",
        description="Nível de logging (DEBUG, INFO, WARNING, ERROR)"
    )
    log_requests: bool = Field(
        default=True,
        description="Logar todas as requisições"
    )
    log_file: Optional[str] = Field(
        default="logs/api.log",
        description="Arquivo de log"
    )
    
    # ========================================
    # MONITORING
    # ========================================
    enable_metrics: bool = Field(
        default=True,
        description="Habilitar métricas de monitoramento"
    )
    metrics_endpoint: str = Field(
        default="/metrics",
        description="Endpoint de métricas"
    )
    
    # ========================================
    # DOCUMENTAÇÃO
    # ========================================
    docs_url: str = Field(
        default="/docs",
        description="URL da documentação Swagger"
    )
    redoc_url: str = Field(
        default="/redoc",
        description="URL da documentação ReDoc"
    )
    openapi_url: str = Field(
        default="/openapi.json",
        description="URL do schema OpenAPI"
    )
    
    # ========================================
    # DESENVOLVIMENTO
    # ========================================
    debug: bool = Field(
        default=False,
        description="Modo debug"
    )
    environment: str = Field(
        default="production",
        description="Ambiente (development, staging, production)"
    )
    
    # ========================================
    # VALIDADORES
    # ========================================
    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Garante que extensões começam com ponto."""
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]
    
    @field_validator("upload_dir", "results_dir", "log_file")
    @classmethod
    def create_directories(cls, v: Optional[str]) -> Optional[str]:
        """Cria diretórios se não existirem."""
        if v:
            path = Path(v)
            if not path.suffix:  # É diretório
                path.mkdir(parents=True, exist_ok=True)
            else:  # É arquivo
                path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    # ========================================
    # MÉTODOS AUXILIARES
    # ========================================
    def get_upload_path(self) -> Path:
        """Retorna Path do diretório de upload."""
        return Path(self.upload_dir)
    
    def get_results_path(self) -> Path:
        """Retorna Path do diretório de resultados."""
        return Path(self.results_dir)
    
    def get_max_file_size_bytes(self) -> int:
        """Retorna tamanho máximo de arquivo em bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)
    
    def is_allowed_extension(self, filename: str) -> bool:
        """Verifica se extensão é permitida."""
        ext = Path(filename).suffix.lower()
        return ext in [e.lower() for e in self.allowed_extensions]
    
    def get_cors_config(self) -> Dict:
        """Retorna configuração de CORS."""
        if not self.cors_enabled:
            return {}
        
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_credentials,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers,
        }
    
    def to_dict(self) -> Dict:
        """Converte configurações para dicionário."""
        return self.model_dump()


# ========================================
# INSTÂNCIA GLOBAL
# ========================================

# Singleton das configurações
_settings: Optional[APISettings] = None


def get_settings() -> APISettings:
    """
    Retorna instância singleton das configurações.
    
    Returns:
        Configurações da API
    """
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings


def reload_settings() -> APISettings:
    """
    Recarrega as configurações.
    
    Returns:
        Novas configurações
    """
    global _settings
    _settings = APISettings()
    return _settings


__all__ = ["APISettings", "get_settings", "reload_settings"]
