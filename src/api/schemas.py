"""
📋 Modelos de Dados da API
Schemas Pydantic para validação e serialização.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator


# Configuração global para todos os modelos
class BaseAPIModel(BaseModel):
    """Modelo base com configuração de serialização JSON."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True
    )

# ========================================
# ENUMS
# ========================================

class ProcessingStatus(str, Enum):
    """Status do processamento."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PROCESSING = "processing"


class OCREngine(str, Enum):
    """Engines de OCR disponíveis."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    OPENOCR = "openocr"
    PARSEQ = "parseq"
    PARSEQ_ENHANCED = "parseq_enhanced"
    TROCR = "trocr"


class ImageFormat(str, Enum):
    """Formatos de imagem suportados."""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"


# ========================================
# REQUEST MODELS
# ========================================

class DetectionConfig(BaseModel):
    """Configuração de detecção YOLO."""
    model_config = ConfigDict(extra='forbid')
    
    confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para detecções (0.0-1.0)"
    )
    iou: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold de IoU para NMS (0.0-1.0)"
    )
    max_detections: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Número máximo de detecções"
    )
    use_gpu: bool = Field(
        default=True,
        description="Usar GPU se disponível"
    )


class OCRConfig(BaseModel):
    """Configuração de OCR."""
    model_config = ConfigDict(extra='forbid')
    
    engine: OCREngine = Field(
        default=OCREngine.OPENOCR,
        description="Engine de OCR a utilizar"
    )
    languages: List[str] = Field(
        default=["por", "eng"],
        description="Idiomas para reconhecimento"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para texto OCR"
    )
    preprocessing: bool = Field(
        default=True,
        description="Aplicar pré-processamento na imagem"
    )


class ProcessingOptions(BaseModel):
    """Opções de processamento."""
    model_config = ConfigDict(extra='forbid')
    
    return_visualization: bool = Field(
        default=False,
        description="Retornar imagem com visualizações"
    )
    return_crops: bool = Field(
        default=False,
        description="Retornar crops das detecções"
    )
    return_full_ocr: bool = Field(
        default=False,
        description="Retornar todos os resultados OCR"
    )
    save_results: bool = Field(
        default=False,
        description="Salvar resultados no servidor"
    )


class ProcessImageRequest(BaseModel):
    """Request para processar uma única imagem."""
    model_config = ConfigDict(extra='forbid')
    
    detection: Optional[DetectionConfig] = Field(
        default=None,
        description="Configuração de detecção (usa padrão se não especificado)"
    )
    ocr: Optional[OCRConfig] = Field(
        default=None,
        description="Configuração de OCR (usa padrão se não especificado)"
    )
    options: Optional[ProcessingOptions] = Field(
        default=None,
        description="Opções de processamento"
    )


# ========================================
# RESPONSE MODELS
# ========================================

class BoundingBox(BaseModel):
    """Bounding box de detecção."""
    x1: float = Field(description="Coordenada x1")
    y1: float = Field(description="Coordenada y1")
    x2: float = Field(description="Coordenada x2")
    y2: float = Field(description="Coordenada y2")
    width: float = Field(description="Largura do box")
    height: float = Field(description="Altura do box")


class DetectionResult(BaseModel):
    """Resultado de uma detecção."""
    bbox: BoundingBox = Field(description="Bounding box")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança da detecção"
    )
    class_id: int = Field(description="ID da classe")
    class_name: str = Field(description="Nome da classe")
    has_mask: bool = Field(
        default=False,
        description="Tem máscara de segmentação"
    )


class OCRResult(BaseModel):
    """Resultado do OCR."""
    text: str = Field(description="Texto extraído")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança do OCR"
    )
    engine: str = Field(description="Engine utilizada")
    processing_time: float = Field(
        ge=0.0,
        description="Tempo de processamento (segundos)"
    )


class ParsedDate(BaseModel):
    """Data parseada e validada."""
    date: Optional[str] = Field(
        default=None,
        description="Data no formato ISO (YYYY-MM-DD)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança do parsing"
    )
    format: Optional[str] = Field(
        default=None,
        description="Formato original detectado"
    )
    is_valid: bool = Field(description="Data é válida")
    is_expired: Optional[bool] = Field(
        default=None,
        description="Data está expirada (se aplicável)"
    )
    days_until_expiry: Optional[int] = Field(
        default=None,
        description="Dias até expiração (negativo se expirado)"
    )


class ProcessingMetrics(BaseModel):
    """Métricas de processamento."""
    total_time: float = Field(ge=0.0, description="Tempo total (segundos)")
    detection_time: float = Field(ge=0.0, description="Tempo de detecção")
    ocr_time: float = Field(ge=0.0, description="Tempo de OCR")
    parsing_time: float = Field(ge=0.0, description="Tempo de parsing")
    num_detections: int = Field(ge=0, description="Número de detecções")
    num_dates_found: int = Field(ge=0, description="Número de datas encontradas")


class ProcessImageResponse(BaseAPIModel):
    """Response do processamento de imagem."""
    status: ProcessingStatus = Field(description="Status do processamento")
    message: str = Field(description="Mensagem descritiva")
    
    # Resultados principais
    detections: List[DetectionResult] = Field(
        default=[],
        description="Detecções encontradas"
    )
    ocr_results: List[OCRResult] = Field(
        default=[],
        description="Resultados de OCR"
    )
    dates: List[ParsedDate] = Field(
        default=[],
        description="Datas extraídas"
    )
    best_date: Optional[ParsedDate] = Field(
        default=None,
        description="Melhor data encontrada (maior confiança)"
    )
    
    # Métricas
    metrics: ProcessingMetrics = Field(description="Métricas de performance")
    
    # Opcionais
    visualization_base64: Optional[str] = Field(
        default=None,
        description="Imagem com visualizações (base64)"
    )
    crops_base64: Optional[List[str]] = Field(
        default=None,
        description="Crops das detecções (base64)"
    )
    
    # Metadata
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp do processamento"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="ID da requisição"
    )


class BatchProcessRequest(BaseModel):
    """Request para processar múltiplas imagens."""
    model_config = ConfigDict(extra='forbid')
    
    detection: Optional[DetectionConfig] = None
    ocr: Optional[OCRConfig] = None
    options: Optional[ProcessingOptions] = None


class BatchImageResult(BaseModel):
    """Resultado de uma imagem no batch."""
    filename: str = Field(description="Nome do arquivo")
    success: bool = Field(description="Processamento bem-sucedido")
    result: Optional[ProcessImageResponse] = Field(
        default=None,
        description="Resultado do processamento"
    )
    error: Optional[str] = Field(
        default=None,
        description="Mensagem de erro (se houver)"
    )


class BatchProcessResponse(BaseAPIModel):
    """Response do processamento em batch."""
    status: ProcessingStatus = Field(description="Status geral do batch")
    total_images: int = Field(ge=0, description="Total de imagens")
    successful: int = Field(ge=0, description="Imagens processadas com sucesso")
    failed: int = Field(ge=0, description="Imagens com falha")
    
    results: List[BatchImageResult] = Field(description="Resultados individuais")
    
    total_time: float = Field(ge=0.0, description="Tempo total (segundos)")
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp do processamento"
    )


# ========================================
# ERROR MODELS
# ========================================

class ErrorDetail(BaseModel):
    """Detalhes de erro."""
    code: str = Field(description="Código do erro")
    message: str = Field(description="Mensagem do erro")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detalhes adicionais"
    )


class ErrorResponse(BaseAPIModel):
    """Response de erro."""
    status: str = Field(default="error", description="Status (sempre 'error')")
    error: ErrorDetail = Field(description="Detalhes do erro")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp do erro"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="ID da requisição"
    )


# ========================================
# HEALTH & INFO MODELS
# ========================================

class HealthStatus(str, Enum):
    """Status de saúde do serviço."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Saúde de um componente."""
    status: HealthStatus = Field(description="Status do componente")
    message: Optional[str] = Field(default=None, description="Mensagem")
    latency_ms: Optional[float] = Field(default=None, description="Latência (ms)")


class HealthResponse(BaseAPIModel):
    """Response do health check."""
    status: HealthStatus = Field(description="Status geral")
    version: str = Field(description="Versão da API")
    uptime_seconds: float = Field(ge=0.0, description="Tempo online (segundos)")
    
    components: Dict[str, ComponentHealth] = Field(
        description="Status dos componentes"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp"
    )


class ModelInfo(BaseModel):
    """Informações sobre um modelo."""
    name: str = Field(description="Nome do modelo")
    type: str = Field(description="Tipo (detection/segmentation)")
    path: str = Field(description="Caminho do modelo")
    loaded: bool = Field(description="Modelo está carregado")
    size_mb: Optional[float] = Field(default=None, description="Tamanho (MB)")


class APIInfo(BaseModel):
    """Informações da API."""
    name: str = Field(description="Nome da API")
    version: str = Field(description="Versão")
    description: str = Field(description="Descrição")
    
    models: Dict[str, ModelInfo] = Field(description="Modelos disponíveis")
    ocr_engines: List[str] = Field(description="Engines de OCR disponíveis")
    
    limits: Dict[str, Any] = Field(description="Limites da API")
    
    docs_url: str = Field(description="URL da documentação")
    openapi_url: str = Field(description="URL do OpenAPI schema")


# ========================================
# VALIDATORS
# ========================================

# Validação customizada pode ser adicionada aqui
def validate_image_size(file_size_mb: float, max_size_mb: float = 10.0) -> bool:
    """Valida tamanho de imagem."""
    return file_size_mb <= max_size_mb


def validate_batch_size(num_images: int, max_batch: int = 50) -> bool:
    """Valida tamanho de batch."""
    return num_images <= max_batch


__all__ = [
    # Enums
    "ProcessingStatus",
    "OCREngine",
    "ImageFormat",
    "HealthStatus",
    
    # Requests
    "DetectionConfig",
    "OCRConfig",
    "ProcessingOptions",
    "ProcessImageRequest",
    "BatchProcessRequest",
    
    # Responses
    "BoundingBox",
    "DetectionResult",
    "OCRResult",
    "ParsedDate",
    "ProcessingMetrics",
    "ProcessImageResponse",
    "BatchImageResult",
    "BatchProcessResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ComponentHealth",
    "ModelInfo",
    "APIInfo",
    
    # Validators
    "validate_image_size",
    "validate_batch_size",
]
