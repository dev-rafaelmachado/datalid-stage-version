"""
🛣️ Rotas da API
Endpoints REST para detecção de datas de validade.
"""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (APIRouter, File, Form, HTTPException, Request, UploadFile,
                     status)
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from .config import get_settings
from .schemas import (BatchImageResult, BatchProcessRequest,
                      BatchProcessResponse, DetectionConfig, ErrorDetail,
                      ErrorResponse, OCRConfig, ProcessImageRequest,
                      ProcessImageResponse, ProcessingOptions,
                      ProcessingStatus)
from .service import get_service
from .utils import decode_image, validate_file_size, validate_image_format

# ========================================
# ROUTER
# ========================================

router = APIRouter()


# ========================================
# HEALTH & INFO
# ========================================

@router.get("/")
async def root():
    """Root endpoint."""
    settings = get_settings()
    
    return {
        "name": settings.api_name,
        "version": settings.api_version,
        "description": settings.api_description,
        "status": "online",
        "docs": settings.docs_url,
        "health": "/health",
    }


@router.get("/health")
async def health_check():
    """
    Health check do serviço.
    
    Verifica se todos os componentes estão funcionando.
    """
    from .schemas import ComponentHealth, HealthResponse, HealthStatus
    
    settings = get_settings()
    service = get_service()
    
    # Verificar componentes
    components = {}
    
    # 1. Serviço de processamento
    try:
        service_status = (
            HealthStatus.HEALTHY if service.is_ready()
            else HealthStatus.UNHEALTHY
        )
        components["processing_service"] = ComponentHealth(
            status=service_status,
            message="Pronto" if service.is_ready() else "Não inicializado"
        )
    except Exception as e:
        components["processing_service"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
    
    # 2. Pipeline
    try:
        pipeline_status = (
            HealthStatus.HEALTHY if service.pipeline is not None
            else HealthStatus.UNHEALTHY
        )
        components["pipeline"] = ComponentHealth(
            status=pipeline_status,
            message="Carregado" if service.pipeline else "Não carregado"
        )
    except Exception as e:
        components["pipeline"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
    
    # Status geral
    all_healthy = all(
        comp.status == HealthStatus.HEALTHY
        for comp in components.values()
    )
    
    overall_status = (
        HealthStatus.HEALTHY if all_healthy
        else HealthStatus.DEGRADED
    )
    
    response = HealthResponse(
        status=overall_status,
        version=settings.api_version,
        uptime_seconds=0.0,  # TODO: implementar tracking de uptime
        components=components
    )
    
    return response


@router.get("/info")
async def get_info():
    """
    Informações detalhadas da API.
    
    Retorna informações sobre modelos, engines, limites, etc.
    """
    from .schemas import APIInfo, ModelInfo
    
    settings = get_settings()
    service = get_service()
    
    # Informações sobre modelos
    models = {}
    
    if service.is_ready() and service.pipeline:
        # Modelo YOLO
        yolo_model_path = settings.default_yolo_model
        models["yolo"] = ModelInfo(
            name="YOLO",
            type="segmentation",
            path=yolo_model_path,
            loaded=True,
            size_mb=None
        )
    
    # Engines de OCR disponíveis
    ocr_engines = [
        "tesseract",
        "easyocr",
        "paddleocr",
        "openocr",
        "parseq",
        "parseq_enhanced",
        "trocr"
    ]
    
    # Limites
    limits = {
        "max_file_size_mb": settings.max_file_size_mb,
        "max_batch_size": settings.max_batch_size,
        "allowed_extensions": settings.allowed_extensions,
        "rate_limit_requests_per_minute": settings.rate_limit_requests,
        "request_timeout_seconds": settings.request_timeout,
    }
    
    return APIInfo(
        name=settings.api_name,
        version=settings.api_version,
        description=settings.api_description,
        models=models,
        ocr_engines=ocr_engines,
        limits=limits,
        docs_url=settings.docs_url,
        openapi_url=settings.openapi_url
    )


# ========================================
# PROCESSAMENTO DE IMAGEM
# ========================================

@router.post(
    "/process",
    response_model=ProcessImageResponse,
    status_code=status.HTTP_200_OK,
    summary="Processar uma imagem",
    description="Detecta e extrai datas de validade de uma imagem",
    responses={
        200: {"description": "Processamento bem-sucedido"},
        400: {"description": "Requisição inválida"},
        413: {"description": "Arquivo muito grande"},
        500: {"description": "Erro interno"},
    }
)
async def process_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Imagem para processar (JPEG, PNG, BMP)"
    ),
    detection_confidence: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para detecções"
    ),
    detection_iou: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Threshold de IoU para NMS"
    ),
    ocr_engine: Optional[str] = Form(
        None,
        description="Engine de OCR (tesseract, openocr, etc)"
    ),
    return_visualization: bool = Form(
        False,
        description="Retornar imagem com visualizações"
    ),
    return_crops: bool = Form(
        False,
        description="Retornar crops das detecções (DEPRECATED)"
    ),
    return_crop_images: bool = Form(
        False,
        description="Retornar imagens dos crops (original e pré-processado) nos resultados OCR"
    ),
    return_full_ocr: bool = Form(
        False,
        description="Retornar todos os resultados OCR"
    ),
    save_results: bool = Form(
        False,
        description="Salvar resultados no servidor"
    ),
) -> ProcessImageResponse:
    """
    Processa uma imagem para detectar e extrair datas de validade.
    
    **Fluxo:**
    1. Detecção de regiões com YOLO
    2. OCR para extrair texto
    3. Parsing e validação de datas
    
    **Retorna:**
    - Detecções (bounding boxes)
    - Resultados de OCR
    - Datas encontradas e validadas
    - Melhor data (maior confiança)
    - Métricas de processamento
    """
    settings = get_settings()
    service = get_service()
    request_id = getattr(request.state, "request_id", None)
    
    # Validar formato
    if not validate_image_format(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato não suportado. Use: {settings.allowed_extensions}"
        )
    
    # Validar tamanho
    content = await file.read()
    if not validate_file_size(len(content), settings.get_max_file_size_bytes()):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande. Máximo: {settings.max_file_size_mb}MB"
        )
    
    # Decodificar imagem
    try:
        image = decode_image(content)
    except Exception as e:
        logger.error(f"Erro ao decodificar imagem: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao decodificar imagem: {str(e)}"
        )
    
    # Construir configurações
    detection_config = None
    if detection_confidence is not None or detection_iou is not None:
        detection_config = DetectionConfig(
            confidence=detection_confidence or settings.default_confidence,
            iou=detection_iou or settings.default_iou
        )
    
    ocr_config = None
    if ocr_engine is not None:
        try:
            from .schemas import OCREngine
            ocr_config = OCRConfig(engine=OCREngine(ocr_engine))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Engine de OCR inválida: {ocr_engine}"
            )
    
    options = ProcessingOptions(
        return_visualization=return_visualization,
        return_crops=return_crops,
        return_crop_images=return_crop_images,
        return_full_ocr=return_full_ocr,
        save_results=save_results
    )
    
    # Processar
    try:
        result = service.process_image(
            image=image,
            image_name=file.filename,
            detection_config=detection_config,
            ocr_config=ocr_config,
            options=options,
            request_id=request_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento: {str(e)}"
        )


@router.post(
    "/process/batch",
    response_model=BatchProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Processar múltiplas imagens",
    description="Processa um lote de imagens",
)
async def process_batch(
    request: Request,
    files: List[UploadFile] = File(
        ...,
        description="Lista de imagens para processar"
    ),
    detection_confidence: Optional[float] = Form(None),
    detection_iou: Optional[float] = Form(None),
    ocr_engine: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
    return_crops: bool = Form(False),
    return_crop_images: bool = Form(False),
    return_full_ocr: bool = Form(False),
    save_results: bool = Form(False),
) -> BatchProcessResponse:
    """
    Processa múltiplas imagens em lote.
    
    **Limitações:**
    - Máximo de imagens por batch: configurável (padrão: 50)
    - Tamanho máximo por imagem: configurável (padrão: 10MB)
    """
    settings = get_settings()
    service = get_service()
    start_time = time.time()
    
    # Validar número de arquivos
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Número máximo de imagens: {settings.max_batch_size}"
        )
    
    # Construir configurações (mesma lógica do endpoint único)
    detection_config = None
    if detection_confidence is not None or detection_iou is not None:
        detection_config = DetectionConfig(
            confidence=detection_confidence or settings.default_confidence,
            iou=detection_iou or settings.default_iou
        )
    
    ocr_config = None
    if ocr_engine is not None:
        try:
            from .schemas import OCREngine
            ocr_config = OCRConfig(engine=OCREngine(ocr_engine))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Engine de OCR inválida: {ocr_engine}"
            )
    
    options = ProcessingOptions(
        return_visualization=return_visualization,
        return_crops=return_crops,
        return_crop_images=return_crop_images,
        return_full_ocr=return_full_ocr,
        save_results=save_results
    )
    
    # Processar cada imagem
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validações
            if not validate_image_format(file.filename, settings.allowed_extensions):
                results.append(BatchImageResult(
                    filename=file.filename,
                    success=False,
                    error=f"Formato não suportado"
                ))
                failed += 1
                continue
            
            content = await file.read()
            if not validate_file_size(len(content), settings.get_max_file_size_bytes()):
                results.append(BatchImageResult(
                    filename=file.filename,
                    success=False,
                    error=f"Arquivo muito grande"
                ))
                failed += 1
                continue
            
            # Decodificar e processar
            image = decode_image(content)
            
            result = service.process_image(
                image=image,
                image_name=file.filename,
                detection_config=detection_config,
                ocr_config=ocr_config,
                options=options,
                request_id=getattr(request.state, "request_id", None)
            )
            
            results.append(BatchImageResult(
                filename=file.filename,
                success=True,
                result=result
            ))
            successful += 1
            
        except Exception as e:
            logger.error(f"Erro ao processar {file.filename}: {e}")
            results.append(BatchImageResult(
                filename=file.filename,
                success=False,
                error=str(e)
            ))
            failed += 1
    
    # Status geral
    if successful == len(files):
        batch_status = ProcessingStatus.SUCCESS
    elif successful > 0:
        batch_status = ProcessingStatus.PARTIAL
    else:
        batch_status = ProcessingStatus.FAILED
    
    return BatchProcessResponse(
        status=batch_status,
        total_images=len(files),
        successful=successful,
        failed=failed,
        results=results,
        total_time=time.time() - start_time
    )


# ========================================
# PROCESSAMENTO POR URL (OPCIONAL)
# ========================================

@router.post(
    "/process/url",
    response_model=ProcessImageResponse,
    summary="Processar imagem de URL",
    description="Baixa e processa uma imagem de uma URL",
)
async def process_url(
    request: Request,
    url: str = Form(..., description="URL da imagem"),
    detection_confidence: Optional[float] = Form(None),
    detection_iou: Optional[float] = Form(None),
    ocr_engine: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
    return_crops: bool = Form(False),
    return_crop_images: bool = Form(False),
    return_full_ocr: bool = Form(False),
    save_results: bool = Form(False),
):
    """
    Processa uma imagem a partir de uma URL.
    
    **Nota:** A URL deve ser acessível publicamente.
    """
    import requests
    
    settings = get_settings()
    service = get_service()
    request_id = getattr(request.state, "request_id", None)
    
    # Baixar imagem
    try:
        logger.info(f"Baixando imagem de: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.content
        
    except Exception as e:
        logger.error(f"Erro ao baixar imagem: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao baixar imagem: {str(e)}"
        )
    
    # Validar tamanho
    if not validate_file_size(len(content), settings.get_max_file_size_bytes()):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande. Máximo: {settings.max_file_size_mb}MB"
        )
    
    # Decodificar imagem
    try:
        image = decode_image(content)
    except Exception as e:
        logger.error(f"Erro ao decodificar imagem: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao decodificar imagem: {str(e)}"
        )
    
    # Construir configurações (mesma lógica dos outros endpoints)
    detection_config = None
    if detection_confidence is not None or detection_iou is not None:
        detection_config = DetectionConfig(
            confidence=detection_confidence or settings.default_confidence,
            iou=detection_iou or settings.default_iou
        )
    
    ocr_config = None
    if ocr_engine is not None:
        try:
            from .schemas import OCREngine
            ocr_config = OCRConfig(engine=OCREngine(ocr_engine))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Engine de OCR inválida: {ocr_engine}"
            )
    
    options = ProcessingOptions(
        return_visualization=return_visualization,
        return_crops=return_crops,
        return_crop_images=return_crop_images,
        return_full_ocr=return_full_ocr,
        save_results=save_results
    )
    
    # Processar
    try:
        image_name = Path(url).name or "url_image"
        
        result = service.process_image(
            image=image,
            image_name=image_name,
            detection_config=detection_config,
            ocr_config=ocr_config,
            options=options,
            request_id=request_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento: {str(e)}"
        )


__all__ = ["router"]
