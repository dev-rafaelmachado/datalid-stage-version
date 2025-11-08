"""
🔧 Serviço de Processamento
Integração com o pipeline existente do Datalid.
"""

import base64
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from src.ocr.postprocessors import DateParser
from src.pipeline.full_pipeline import FullPipeline, load_pipeline_config

from .config import get_settings
from .schemas import (BoundingBox, DetectionConfig, DetectionResult, OCRConfig,
                      OCRResult, ParsedDate, ProcessImageResponse,
                      ProcessingMetrics, ProcessingOptions, ProcessingStatus)


class ProcessingService:
    """
    Serviço principal de processamento.
    Gerencia o pipeline e cache de modelos.
    """
    
    def __init__(self):
        """Inicializa o serviço."""
        self.settings = get_settings()
        self.pipeline: Optional[FullPipeline] = None
        self._initialized = False
        
        logger.info("🔧 Serviço de processamento criado")
    
    def initialize(self) -> None:
        """Inicializa o pipeline e modelos."""
        if self._initialized:
            logger.debug("Serviço já inicializado")
            return
        
        logger.info("🚀 Inicializando pipeline...")
        
        try:
            # Carregar configuração do pipeline
            pipeline_config = self._build_pipeline_config()
            
            # Criar pipeline
            self.pipeline = FullPipeline(pipeline_config)
            
            self._initialized = True
            logger.success("✅ Pipeline inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar pipeline: {e}")
            raise
    
    def _build_pipeline_config(self) -> Dict[str, Any]:
        """Constrói configuração do pipeline."""
        return {
            'name': 'api_pipeline',
            'detection': {
                'model_path': self.settings.default_yolo_model,
                'confidence': self.settings.default_confidence,
                'iou': self.settings.default_iou,
                'device': self.settings.model_device,
            },
            'ocr': {
                'engine': self.settings.default_ocr_engine,
                'config': self.settings.default_ocr_config,
                'preprocessing': self.settings.default_preprocessing_config,
            },
            'output': {
                'output_dir': str(self.settings.get_results_path()),
                'save_visualizations': False,
                'save_crops': False,
            }
        }
    
    def process_image(
        self,
        image: np.ndarray,
        image_name: str = "image",
        detection_config: Optional[DetectionConfig] = None,
        ocr_config: Optional[OCRConfig] = None,
        options: Optional[ProcessingOptions] = None,
        request_id: Optional[str] = None,
    ) -> ProcessImageResponse:
        """
        Processa uma imagem através do pipeline.
        
        Args:
            image: Imagem numpy array (BGR)
            image_name: Nome da imagem
            detection_config: Configuração de detecção
            ocr_config: Configuração de OCR
            options: Opções de processamento
            request_id: ID da requisição
            
        Returns:
            Resultado do processamento
        """
        if not self._initialized:
            self.initialize()
        
        # Configurações padrão
        if options is None:
            options = ProcessingOptions()
        
        logger.info(f"🔄 Processando imagem: {image_name}")
        start_time = time.time()
        
        try:
            # Processar com o pipeline
            result = self.pipeline.process(image, image_name=image_name)
            
            # Converter resultado do pipeline para response da API
            response = self._convert_pipeline_result(
                result=result,
                image=image,
                options=options,
                request_id=request_id,
                start_time=start_time
            )
            
            # Salvar resultados se solicitado
            if options.save_results:
                self._save_results(image_name, response)
            
            logger.success(
                f"✅ Imagem processada: {len(response.dates)} data(s) encontrada(s)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar imagem: {e}")
            
            # Retornar resposta de erro
            return ProcessImageResponse(
                status=ProcessingStatus.FAILED,
                message=f"Erro no processamento: {str(e)}",
                detections=[],
                ocr_results=[],
                dates=[],
                best_date=None,
                metrics=ProcessingMetrics(
                    total_time=time.time() - start_time,
                    detection_time=0,
                    ocr_time=0,
                    parsing_time=0,
                    num_detections=0,
                    num_dates_found=0
                ),
                request_id=request_id
            )
    
    def _convert_pipeline_result(
        self,
        result: Dict[str, Any],
        image: np.ndarray,
        options: ProcessingOptions,
        request_id: Optional[str],
        start_time: float
    ) -> ProcessImageResponse:
        """Converte resultado do pipeline para response da API."""
        
        # Converter detecções
        detections = [
            self._convert_detection(det)
            for det in result.get('detections', [])
        ]
        
        # Converter resultados OCR
        ocr_results = [
            self._convert_ocr_result(ocr)
            for ocr in result.get('ocr_results', [])
        ] if options.return_full_ocr else []
        
        # Converter datas
        dates = [
            self._convert_date(date)
            for date in result.get('dates', [])
        ]
        
        # Melhor data
        best_date = None
        if result.get('best_date'):
            best_date = self._convert_date(result['best_date'])
        
        # Métricas
        metrics = ProcessingMetrics(
            total_time=time.time() - start_time,
            detection_time=result.get('detection_time', 0),
            ocr_time=result.get('ocr_time', 0),
            parsing_time=result.get('parsing_time', 0),
            num_detections=len(detections),
            num_dates_found=len(dates)
        )
        
        # Status
        if len(dates) > 0:
            status = ProcessingStatus.SUCCESS
            message = f"{len(dates)} data(s) de validade encontrada(s)"
        elif len(detections) > 0:
            status = ProcessingStatus.PARTIAL
            message = "Regiões detectadas mas nenhuma data válida encontrada"
        else:
            status = ProcessingStatus.FAILED
            message = "Nenhuma região de data detectada"
        
        # Visualizações (opcional)
        visualization_base64 = None
        if options.return_visualization and result.get('success'):
            visualization_base64 = self._create_visualization(image, result)
        
        # Crops (opcional)
        crops_base64 = None
        if options.return_crops and len(detections) > 0:
            crops_base64 = self._extract_crops(image, result.get('detections', []))
        
        return ProcessImageResponse(
            status=status,
            message=message,
            detections=detections,
            ocr_results=ocr_results,
            dates=dates,
            best_date=best_date,
            metrics=metrics,
            visualization_base64=visualization_base64,
            crops_base64=crops_base64,
            request_id=request_id
        )
    
    def _convert_detection(self, detection: Dict[str, Any]) -> DetectionResult:
        """Converte detecção do pipeline para schema da API."""
        bbox_data = detection.get('bbox', {})
        
        bbox = BoundingBox(
            x1=bbox_data.get('x1', 0),
            y1=bbox_data.get('y1', 0),
            x2=bbox_data.get('x2', 0),
            y2=bbox_data.get('y2', 0),
            width=bbox_data.get('width', 0),
            height=bbox_data.get('height', 0)
        )
        
        return DetectionResult(
            bbox=bbox,
            confidence=detection.get('confidence', 0.0),
            class_id=detection.get('class_id', 0),
            class_name=detection.get('class_name', 'exp_date'),
            has_mask=detection.get('has_mask', False)
        )
    
    def _convert_ocr_result(self, ocr: Dict[str, Any]) -> OCRResult:
        """Converte resultado OCR do pipeline para schema da API."""
        return OCRResult(
            text=ocr.get('text', ''),
            confidence=ocr.get('confidence', 0.0),
            engine=ocr.get('engine', 'unknown'),
            processing_time=ocr.get('processing_time', 0.0)
        )
    
    def _convert_date(self, date: Dict[str, Any]) -> ParsedDate:
        """Converte data do pipeline para schema da API."""
        date_obj = date.get('date')
        date_str = None
        is_expired = None
        days_until_expiry = None
        
        if isinstance(date_obj, datetime):
            date_str = date_obj.strftime('%Y-%m-%d')
            
            # Calcular se está expirado
            today = datetime.now()
            delta = (date_obj - today).days
            days_until_expiry = delta
            is_expired = delta < 0
        
        # O pipeline usa 'combined_confidence' ao invés de 'confidence'
        confidence = date.get('combined_confidence', date.get('confidence', 0.0))
        
        return ParsedDate(
            date=date_str,
            confidence=confidence,
            format=date.get('format'),
            is_valid=date.get('is_valid', True),
            is_expired=is_expired,
            days_until_expiry=days_until_expiry,
            text=date.get('text', '')
        )
    
    def _create_visualization(
        self,
        image: np.ndarray,
        result: Dict[str, Any]
    ) -> str:
        """Cria visualização e retorna em base64."""
        try:
            # Copiar imagem
            vis_image = image.copy()
            
            # Desenhar detecções
            for detection in result.get('detections', []):
                bbox = detection.get('bbox', {})
                x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                conf = detection.get('confidence', 0.0)
                
                # Retângulo
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label
                label = f"{conf:.2f}"
                cv2.putText(
                    vis_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Adicionar data encontrada
            best_date = result.get('best_date')
            if best_date:
                date_obj = best_date.get('date')
                if isinstance(date_obj, datetime):
                    date_str = date_obj.strftime('%d/%m/%Y')
                    cv2.putText(
                        vis_image, f"Data: {date_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                    )
            
            # Converter para base64
            _, buffer = cv2.imencode('.jpg', vis_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.warning(f"Erro ao criar visualização: {e}")
            return None
    
    def _extract_crops(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[str]:
        """Extrai crops das detecções e retorna em base64."""
        crops = []
        
        try:
            for detection in detections:
                bbox = detection.get('bbox', {})
                x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                
                # Extrair crop
                crop = image[y1:y2, x1:x2]
                
                # Converter para base64
                _, buffer = cv2.imencode('.jpg', crop)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                crops.append(crop_base64)
        
        except Exception as e:
            logger.warning(f"Erro ao extrair crops: {e}")
        
        return crops if crops else None
    
    def _save_results(
        self,
        image_name: str,
        response: ProcessImageResponse
    ) -> None:
        """Salva resultados em arquivo."""
        try:
            results_dir = self.settings.get_results_path()
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{image_name}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Salvar JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.model_dump_json(indent=2))
            
            logger.info(f"💾 Resultados salvos: {filepath}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar resultados: {e}")
    
    def is_ready(self) -> bool:
        """Verifica se o serviço está pronto."""
        return self._initialized and self.pipeline is not None


# ========================================
# SINGLETON
# ========================================

_service: Optional[ProcessingService] = None


def get_service() -> ProcessingService:
    """
    Retorna instância singleton do serviço.
    
    Returns:
        Serviço de processamento
    """
    global _service
    if _service is None:
        _service = ProcessingService()
    return _service


__all__ = ["ProcessingService", "get_service"]
