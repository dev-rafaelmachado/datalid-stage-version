"""
🚀 Full Pipeline: YOLO → OCR → Parse
Pipeline completo para detecção e extração de datas de validade.

Fluxo:
1. YOLO detecta região da data (bounding box + máscara)
2. Crop e processamento da região
3. OCR extrai texto
4. Parse e validação da data
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from src.ocr.config import load_ocr_config, load_preprocessing_config
from src.ocr.engines.easyocr import EasyOCREngine
from src.ocr.engines.openocr import OpenOCREngine
from src.ocr.engines.paddleocr import PaddleOCREngine
from src.ocr.engines.parseq import PARSeqEngine
from src.ocr.engines.parseq_enhanced import EnhancedPARSeqEngine
from src.ocr.engines.tesseract import TesseractEngine
from src.ocr.engines.trocr import TrOCREngine
from src.ocr.postprocessors import DateParser
from src.ocr.preprocessors import ImagePreprocessor
from src.pipeline.base import PipelineBase


class FullPipeline(PipelineBase):
    """
    Pipeline completo: YOLO → OCR → Parse.
    
    Detecta, extrai e valida datas de validade em imagens de produtos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline completo.
        
        Args:
            config: Configuração do pipeline (formato YAML)
        """
        super().__init__(config)
        
        # Configuração de detecção (YOLO)
        self.detection_config = config.get('detection', {})
        self.yolo_model = None
        self._load_yolo_model()
        
        # Configuração de OCR
        self.ocr_config = config.get('ocr', {})
        self.ocr_engine = None
        self.preprocessor = None
        self._load_ocr_engine()
        
        # Configuração de parsing
        self.parsing_config = config.get('parsing', {})
        self.date_parser = DateParser(self.parsing_config)
        
        # Opções de visualização
        self.save_visualizations = config.get('output', {}).get('save_visualizations', True)
        self.save_crops = config.get('output', {}).get('save_crops', False)
        
        logger.info(f"✅ {self.name} inicializado com sucesso!")
    
    def _load_yolo_model(self) -> None:
        """Carrega modelo YOLO."""
        from ultralytics import YOLO
        
        model_path = self.detection_config.get('model_path')
        if not model_path:
            raise ValueError("❌ 'detection.model_path' não especificado na configuração")
        
        logger.info(f"🔄 Carregando modelo YOLO: {model_path}")
        self.yolo_model = YOLO(model_path)
        logger.info("✅ Modelo YOLO carregado")
    
    def _load_ocr_engine(self) -> None:
        """Carrega engine OCR e preprocessador."""
        # Carregar configuração do OCR
        ocr_config_path = self.ocr_config.get('config')
        if ocr_config_path:
            engine_config = load_ocr_config(ocr_config_path)
        else:
            # Configuração padrão (PaddleOCR)
            engine_config = {'engine': 'openocr'}
        
        # Determinar engine
        engine_name = engine_config.get('engine', 'openocr')
        
        # Criar engine
        logger.info(f"🔄 Inicializando OCR engine: {engine_name}")
        engine_class = {
            'tesseract': TesseractEngine,
            'easyocr': EasyOCREngine,
            'openocr': OpenOCREngine,
            'paddleocr': PaddleOCREngine,
            'parseq': PARSeqEngine,
            'parseq_enhanced': EnhancedPARSeqEngine,
            'trocr': TrOCREngine
        }.get(engine_name.lower())
        
        if engine_class is None:
            raise ValueError(f"❌ Engine desconhecido: {engine_name}")
        
        self.ocr_engine = engine_class(engine_config)
        self.ocr_engine.initialize()
        logger.info(f"✅ OCR engine '{engine_name}' inicializado")
        
        # Carregar preprocessador se especificado
        preprocessing_config_path = self.ocr_config.get('preprocessing')
        if preprocessing_config_path:
            logger.info(f"🔄 Carregando preprocessador: {preprocessing_config_path}")
            prep_config = load_preprocessing_config(preprocessing_config_path)
            self.preprocessor = ImagePreprocessor(prep_config)
            logger.info("✅ Preprocessador carregado")
    
    def process(self, image: np.ndarray, image_name: str = "image", **kwargs) -> Dict[str, Any]:
        """
        Processa uma imagem através do pipeline completo.
        
        Args:
            image: Imagem numpy array (BGR)
            image_name: Nome da imagem (para logs/salvamento)
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com resultados:
            {
                'success': bool,
                'detections': List[Dict],  # Detecções YOLO
                'ocr_results': List[Dict],  # Resultados OCR por detecção
                'dates': List[Dict],  # Datas extraídas e validadas
                'best_date': Optional[Dict],  # Melhor data (maior confiança)
                'processing_time': float
            }
        """
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Processando imagem: {image_name}")
        
        # 1. DETECÇÃO YOLO
        logger.info("📍 [1/3] Executando detecção YOLO...")
        detections = self._detect_regions(image)
        
        if not detections:
            logger.warning("⚠️  Nenhuma região detectada pelo YOLO")
            return {
                'success': False,
                'detections': [],
                'ocr_results': [],
                'dates': [],
                'best_date': None,
                'processing_time': time.time() - start_time,
                'error': 'No detections'
            }
        
        logger.info(f"✅ {len(detections)} região(ões) detectada(s)")
        
        # 2. OCR EM CADA REGIÃO
        logger.info("🔍 [2/3] Executando OCR nas regiões...")
        ocr_results = []
        for i, detection in enumerate(detections):
            logger.info(f"   Processando região {i+1}/{len(detections)}...")
            logger.debug(f"   Detection keys: {detection.keys()}")
            logger.debug(f"   BBox type: {type(detection.get('bbox'))}")
            logger.debug(f"   Has mask: {detection.get('has_mask', False)}")
            
            # Extrair crop
            crop = self._extract_crop(image, detection)
            
            # Salvar crop se configurado
            if self.save_crops:
                self._save_crop(crop, image_name, i)
            
            # Pré-processar
            if self.preprocessor:
                crop_processed = self.preprocessor.process(crop)
            else:
                crop_processed = crop
            
            # OCR
            ocr_start = time.time()
            text, confidence = self.ocr_engine.extract_text(crop_processed)
            ocr_time = time.time() - ocr_start
            
            ocr_result = {
                'detection_index': i,
                'text': text,
                'confidence': confidence,
                'bbox': detection['bbox'],
                'engine': self.ocr_engine.__class__.__name__,
                'processing_time': ocr_time,
                'crop_original': crop,  # Crop original
                'crop_processed': crop_processed  # Crop pré-processado
            }
            ocr_results.append(ocr_result)
            
            logger.info(f"   ✓ Texto: '{text}' (conf: {confidence:.2%})")
        
        # 3. PARSING DE DATAS
        logger.info("📅 [3/3] Fazendo parse de datas...")
        dates = []
        for ocr_result in ocr_results:
            parsed_date, parse_confidence = self.date_parser.parse(ocr_result['text'])
            
            if parsed_date:
                date_result = {
                    'date': parsed_date,
                    'date_str': parsed_date.strftime('%d/%m/%Y'),
                    'text': ocr_result['text'],
                    'ocr_confidence': ocr_result['confidence'],
                    'parse_confidence': parse_confidence,
                    'combined_confidence': (ocr_result['confidence'] + parse_confidence) / 2,
                    'bbox': ocr_result['bbox']
                }
                dates.append(date_result)
                logger.info(f"   ✓ Data extraída: {date_result['date_str']} (conf: {date_result['combined_confidence']:.2%})")
            else:
                logger.debug(f"   ✗ Não foi possível extrair data de: '{ocr_result['text']}'")
        
        # Melhor data (maior confiança)
        best_date = None
        if dates:
            best_date = max(dates, key=lambda x: x['combined_confidence'])
            logger.info(f"🏆 Melhor resultado: {best_date['date_str']} (conf: {best_date['combined_confidence']:.2%})")
        else:
            logger.warning("⚠️  Nenhuma data válida extraída")
        
        # Resultado final
        processing_time = time.time() - start_time
        result = {
            'success': len(dates) > 0,
            'detections': detections,
            'ocr_results': ocr_results,
            'dates': dates,
            'best_date': best_date,
            'processing_time': processing_time,
            'image_name': image_name
        }
        
        logger.info(f"✅ Pipeline concluído em {processing_time:.2f}s")
        
        # Salvar visualização se configurado
        if self.save_visualizations:
            self._save_visualization(image, result, image_name)
        
        return result
    
    def _detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta regiões de interesse com YOLO (SEGMENTAÇÃO POLIGONAL)."""
        confidence = self.detection_config.get('confidence', 0.25)
        iou = self.detection_config.get('iou', 0.7)
        device = self.detection_config.get('device', 0)
        
        # Predição
        results = self.yolo_model.predict(
            image,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False
        )[0]
        
        # Extrair detecções
        detections = []
        h, w = image.shape[:2]
        
        if len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                # Converter bbox para formato estruturado (mantido para compatibilidade)
                bbox_coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = bbox_coords
                
                detection = {
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': results.names[int(box.cls[0])],
                    'has_mask': False,
                    'segmentation': None  # Coordenadas do polígono
                }
                
                # Adicionar MÁSCARA e POLÍGONO se disponível (PRIORIDADE)
                if hasattr(results, 'masks') and results.masks is not None:
                    mask = results.masks.data[i].cpu().numpy()
                    detection['mask'] = mask
                    detection['has_mask'] = True
                    
                    # Extrair coordenadas do polígono
                    polygon = self._extract_polygon_from_mask(mask, (h, w))
                    detection['segmentation'] = polygon
                    
                    logger.debug(f"   🎭 Detecção {i+1}: Máscara com {len(polygon)} pontos")
                else:
                    logger.debug(f"   📦 Detecção {i+1}: Bbox retangular (sem máscara)")
                
                detections.append(detection)
        
        return detections
    
    def _extract_polygon_from_mask(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> List[List[int]]:
        """
        Extrai coordenadas do polígono a partir da máscara de segmentação.
        
        Args:
            mask: Máscara de segmentação do YOLO
            image_shape: (height, width) da imagem original
            
        Returns:
            Lista de pontos do polígono [[x1, y1], [x2, y2], ...]
        """
        h, w = image_shape
        
        # Redimensionar máscara para o tamanho da imagem
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_resized = mask
        
        # Converter para binária
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Pegar o maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Converter para lista de pontos
        polygon = largest_contour.reshape(-1, 2).tolist()
        
        return polygon
    
    def _extract_bbox_coords(self, bbox: Any) -> Tuple[int, int, int, int]:
        """
        Extrai coordenadas x1, y1, x2, y2 de um bbox em qualquer formato.
        
        Suporta:
        - Dict: {'x1': float, 'y1': float, 'x2': float, 'y2': float}
        - List/Tuple: [x1, y1, x2, y2]
        - Qualquer combinação dos acima
        
        Args:
            bbox: BBox em qualquer formato suportado
            
        Returns:
            Tupla (x1, y1, x2, y2) como inteiros
        """
        try:
            if isinstance(bbox, dict):
                # Formato dicionário - tentar várias chaves possíveis
                x1 = float(bbox.get('x1', bbox.get('xmin', bbox.get('left', 0))))
                y1 = float(bbox.get('y1', bbox.get('ymin', bbox.get('top', 0))))
                x2 = float(bbox.get('x2', bbox.get('xmax', bbox.get('right', 0))))
                y2 = float(bbox.get('y2', bbox.get('ymax', bbox.get('bottom', 0))))
                
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                # Formato lista/tupla
                x1, y1, x2, y2 = map(float, bbox[:4])
                
            else:
                raise ValueError(f"❌ Formato de bbox não reconhecido: {type(bbox)}")
            
            # Converter para int e validar
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Garantir ordem correta (x1 < x2, y1 < y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            return x1, y1, x2, y2
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"❌ Erro ao extrair coordenadas do bbox: {bbox}")
            logger.error(f"   Erro: {e}")
            raise ValueError(f"Formato de bbox inválido: {bbox}. Erro: {e}")
    
    def _extract_crop(self, image: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """
        Extrai crop da região detectada usando SEGMENTAÇÃO POLIGONAL.
        
        Prioridade:
        1. Se tiver máscara de segmentação: usa a máscara (PRIORIDADE)
        2. Se não tiver: fallback para bbox retangular
        
        Args:
            image: Imagem original
            detection: Dicionário com 'bbox' e opcionalmente 'mask'
            
        Returns:
            Crop da região com máscara aplicada
        """
        h, w = image.shape[:2]
        
        # PRIORIDADE: Usar máscara de segmentação se disponível
        if detection.get('has_mask', False) and 'mask' in detection and detection['mask'] is not None:
            logger.debug("   🎭 Usando MÁSCARA DE SEGMENTAÇÃO (poligonal)")
            mask = detection['mask']
            
            # A máscara do YOLO vem em escala reduzida, precisa redimensionar
            # para o tamanho da imagem original
            if mask.shape != (h, w):
                mask_resized = cv2.resize(
                    mask, 
                    (w, h), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                mask_resized = mask
            
            # Converter máscara para binária (threshold)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Encontrar o bounding box MÍNIMO que contém a máscara (para crop eficiente)
            # Isso evita processar pixels desnecessários, mas mantém a forma poligonal
            coords = np.column_stack(np.where(mask_binary > 0))
            
            if coords.size == 0:
                logger.warning("⚠️ Máscara vazia! Usando bbox como fallback.")
                x1, y1, x2, y2 = self._extract_bbox_coords(detection['bbox'])
            else:
                # Encontrar limites da máscara
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Adicionar pequeno padding para não cortar nas bordas
                padding = 5
                x1 = max(0, x_min - padding)
                y1 = max(0, y_min - padding)
                x2 = min(w, x_max + padding)
                y2 = min(h, y_max + padding)
            
            # Criar imagem mascarada (fundo branco para melhor OCR)
            masked_image = image.copy()
            
            # Aplicar máscara poligonal: região de interesse fica, resto fica branco
            for c in range(image.shape[2]):
                masked_image[:, :, c] = np.where(
                    mask_binary > 0,
                    image[:, :, c],
                    255  # Fundo branco
                )
            
            # Extrair crop da imagem mascarada (crop mínimo que contém a máscara)
            crop = masked_image[y1:y2, x1:x2]
            logger.debug(f"   ✓ Crop com máscara: {crop.shape}")
            
        else:
            # FALLBACK: Usar bbox retangular se não tiver máscara
            logger.debug("   📦 Usando BBOX RETANGULAR (fallback)")
            x1, y1, x2, y2 = self._extract_bbox_coords(detection['bbox'])
            
            # Garantir que está dentro dos limites
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop simples
            crop = image[y1:y2, x1:x2].copy()
            logger.debug(f"   ✓ Crop retangular: {crop.shape}")
        
        # Validar crop
        if crop.size == 0:
            logger.warning(f"⚠️ Crop vazio! Criando imagem placeholder.")
            # Retornar imagem pequena válida para não quebrar
            crop = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        return crop
    
    def _save_crop(self, crop: np.ndarray, image_name: str, index: int) -> None:
        """Salva crop em arquivo."""
        crop_dir = self.output_dir / "crops" / image_name
        crop_dir.mkdir(parents=True, exist_ok=True)
        
        crop_path = crop_dir / f"crop_{index}.jpg"
        cv2.imwrite(str(crop_path), crop)
    
    def _save_visualization(self, image: np.ndarray, result: Dict[str, Any], image_name: str) -> None:
        """Salva visualização com anotações de SEGMENTAÇÃO POLIGONAL."""
        vis = image.copy()
        h, w = vis.shape[:2]
        
        # Desenhar detecções com MÁSCARA POLIGONAL (se disponível)
        for detection in result['detections']:
            
            # Se tiver máscara, desenhar contorno poligonal
            if detection.get('has_mask', False) and 'mask' in detection and detection['mask'] is not None:
                mask = detection['mask']
                
                # Redimensionar máscara para o tamanho da imagem
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask_resized = mask
                
                # Converter para binária
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Encontrar contornos da máscara
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Desenhar contornos POLIGONAIS (verde)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
                
                # Desenhar máscara semi-transparente (overlay verde)
                overlay = vis.copy()
                overlay[mask_binary > 0] = overlay[mask_binary > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
                vis = overlay.astype(np.uint8)
                
                # Posição do texto (topo do contorno)
                if contours:
                    x, y, w_box, h_box = cv2.boundingRect(contours[0])
                    text_pos = (x, y - 10)
                else:
                    text_pos = (10, 30)
            else:
                # Fallback: desenhar bbox retangular
                bbox = detection['bbox']
                x1, y1, x2, y2 = self._extract_bbox_coords(bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_pos = (x1, y1 - 10)
            
            # Confiança da detecção
            conf_text = f"Det: {detection['confidence']:.2f}"
            cv2.putText(vis, conf_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Desenhar melhor data (destaque em vermelho)
        if result['best_date']:
            best = result['best_date']
            
            # Encontrar a detecção correspondente para usar sua máscara
            best_detection = None
            for det in result['detections']:
                if det['bbox'] == best['bbox']:
                    best_detection = det
                    break
            
            if best_detection and best_detection.get('has_mask', False) and 'mask' in best_detection:
                mask = best_detection['mask']
                
                # Redimensionar máscara
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask_resized = mask
                
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Contornos em vermelho (destaque)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 0, 255), 3)
                
                # Posição do texto
                if contours:
                    x, y, w_box, h_box = cv2.boundingRect(contours[0])
                    text_pos = (x, y + h_box + 25)
                else:
                    text_pos = (10, h - 30)
            else:
                # Fallback: bbox retangular
                bbox = best['bbox']
                x1, y1, x2, y2 = self._extract_bbox_coords(bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                text_pos = (x1, y2 + 25)
            
            # Data extraída
            date_text = f"Data: {best['date_str']} ({best['combined_confidence']:.1%})"
            cv2.putText(vis, date_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Salvar
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        vis_path = vis_dir / f"{image_name}_result.jpg"
        cv2.imwrite(str(vis_path), vis)
        
        logger.debug(f"💾 Visualização salva: {vis_path}")
    
    def process_batch(self, images: List[np.ndarray], image_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Processa múltiplas imagens.
        
        Args:
            images: Lista de imagens
            image_names: Nomes das imagens (opcional)
            
        Returns:
            Lista de resultados
        """
        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(images))]
        
        results = []
        for i, (image, name) in enumerate(zip(images, image_names)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processando {i+1}/{len(images)}: {name}")
            logger.info(f"{'='*60}")
            
            result = self.process(image, name, **kwargs)
            results.append(result)
        
        return results
    
    def process_directory(self, image_dir: str, pattern: str = "*.jpg", **kwargs) -> List[Dict[str, Any]]:
        """
        Processa todas as imagens de um diretório.
        
        Args:
            image_dir: Diretório com imagens
            pattern: Padrão de arquivos (ex: "*.jpg", "*.png")
            
        Returns:
            Lista de resultados
        """
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.glob(pattern))
        
        if not image_paths:
            logger.warning(f"⚠️  Nenhuma imagem encontrada em: {image_dir}")
            return []
        
        logger.info(f"📁 Processando {len(image_paths)} imagens de: {image_dir}")
        
        results = []
        for i, img_path in enumerate(image_paths):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processando {i+1}/{len(image_paths)}: {img_path.name}")
            logger.info(f"{'='*60}")
            
            # Ler imagem
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"❌ Erro ao ler: {img_path}")
                continue
            
            # Processar
            result = self.process(image, img_path.stem, **kwargs)
            result['image_path'] = str(img_path)
            results.append(result)
        
        # Salvar resumo
        self._save_batch_summary(results)
        
        return results
    
    def _save_batch_summary(self, results: List[Dict[str, Any]]) -> None:
        """Salva resumo do processamento em batch."""
        summary = {
            'total_images': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'avg_processing_time': sum(r['processing_time'] for r in results) / len(results) if results else 0,
            'results': []
        }
        
        for result in results:
            summary_item = {
                'image_name': result.get('image_name', 'unknown'),
                'success': result['success'],
                'processing_time': result['processing_time'],
            }
            
            if result['best_date']:
                summary_item['date'] = result['best_date']['date_str']
                summary_item['confidence'] = result['best_date']['combined_confidence']
            
            summary['results'].append(summary_item)
        
        # Salvar JSON
        summary_path = self.output_dir / "batch_summary.json"
        self.save_results(summary, str(summary_path))
        
        # Log resumo
        logger.info(f"\n{'='*60}")
        logger.info("📊 RESUMO DO PROCESSAMENTO")
        logger.info(f"{'='*60}")
        logger.info(f"Total de imagens: {summary['total_images']}")
        logger.info(f"✅ Sucesso: {summary['successful']}")
        logger.info(f"❌ Falhas: {summary['failed']}")
        logger.info(f"⏱️  Tempo médio: {summary['avg_processing_time']:.2f}s")
        logger.info(f"{'='*60}\n")


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega configuração do pipeline de arquivo YAML.
    
    Args:
        config_path: Caminho para arquivo de configuração
        
    Returns:
        Dicionário de configuração
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


__all__ = ['FullPipeline', 'load_pipeline_config']
