"""
📊 Avaliação Completa da Pipeline
Avalia pipeline end-to-end (YOLO → OCR → Parsing) com métricas detalhadas.

Funcionalidades:
- Avaliação em múltiplas imagens (configurável)
- Métricas por etapa (detecção, OCR, parsing)
- Métricas end-to-end
- Análise de erros detalhada
- Visualizações e relatórios
- Comparação com ground truth

Uso:
    # Avaliar com configuração padrão
    python scripts/pipeline/evaluate_pipeline.py
    
    # Com config customizada
    python scripts/pipeline/evaluate_pipeline.py --config config/pipeline/pipeline_evaluation.yaml
    
    # Avaliar apenas 10 imagens
    python scripts/pipeline/evaluate_pipeline.py --num-images 10
    
    # Modo de seleção aleatória
    python scripts/pipeline/evaluate_pipeline.py --num-images 20 --selection-mode random
    
    # Output customizado
    python scripts/pipeline/evaluate_pipeline.py --output outputs/minha_avaliacao
"""

import argparse
import json
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from tqdm import tqdm

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.full_pipeline import FullPipeline, load_pipeline_config


class PipelineEvaluator:
    """Avaliador completo da pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o avaliador.
        
        Args:
            config: Configuração da avaliação
        """
        self.config = config
        self.pipeline = None
        self.results = []
        self.metrics_summary = {}
        self.error_analysis = {}
        
        # Para salvar imagens de debug
        self.debug_images_count = 0
        self.max_debug_images = 3  # Salvar apenas 3 exemplos
        self.debug_output_dir = None
        
    def initialize_pipeline(self):
        """Inicializa a pipeline."""
        logger.info("🔄 Inicializando pipeline...")
        
        pipeline_config_path = self.config['pipeline']['config_file']
        pipeline_config = load_pipeline_config(pipeline_config_path)
        
        self.pipeline = FullPipeline(pipeline_config)
        
        logger.info("✅ Pipeline inicializada!")
        
    def load_ground_truth(self) -> Dict[str, str]:
        """
        Carrega ground truth e extrai expiry_date por imagem.
        
        Suporta dois formatos:
        1. Formato antigo: {"annotations": {"image.jpg": "text", ...}}
        2. Formato novo: {"detection_id": {"image": "image.jpg", "expiry_date": "...", ...}, ...}
        
        Returns:
            Dict mapeando nome da imagem para expiry_date esperado
        """
        gt_path = Path(self.config['dataset']['ground_truth'])
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth não encontrado: {gt_path}")
        
        logger.info(f"📥 Carregando ground truth: {gt_path}")
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Extrair anotações baseado no formato
        json_key = self.config['validation']['ground_truth_format']['json_key']
        field_name = self.config['validation']['ground_truth_format'].get('field_name', 'expiry_date')
        
        annotations = {}
        
        # Tenta o formato antigo primeiro
        if json_key in gt_data:
            raw_annotations = gt_data[json_key]
            # Se for o formato simples (image -> text)
            if isinstance(next(iter(raw_annotations.values()), None), str):
                annotations = raw_annotations
            else:
                # Formato nested
                for key, value in raw_annotations.items():
                    if isinstance(value, dict) and 'image' in value:
                        image_name = value['image']
                        expiry = value.get(field_name, '')
                        annotations[image_name] = expiry
        else:
            # Formato novo direto: cada chave é uma detecção
            for detection_id, detection_data in gt_data.items():
                if isinstance(detection_data, dict) and 'image' in detection_data:
                    image_name = detection_data['image']
                    expiry = detection_data.get(field_name, '')
                    # Pega a primeira detecção de cada imagem
                    if image_name not in annotations:
                        annotations[image_name] = expiry
        
        logger.info(f"✅ Ground truth carregado: {len(annotations)} anotações")
        
        return annotations
    
    def _extract_image_number(self, filename: str) -> Optional[str]:
        """
        Extrai o número identificador de uma imagem.
        
        Exemplo: "train_101_jpg_6fee8d46fda6271dea968494c8a2c663.jpg" -> "101"
        
        Args:
            filename: Nome do arquivo ou ID da imagem
            
        Returns:
            Número extraído ou None se não encontrado
        """
        import re

        # Procura por padrão _XXX_ onde XXX são dígitos
        match = re.search(r'_(\d+)_', filename)
        if match:
            return match.group(1)
        # Fallback: procura qualquer sequência de dígitos
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
        return None
    
    def _build_image_mapping(self, annotations: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        Cria mapeamento entre números de imagens e arquivos reais.
        
        Args:
            annotations: Dicionário com anotações (keys são IDs do ground truth)
            
        Returns:
            Dict mapeando número da imagem -> (ground_truth_id, arquivo_real, expiry_date)
        """
        images_dir = Path(self.config['dataset']['images_dir'])
        
        # Mapear números das anotações
        gt_number_to_id = {}
        for gt_id, expiry_date in annotations.items():
            number = self._extract_image_number(gt_id)
            if number:
                gt_number_to_id[number] = (gt_id, expiry_date)
                logger.debug(f"  GT: '{gt_id}' -> número '{number}' -> expiry '{expiry_date}'")
        
        # Mapear números dos arquivos reais
        real_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        number_to_file = {}
        for file_path in real_files:
            number = self._extract_image_number(file_path.name)
            if number:
                number_to_file[number] = file_path.name
                logger.debug(f"  Arquivo: '{file_path.name}' -> número '{number}'")
        
        # Criar mapeamento final: número -> (gt_id, arquivo_real, expiry_date)
        mapping = {}
        matched_numbers = []
        unmatched_gt = []
        unmatched_files = []
        
        for number, (gt_id, expiry_date) in gt_number_to_id.items():
            if number in number_to_file:
                mapping[number] = (gt_id, number_to_file[number], expiry_date)
                matched_numbers.append(number)
                logger.debug(f"  ✅ Match #{number}: GT '{gt_id}' <-> File '{number_to_file[number]}'")
            else:
                unmatched_gt.append((number, gt_id))
                logger.warning(f"  ❌ Número {number} do ground truth '{gt_id}' não encontrado nos arquivos")
        
        # Arquivos sem match no GT
        for number, filename in number_to_file.items():
            if number not in gt_number_to_id:
                unmatched_files.append((number, filename))
        
        # Relatório de mapeamento
        logger.info(f"🔗 Mapeamento criado:")
        logger.info(f"   📝 Ground truth entries: {len(annotations)}")
        logger.info(f"   📁 Arquivos encontrados: {len(real_files)}")
        logger.info(f"   ✅ Matches bem-sucedidos: {len(mapping)}")
        
        if unmatched_gt:
            logger.warning(f"   ⚠️  GT sem arquivo correspondente: {len(unmatched_gt)}")
            for num, gt_id in unmatched_gt[:3]:  # Mostra apenas os 3 primeiros
                logger.warning(f"      - #{num}: {gt_id}")
            if len(unmatched_gt) > 3:
                logger.warning(f"      ... e mais {len(unmatched_gt) - 3}")
        
        if unmatched_files:
            logger.info(f"   ℹ️  Arquivos sem GT: {len(unmatched_files)}")
            for num, filename in unmatched_files[:3]:
                logger.info(f"      - #{num}: {filename}")
            if len(unmatched_files) > 3:
                logger.info(f"      ... e mais {len(unmatched_files) - 3}")
        
        return mapping
    
    def select_images(self, annotations: Dict[str, str]) -> List[Tuple[str, str, str]]:
        """
        Seleciona imagens para avaliar e faz matching com arquivos reais.
        
        Args:
            annotations: Dicionário com anotações
            
        Returns:
            Lista de tuplas (número, arquivo_real, expiry_date)
        """
        # Criar mapeamento entre ground truth e arquivos reais
        mapping = self._build_image_mapping(annotations)
        
        if not mapping:
            logger.error("❌ Nenhuma imagem foi mapeada com sucesso!")
            return []
        
        all_numbers = list(mapping.keys())
        num_images = self.config['dataset'].get('num_images')
        selection_mode = self.config['dataset'].get('selection_mode', 'random')
        
        # Se num_images for None, usar todas
        if num_images is None or selection_mode == 'all':
            selected_numbers = all_numbers
        elif selection_mode == 'first':
            selected_numbers = sorted(all_numbers, key=lambda x: int(x))[:num_images]
        elif selection_mode == 'random':
            import random
            random.seed(self.config['dataset'].get('random_seed', 42))
            selected_numbers = random.sample(all_numbers, min(num_images, len(all_numbers)))
        else:
            logger.warning(f"⚠️ Modo de seleção inválido: {selection_mode}, usando 'all'")
            selected_numbers = all_numbers
        
        # Converter para lista de tuplas (número, arquivo_real, expiry_date)
        selected = [(num, mapping[num][1], mapping[num][2]) for num in selected_numbers]
        
        logger.info(f"📸 Imagens selecionadas: {len(selected)}/{len(all_numbers)}")
        logger.info(f"   Modo: {selection_mode}")
        
        return selected
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Executa avaliação completa.
        
        Returns:
            Dicionário com resultados e métricas
        """
        logger.info("=" * 80)
        logger.info("📊 INICIANDO AVALIAÇÃO DA PIPELINE")
        logger.info("=" * 80)
        
        # Carregar ground truth
        annotations = self.load_ground_truth()
        
        # Selecionar imagens
        selected_images = self.select_images(annotations)
        
        # Diretório de imagens
        images_dir = Path(self.config['dataset']['images_dir'])
        
        # Processar cada imagem
        logger.info(f"\n🔄 Processando {len(selected_images)} imagens...")
        
        for number, real_filename, expiry_date in tqdm(selected_images, desc="Avaliando"):
            img_path = images_dir / real_filename
            
            if not img_path.exists():
                logger.warning(f"⚠️ Imagem não encontrada: {real_filename} (número: {number})")
                continue
            
            # Processar imagem com o arquivo real e expiry_date correto
            result = self._evaluate_single_image(
                img_path,
                real_filename,
                expiry_date
            )
            # Adicionar número da imagem ao resultado para rastreamento
            result['image_number'] = number
            
            self.results.append(result)
        
        # Calcular métricas agregadas
        logger.info("\n📊 Calculando métricas agregadas...")
        self.metrics_summary = self._calculate_metrics_summary()
        
        # Análise de erros
        if self.config['error_analysis']['enabled']:
            logger.info("\n🔍 Analisando erros...")
            self.error_analysis = self._analyze_errors()
        
        return {
            'results': self.results,
            'metrics_summary': self.metrics_summary,
            'error_analysis': self.error_analysis
        }
    
    def _evaluate_single_image(
        self,
        img_path: Path,
        img_name: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Avalia uma única imagem.
        
        Args:
            img_path: Caminho da imagem
            img_name: Nome da imagem
            ground_truth: Texto esperado (ground truth)
            
        Returns:
            Dicionário com resultados
        """
        result = {
            'image_name': img_name,
            'image_path': str(img_path),
            'ground_truth': ground_truth,
            'success': False,
            'error': None,
            # Inicializar todos os campos para evitar KeyError
            'processing_time': 0.0,
            'num_detections': 0,
            'detection_confidences': [],
            'avg_detection_confidence': 0.0,
            'ocr_texts': [],
            'ocr_confidences': [],
            'num_dates_found': 0,
            'dates': [],
            'predicted_text': '',
            'predicted_date': '',
            'final_confidence': 0.0,
            'exact_match': False,
            'partial_match': False,
            'similarity': 0.0,
            'cer': 1.0,
            'wer': 1.0
        }
        
        try:
            # Carregar imagem
            image = cv2.imread(str(img_path))
            
            if image is None:
                result['error'] = "Erro ao carregar imagem"
                return result
            
            # Verificar se deve salvar debug desta imagem
            save_debug = self.debug_images_count < self.max_debug_images
            
            # Processar com pipeline (com debug se necessário)
            start_time = time.time()
            if save_debug:
                pipeline_result = self._process_with_debug(image, img_name)
                self.debug_images_count += 1
                logger.info(f"🖼️  Imagens de debug salvas para: {img_name} ({self.debug_images_count}/{self.max_debug_images})")
            else:
                pipeline_result = self.pipeline.process(image)
            processing_time = time.time() - start_time
            
            result['processing_time'] = float(processing_time)
            result['success'] = True
            
            # Extrair informações das etapas
            # 1. Detecção
            detections = pipeline_result.get('detections', [])
            result['num_detections'] = int(len(detections))
            result['detection_confidences'] = [float(d.get('confidence', 0)) for d in detections]
            result['avg_detection_confidence'] = float(np.mean(result['detection_confidences']) if result['detection_confidences'] else 0)
            
            # 2. OCR
            ocr_results = pipeline_result.get('ocr_results', [])
            result['ocr_texts'] = [str(ocr.get('text', '')) for ocr in ocr_results]
            result['ocr_confidences'] = [float(ocr.get('confidence', 0)) for ocr in ocr_results]
            
            # 3. Datas parseadas
            dates = pipeline_result.get('dates', [])
            result['num_dates_found'] = int(len(dates))
            result['dates'] = [str(d.get('date_str', '')) for d in dates]
            
            # 4. Melhor resultado (best_date)
            best_date = pipeline_result.get('best_date')
            if best_date:
                predicted_text = str(best_date.get('text', ''))
                predicted_date = str(best_date.get('date_str', ''))
                final_confidence = float(best_date.get('combined_confidence', 0))
            else:
                predicted_text = ''
                predicted_date = ''
                final_confidence = 0.0
            
            result['predicted_text'] = predicted_text
            result['predicted_date'] = predicted_date
            result['final_confidence'] = final_confidence
            
            # Calcular métricas de comparação (usar predicted_date ao invés de predicted_text)
            metrics = self._calculate_comparison_metrics(
                predicted_date,
                ground_truth
            )
            result.update(metrics)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Erro ao processar {img_name}: {e}")
        
        return result
    
    def _process_with_debug(
        self,
        image: np.ndarray,
        img_name: str
    ) -> Dict[str, Any]:
        """
        Processa uma imagem salvando imagens de debug intermediárias.
        
        Salva:
        - input.jpg: Imagem original
        - crop.jpg: Região detectada pelo YOLO
        - preprocessamento.jpg: Após pré-processamento
        - ocr.jpg: Resultado do OCR com anotações
        
        Args:
            image: Imagem numpy array (BGR)
            img_name: Nome da imagem
            
        Returns:
            Resultado do processamento (igual ao pipeline.process())
        """
        import time

        # Criar diretório de debug para esta imagem
        safe_name = img_name.replace('.', '_').replace('/', '_')
        img_debug_dir = self.debug_output_dir / safe_name
        img_debug_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Salvando imagens de debug em: {img_debug_dir}")
        
        # 1. Salvar input original
        input_path = img_debug_dir / "input.jpg"
        cv2.imwrite(str(input_path), image)
        logger.debug(f"   ✓ Salvo: input.jpg")
        
        # Executar detecção YOLO
        detections = self.pipeline._detect_regions(image)
        
        if not detections:
            logger.warning("⚠️  Nenhuma região detectada - não há crop para salvar")
            return {
                'success': False,
                'detections': [],
                'ocr_results': [],
                'dates': [],
                'best_date': None,
                'processing_time': 0,
                'error': 'No detections'
            }
        
        # Processar primeira detecção (salvando debug)
        detection = detections[0]
        
        # 2. Extrair e salvar crop
        crop = self.pipeline._extract_crop(image, detection)
        crop_path = img_debug_dir / "crop.jpg"
        cv2.imwrite(str(crop_path), crop)
        logger.debug(f"   ✓ Salvo: crop.jpg")
        
        # 3. Pré-processar e salvar
        if self.pipeline.preprocessor:
            crop_processed = self.pipeline.preprocessor.process(crop)
            preprocess_path = img_debug_dir / "preprocessamento.jpg"
            cv2.imwrite(str(preprocess_path), crop_processed)
            logger.debug(f"   ✓ Salvo: preprocessamento.jpg")
        else:
            crop_processed = crop
            # Salvar mesmo sem pré-processamento
            preprocess_path = img_debug_dir / "preprocessamento.jpg"
            cv2.imwrite(str(preprocess_path), crop)
            logger.debug(f"   ✓ Salvo: preprocessamento.jpg (sem pré-processamento)")
        
        # 4. Executar OCR
        text, confidence = self.pipeline.ocr_engine.extract_text(crop_processed)
        
        # Criar visualização do OCR
        ocr_viz = crop_processed.copy()
        
        # Adicionar texto na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Background para o texto
        text_label = f"OCR: '{text}'"
        conf_label = f"Conf: {confidence:.2%}"
        
        # Posição do texto
        y_offset = 30
        
        # Desenhar fundo branco para melhor legibilidade
        (text_w, text_h), _ = cv2.getTextSize(text_label, font, font_scale, thickness)
        cv2.rectangle(ocr_viz, (5, 5), (text_w + 15, y_offset + 10), (255, 255, 255), -1)
        
        # Desenhar texto
        cv2.putText(ocr_viz, text_label, (10, y_offset), font, font_scale, (0, 0, 255), thickness)
        
        # Adicionar confiança
        y_offset += 35
        (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale, thickness)
        cv2.rectangle(ocr_viz, (5, y_offset - 25), (conf_w + 15, y_offset + 10), (255, 255, 255), -1)
        cv2.putText(ocr_viz, conf_label, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        ocr_path = img_debug_dir / "ocr.jpg"
        cv2.imwrite(str(ocr_path), ocr_viz)
        logger.debug(f"   ✓ Salvo: ocr.jpg")
        
        # Agora executar o pipeline completo normalmente para obter todos os resultados
        pipeline_result = self.pipeline.process(image, image_name=img_name)
        
        logger.info(f"✅ Debug completo salvo em: {img_debug_dir}")
        
        return pipeline_result
    
    def _normalize_date_format(self, date_str: str) -> str:
        """
        Normaliza formato de data para YYYY-MM-DD.
        
        Aceita:
        - DD/MM/YYYY
        - YYYY-MM-DD
        - DD-MM-YYYY
        
        Returns:
            Data no formato YYYY-MM-DD ou string vazia se inválida
        """
        from datetime import datetime
        
        if not date_str:
            return ''
        
        # Já está no formato correto
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str
        
        # Tentar parsear outros formatos
        formats = [
            '%d/%m/%Y',  # DD/MM/YYYY
            '%d-%m-%Y',  # DD-MM-YYYY
            '%Y/%m/%d',  # YYYY/MM/DD
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Não conseguiu parsear
        return date_str
    
    def _calculate_comparison_metrics(
        self,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Calcula métricas de comparação entre texto predito e ground truth.
        
        Args:
            predicted: Texto predito
            ground_truth: Texto esperado
            
        Returns:
            Dicionário com métricas
        """
        metrics = {}
        
        # Normalizar datas para o mesmo formato (YYYY-MM-DD)
        pred_normalized = self._normalize_date_format(predicted)
        gt_normalized = self._normalize_date_format(ground_truth)
        
        # Normalizar textos
        pred = pred_normalized.strip()
        gt = gt_normalized.strip()
        
        # 1. Exact Match
        metrics['exact_match'] = float(1.0 if pred == gt else 0.0)
        
        # 2. Similarity (SequenceMatcher) - usar datas normalizadas
        similarity = SequenceMatcher(None, pred, gt).ratio()
        metrics['similarity'] = float(similarity)
        
        # 3. Partial Match (>50% similaridade)
        metrics['partial_match'] = float(1.0 if similarity >= 0.5 else 0.0)
        
        # 4. Character Error Rate (CER)
        metrics['cer'] = float(self._calculate_cer(pred, gt))
        
        # 5. Word Error Rate (WER)
        metrics['wer'] = float(self._calculate_wer(pred, gt))
        
        return metrics
    
    def _calculate_cer(self, predicted: str, ground_truth: str) -> float:
        """Calcula Character Error Rate."""
        if not ground_truth:
            return 1.0 if predicted else 0.0
        
        # Levenshtein distance
        m, n = len(predicted), len(ground_truth)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if predicted[i-1] == ground_truth[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n] / n
    
    def _calculate_wer(self, predicted: str, ground_truth: str) -> float:
        """Calcula Word Error Rate."""
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        
        if not gt_words:
            return 1.0 if pred_words else 0.0
        
        # Levenshtein distance para palavras
        m, n = len(pred_words), len(gt_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == gt_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n] / n
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calcula métricas agregadas com estatísticas detalhadas."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {}
        
        # Métricas gerais
        total = int(len(df))
        successful = int(df['success'].sum())
        
        summary['general'] = {
            'total_images': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': float((successful / total * 100) if total > 0 else 0)
        }
        
        # ========================================
        # 📊 ESTATÍSTICAS DETALHADAS POR ETAPA
        # ========================================
        
        # 1. ANÁLISE DE DETECÇÃO (YOLO)
        no_detections = int((df['num_detections'] == 0).sum()) if 'num_detections' in df.columns else 0
        single_detection = int((df['num_detections'] == 1).sum()) if 'num_detections' in df.columns else 0
        multiple_detections = int((df['num_detections'] > 1).sum()) if 'num_detections' in df.columns else 0
        
        summary['detection_analysis'] = {
            'no_detections': no_detections,
            'no_detections_rate': float((no_detections / total * 100) if total > 0 else 0),
            'single_detection': single_detection,
            'single_detection_rate': float((single_detection / total * 100) if total > 0 else 0),
            'multiple_detections': multiple_detections,
            'multiple_detections_rate': float((multiple_detections / total * 100) if total > 0 else 0),
            'avg_detections_per_image': float(df['num_detections'].mean()) if 'num_detections' in df.columns else 0.0,
            'max_detections': int(df['num_detections'].max()) if 'num_detections' in df.columns else 0,
            'avg_detection_confidence': float(df['avg_detection_confidence'].mean()) if 'avg_detection_confidence' in df.columns else 0.0
        }
        
        # 2. ANÁLISE DE OCR
        # Identificar textos vazios/problemáticos
        empty_ocr = 0
        whitespace_only = 0
        short_text = 0  # Menos de 3 caracteres
        
        for result in self.results:
            ocr_texts = result.get('ocr_texts', [])
            predicted_text = result.get('predicted_text', '')
            
            # Verificar se OCR retornou vazio
            if not ocr_texts or all(not text or not text.strip() for text in ocr_texts):
                empty_ocr += 1
            elif predicted_text and predicted_text.strip() == '':
                whitespace_only += 1
            elif predicted_text and len(predicted_text.strip()) < 3:
                short_text += 1
        
        summary['ocr_analysis'] = {
            'empty_ocr': empty_ocr,
            'empty_ocr_rate': float((empty_ocr / total * 100) if total > 0 else 0),
            'whitespace_only': whitespace_only,
            'whitespace_only_rate': float((whitespace_only / total * 100) if total > 0 else 0),
            'short_text': short_text,
            'short_text_rate': float((short_text / total * 100) if total > 0 else 0),
            'avg_ocr_confidence': float(df['ocr_confidences'].apply(lambda x: np.mean(x) if x else 0).mean()) if 'ocr_confidences' in df.columns else 0.0,
            'low_confidence_ocr': int((df['ocr_confidences'].apply(lambda x: np.mean(x) if x else 0) < 0.5).sum()) if 'ocr_confidences' in df.columns else 0
        }
        
        # 3. ANÁLISE DE PARSING DE DATAS
        no_dates_found = int((df['num_dates_found'] == 0).sum()) if 'num_dates_found' in df.columns else 0
        single_date = int((df['num_dates_found'] == 1).sum()) if 'num_dates_found' in df.columns else 0
        multiple_dates = int((df['num_dates_found'] > 1).sum()) if 'num_dates_found' in df.columns else 0
        
        # Casos onde OCR retornou algo mas parser falhou
        ocr_success_parser_fail = 0
        ocr_fail_parser_success = 0  # Improvável mas possível
        
        # ✨ NOVA MÉTRICA: OCR com texto válido + Parser com data correta
        ocr_valid_parser_correct = 0
        ocr_valid_parser_correct_exact_match = 0
        
        for result in self.results:
            predicted_text = result.get('predicted_text', '').strip()
            has_ocr_text = predicted_text != ''
            has_dates = result.get('num_dates_found', 0) > 0
            exact_match = result.get('exact_match', 0.0) == 1.0
            
            if has_ocr_text and not has_dates:
                ocr_success_parser_fail += 1
            elif not has_ocr_text and has_dates:
                ocr_fail_parser_success += 1
            
            # ✨ NOVA: OCR válido (não vazio) + Parser encontrou data
            if has_ocr_text and has_dates:
                ocr_valid_parser_correct += 1
                # E se além de encontrar, acertou exatamente
                if exact_match:
                    ocr_valid_parser_correct_exact_match += 1
        
        summary['date_parsing_analysis'] = {
            'no_dates_found': no_dates_found,
            'no_dates_found_rate': float((no_dates_found / total * 100) if total > 0 else 0),
            'single_date': single_date,
            'single_date_rate': float((single_date / total * 100) if total > 0 else 0),
            'multiple_dates': multiple_dates,
            'multiple_dates_rate': float((multiple_dates / total * 100) if total > 0 else 0),
            'ocr_success_parser_fail': ocr_success_parser_fail,
            'ocr_success_parser_fail_rate': float((ocr_success_parser_fail / total * 100) if total > 0 else 0),
            'ocr_fail_parser_success': ocr_fail_parser_success,
            # ✨ NOVAS MÉTRICAS
            'ocr_valid_parser_correct': ocr_valid_parser_correct,
            'ocr_valid_parser_correct_rate': float((ocr_valid_parser_correct / total * 100) if total > 0 else 0),
            'ocr_valid_parser_exact_match': ocr_valid_parser_correct_exact_match,
            'ocr_valid_parser_exact_match_rate': float((ocr_valid_parser_correct_exact_match / total * 100) if total > 0 else 0),
            'avg_parse_confidence': float(df['final_confidence'].mean()) if 'final_confidence' in df.columns else 0.0
        }
        
        # 4. ANÁLISE DE CASCATA (Como erros se propagam)
        yolo_ok_ocr_fail = 0
        yolo_ok_ocr_ok_parser_fail = 0
        full_pipeline_success = 0
        
        for result in self.results:
            has_detection = result.get('num_detections', 0) > 0
            has_ocr_text = result.get('predicted_text', '').strip() != ''
            has_dates = result.get('num_dates_found', 0) > 0
            
            if has_detection and not has_ocr_text:
                yolo_ok_ocr_fail += 1
            elif has_detection and has_ocr_text and not has_dates:
                yolo_ok_ocr_ok_parser_fail += 1
            elif has_detection and has_ocr_text and has_dates:
                full_pipeline_success += 1
        
        summary['pipeline_cascade'] = {
            'yolo_ok_ocr_fail': yolo_ok_ocr_fail,
            'yolo_ok_ocr_fail_rate': float((yolo_ok_ocr_fail / total * 100) if total > 0 else 0),
            'yolo_ok_ocr_ok_parser_fail': yolo_ok_ocr_ok_parser_fail,
            'yolo_ok_ocr_ok_parser_fail_rate': float((yolo_ok_ocr_ok_parser_fail / total * 100) if total > 0 else 0),
            'full_pipeline_success': full_pipeline_success,
            'full_pipeline_success_rate': float((full_pipeline_success / total * 100) if total > 0 else 0)
        }
        
        # Filtrar apenas sucessos para métricas tradicionais
        df_success = df[df['success'] == True]
        
        if len(df_success) == 0:
            return summary
        
        # Métricas de detecção (tradicionais)
        if self.config['metrics']['detection']['enabled'] and 'num_detections' in df_success.columns:
            summary['detection'] = {
                'detection_rate': float((df_success['num_detections'] > 0).mean() * 100),
                'avg_detections_per_image': float(df_success['num_detections'].mean()),
                'avg_confidence': float(df_success['avg_detection_confidence'].mean()) if 'avg_detection_confidence' in df_success.columns else 0.0
            }
        
        # Métricas de OCR (tradicionais)
        if self.config['metrics']['ocr']['enabled'] and 'exact_match' in df_success.columns:
            summary['ocr'] = {
                'exact_match_rate': float(df_success['exact_match'].mean() * 100),
                'partial_match_rate': float(df_success['partial_match'].mean() * 100) if 'partial_match' in df_success.columns else 0.0,
                'avg_similarity': float(df_success['similarity'].mean()) if 'similarity' in df_success.columns else 0.0,
                'avg_cer': float(df_success['cer'].mean()) if 'cer' in df_success.columns else 1.0,
                'avg_wer': float(df_success['wer'].mean()) if 'wer' in df_success.columns else 1.0
            }
        
        # Métricas de parsing de data (tradicionais)
        if self.config['metrics']['date_parsing']['enabled'] and 'num_dates_found' in df_success.columns:
            summary['date_parsing'] = {
                'date_found_rate': float((df_success['num_dates_found'] > 0).mean() * 100),
                'avg_dates_per_image': float(df_success['num_dates_found'].mean())
            }
        
        # Métricas end-to-end (tradicionais)
        if self.config['metrics']['end_to_end']['enabled']:
            summary['end_to_end'] = {
                'pipeline_accuracy': float(df_success['exact_match'].mean() * 100) if 'exact_match' in df_success.columns else 0.0,
                'avg_processing_time': float(df_success['processing_time'].mean()) if 'processing_time' in df_success.columns else 0.0,
                'avg_final_confidence': float(df_success['final_confidence'].mean()) if 'final_confidence' in df_success.columns else 0.0
            }
        
        return summary
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analisa erros da pipeline."""
        df = pd.DataFrame(self.results)
        
        analysis = {
            'error_breakdown': {},
            'error_examples': []
        }
        
        # Classificar erros por etapa
        if self.config['error_analysis']['classify_by_stage']:
            # Erros de detecção
            if 'num_detections' in df.columns:
                no_detections = df[df['num_detections'] == 0]
                analysis['error_breakdown']['no_detection'] = int(len(no_detections))
            
            # Erros de OCR
            if 'num_detections' in df.columns and 'exact_match' in df.columns:
                has_detection_no_match = df[(df['num_detections'] > 0) & (df['exact_match'] == 0)]
                analysis['error_breakdown']['detection_but_wrong_ocr'] = int(len(has_detection_no_match))
            
            # Erros de parsing
            has_ocr_no_date = df[(df['num_detections'] > 0) & (df['num_dates_found'] == 0)]
            analysis['error_breakdown']['ocr_but_no_date_parsed'] = int(len(has_ocr_no_date))
        
        # Salvar exemplos de erros
        if self.config['error_analysis']['save_error_examples']:
            max_examples = self.config['error_analysis']['max_error_examples']
            
            errors = df[df['exact_match'] == 0].head(max_examples)
            
            for _, row in errors.iterrows():
                analysis['error_examples'].append({
                    'image_name': row['image_name'],
                    'ground_truth': row['ground_truth'],
                    'predicted_text': row.get('predicted_text', ''),
                    'cer': row.get('cer', 1.0),
                    'similarity': row.get('similarity', 0.0)
                })
        
        return analysis
    
    def _convert_to_native_types(self, obj):
        """
        Converte recursivamente tipos numpy/pandas para tipos nativos Python.
        
        Args:
            obj: Objeto a converter
            
        Returns:
            Objeto com tipos nativos Python
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.notna(obj) and pd.api.types.is_integer_dtype(type(obj)):
            return int(obj)
        elif pd.notna(obj) and pd.api.types.is_float_dtype(type(obj)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def save_results(self, output_dir: Path):
        """Salva resultados da avaliação."""
        logger.info(f"\n💾 Salvando resultados em: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. CSV detalhado
        if self.config['output']['save_results']['detailed_csv']:
            csv_path = output_dir / 'evaluation_results.csv'
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"   ✅ CSV detalhado: {csv_path}")
        
        # 2. JSON completo (converter tipos antes de salvar)
        if self.config['output']['save_results']['full_json']:
            json_path = output_dir / 'evaluation_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json_data = {
                    'results': self._convert_to_native_types(self.results),
                    'metrics_summary': self._convert_to_native_types(self.metrics_summary),
                    'error_analysis': self._convert_to_native_types(self.error_analysis)
                }
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ JSON completo: {json_path}")
        
        # 3. Métricas agregadas (converter tipos antes de salvar)
        if self.config['output']['save_results']['metrics_summary']:
            metrics_path = output_dir / 'metrics_summary.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                metrics_data = self._convert_to_native_types(self.metrics_summary)
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Métricas agregadas: {metrics_path}")
        
        # 4. Relatório markdown
        if self.config['output']['save_results']['markdown_report']:
            report_path = output_dir / 'evaluation_report.md'
            self._generate_markdown_report(report_path)
            logger.info(f"   ✅ Relatório markdown: {report_path}")
    
    def save_visualizations(self, output_dir: Path):
        """Gera e salva visualizações."""
        if not self.config['output']['save_visualizations']['metrics_plots']:
            return
        
        logger.info(f"\n📊 Gerando visualizações...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        df_success = df[df['success'] == True]
        
        if len(df_success) == 0:
            logger.warning("⚠️ Nenhum resultado bem-sucedido para visualizar")
            return
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        # 1. Métricas principais
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Exact Match Rate
        ax = axes[0, 0]
        exact_rate = df_success['exact_match'].mean() * 100
        partial_rate = df_success['partial_match'].mean() * 100
        categories = ['Exact Match', 'Partial Match', 'No Match']
        values = [exact_rate, partial_rate - exact_rate, 100 - partial_rate]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(categories, values, color=colors, edgecolor='black')
        ax.set_ylabel('Porcentagem (%)')
        ax.set_title('Taxa de Match', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        for i, v in enumerate(values):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # CER Distribution
        ax = axes[0, 1]
        df_success['cer'].hist(bins=20, ax=ax, edgecolor='black', color='skyblue')
        ax.axvline(df_success['cer'].mean(), color='r', linestyle='--',
                   label=f'Média: {df_success["cer"].mean():.3f}')
        ax.set_xlabel('Character Error Rate')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de CER', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Processing Time
        ax = axes[0, 2]
        (df_success['processing_time'] * 1000).hist(bins=20, ax=ax, edgecolor='black', color='coral')
        ax.axvline(df_success['processing_time'].mean() * 1000, color='r', linestyle='--',
                   label=f'Média: {df_success["processing_time"].mean()*1000:.1f}ms')
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('Frequência')
        ax.set_title('Tempo de Processamento', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Detection Confidence
        ax = axes[1, 0]
        df_success['avg_detection_confidence'].hist(bins=20, ax=ax, edgecolor='black', color='mediumseagreen')
        ax.set_xlabel('Confiança')
        ax.set_ylabel('Frequência')
        ax.set_title('Confiança das Detecções', fontsize=14, fontweight='bold')
        
        # Similarity Distribution
        ax = axes[1, 1]
        df_success['similarity'].hist(bins=20, ax=ax, edgecolor='black', color='mediumpurple')
        ax.set_xlabel('Similaridade')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Similaridade', fontsize=14, fontweight='bold')
        
        # Error Breakdown
        ax = axes[1, 2]
        if self.error_analysis.get('error_breakdown'):
            breakdown = self.error_analysis['error_breakdown']
            labels = list(breakdown.keys())
            values = list(breakdown.values())
            ax.bar(range(len(labels)), values, color='indianred', edgecolor='black')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Quantidade')
            ax.set_title('Breakdown de Erros', fontsize=14, fontweight='bold')
            for i, v in enumerate(values):
                ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plot_path = viz_dir / 'metrics_overview.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ Gráficos salvos: {viz_dir}")
    
    def print_detailed_summary(self):
        """Imprime resumo detalhado das métricas no console."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 RESUMO DETALHADO DA AVALIAÇÃO")
        logger.info("=" * 80)
        
        # Métricas gerais
        general = self.metrics_summary.get('general', {})
        logger.info(f"\n📈 MÉTRICAS GERAIS:")
        logger.info(f"   Total de imagens: {general.get('total_images', 0)}")
        logger.info(f"   ✅ Sucessos: {general.get('successful', 0)}")
        logger.info(f"   ❌ Falhas: {general.get('failed', 0)}")
        logger.info(f"   Taxa de sucesso: {general.get('success_rate', 0):.2f}%")
        
        # ANÁLISE DE DETECÇÃO
        if 'detection_analysis' in self.metrics_summary:
            det_analysis = self.metrics_summary['detection_analysis']
            logger.info(f"\n🎯 ANÁLISE DE DETECÇÃO (YOLO):")
            logger.info(f"   ❌ Nenhuma detecção: {det_analysis.get('no_detections', 0)} ({det_analysis.get('no_detections_rate', 0):.2f}%)")
            logger.info(f"   ✅ Uma detecção: {det_analysis.get('single_detection', 0)} ({det_analysis.get('single_detection_rate', 0):.2f}%)")
            logger.info(f"   🔢 Múltiplas detecções: {det_analysis.get('multiple_detections', 0)} ({det_analysis.get('multiple_detections_rate', 0):.2f}%)")
            logger.info(f"   📊 Média detecções/img: {det_analysis.get('avg_detections_per_image', 0):.2f}")
            logger.info(f"   🎯 Confiança média: {det_analysis.get('avg_detection_confidence', 0):.3f}")
        
        # ANÁLISE DE OCR
        if 'ocr_analysis' in self.metrics_summary:
            ocr_analysis = self.metrics_summary['ocr_analysis']
            logger.info(f"\n🔤 ANÁLISE DE OCR:")
            logger.info(f"   ❌ OCR vazio: {ocr_analysis.get('empty_ocr', 0)} ({ocr_analysis.get('empty_ocr_rate', 0):.2f}%)")
            logger.info(f"   ⚠️  Apenas espaços: {ocr_analysis.get('whitespace_only', 0)} ({ocr_analysis.get('whitespace_only_rate', 0):.2f}%)")
            logger.info(f"   ⚠️  Texto curto (<3 chars): {ocr_analysis.get('short_text', 0)} ({ocr_analysis.get('short_text_rate', 0):.2f}%)")
            logger.info(f"   🎯 Confiança média: {ocr_analysis.get('avg_ocr_confidence', 0):.3f}")
            logger.info(f"   ⚠️  Baixa confiança (<0.5): {ocr_analysis.get('low_confidence_ocr', 0)} imagens")
        
        # ANÁLISE DE PARSING
        if 'date_parsing_analysis' in self.metrics_summary:
            parse_analysis = self.metrics_summary['date_parsing_analysis']
            logger.info(f"\n📅 ANÁLISE DE PARSING DE DATAS:")
            logger.info(f"   ❌ Nenhuma data: {parse_analysis.get('no_dates_found', 0)} ({parse_analysis.get('no_dates_found_rate', 0):.2f}%)")
            logger.info(f"   ✅ Uma data: {parse_analysis.get('single_date', 0)} ({parse_analysis.get('single_date_rate', 0):.2f}%)")
            logger.info(f"   🔢 Múltiplas datas: {parse_analysis.get('multiple_dates', 0)} ({parse_analysis.get('multiple_dates_rate', 0):.2f}%)")
            logger.info(f"   ⚠️  OCR OK mas parser falhou: {parse_analysis.get('ocr_success_parser_fail', 0)} ({parse_analysis.get('ocr_success_parser_fail_rate', 0):.2f}%)")
            logger.info(f"   🎯 Parser acertou sem OCR: {parse_analysis.get('ocr_fail_parser_success', 0)} imagens")
            logger.info(f"   ✨ OCR válido + Parser encontrou data: {parse_analysis.get('ocr_valid_parser_correct', 0)} ({parse_analysis.get('ocr_valid_parser_correct_rate', 0):.2f}%)")
            logger.info(f"   🏆 OCR válido + Parser acertou exato: {parse_analysis.get('ocr_valid_parser_exact_match', 0)} ({parse_analysis.get('ocr_valid_parser_exact_match_rate', 0):.2f}%)")
            logger.info(f"   🎯 Confiança média: {parse_analysis.get('avg_parse_confidence', 0):.3f}")
        
        # ANÁLISE DE CASCATA
        if 'pipeline_cascade' in self.metrics_summary:
            cascade = self.metrics_summary['pipeline_cascade']
            logger.info(f"\n🔄 ANÁLISE DE CASCATA (Propagação de Erros):")
            logger.info(f"   YOLO ✅ → OCR ❌: {cascade.get('yolo_ok_ocr_fail', 0)} ({cascade.get('yolo_ok_ocr_fail_rate', 0):.2f}%)")
            logger.info(f"   YOLO ✅ → OCR ✅ → Parser ❌: {cascade.get('yolo_ok_ocr_ok_parser_fail', 0)} ({cascade.get('yolo_ok_ocr_ok_parser_fail_rate', 0):.2f}%)")
            logger.info(f"   🎯 Pipeline completa OK: {cascade.get('full_pipeline_success', 0)} ({cascade.get('full_pipeline_success_rate', 0):.2f}%)")
        
        # MÉTRICAS TRADICIONAIS (APENAS SUCESSOS)
        if 'ocr' in self.metrics_summary:
            ocr = self.metrics_summary['ocr']
            logger.info(f"\n📊 MÉTRICAS DE ACURÁCIA (Apenas Sucessos):")
            logger.info(f"   Exact Match: {ocr.get('exact_match_rate', 0):.2f}%")
            logger.info(f"   Partial Match: {ocr.get('partial_match_rate', 0):.2f}%")
            logger.info(f"   Similaridade média: {ocr.get('avg_similarity', 0):.3f}")
            logger.info(f"   CER médio: {ocr.get('avg_cer', 0):.3f}")
            logger.info(f"   WER médio: {ocr.get('avg_wer', 0):.3f}")
        
        if 'end_to_end' in self.metrics_summary:
            e2e = self.metrics_summary['end_to_end']
            logger.info(f"\n⏱️  PERFORMANCE:")
            logger.info(f"   Tempo médio: {e2e.get('avg_processing_time', 0):.3f}s")
            logger.info(f"   Confiança média: {e2e.get('avg_final_confidence', 0):.3f}")
        
        logger.info("\n" + "=" * 80 + "\n")
    
    def _generate_markdown_report(self, output_path: Path):
        """Gera relatório markdown com estatísticas detalhadas."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Relatório de Avaliação da Pipeline\n\n")
            f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Métricas gerais
            f.write("## 📈 Métricas Gerais\n\n")
            general = self.metrics_summary.get('general', {})
            f.write(f"- **Total de Imagens:** {general.get('total_images', 0)}\n")
            f.write(f"- **Processadas com Sucesso:** {general.get('successful', 0)}\n")
            f.write(f"- **Falhas:** {general.get('failed', 0)}\n")
            f.write(f"- **Taxa de Sucesso:** {general.get('success_rate', 0):.2f}%\n\n")
            
            # ========================================
            # ESTATÍSTICAS DETALHADAS POR ETAPA
            # ========================================
            
            # 1. ANÁLISE DE DETECÇÃO (YOLO)
            if 'detection_analysis' in self.metrics_summary:
                f.write("## 🎯 Análise Detalhada de Detecção (YOLO)\n\n")
                det_analysis = self.metrics_summary['detection_analysis']
                f.write("### Distribuição de Detecções\n\n")
                f.write(f"- **❌ Nenhuma Detecção:** {det_analysis.get('no_detections', 0)} imagens ({det_analysis.get('no_detections_rate', 0):.2f}%)\n")
                f.write(f"- **✅ Uma Detecção:** {det_analysis.get('single_detection', 0)} imagens ({det_analysis.get('single_detection_rate', 0):.2f}%)\n")
                f.write(f"- **🔢 Múltiplas Detecções:** {det_analysis.get('multiple_detections', 0)} imagens ({det_analysis.get('multiple_detections_rate', 0):.2f}%)\n\n")
                f.write("### Estatísticas\n\n")
                f.write(f"- **Média de Detecções/Imagem:** {det_analysis.get('avg_detections_per_image', 0):.2f}\n")
                f.write(f"- **Máximo de Detecções:** {det_analysis.get('max_detections', 0)}\n")
                f.write(f"- **Confiança Média:** {det_analysis.get('avg_detection_confidence', 0):.3f}\n\n")
            
            # 2. ANÁLISE DE OCR
            if 'ocr_analysis' in self.metrics_summary:
                f.write("## 🔤 Análise Detalhada de OCR\n\n")
                ocr_analysis = self.metrics_summary['ocr_analysis']
                f.write("### Problemas de OCR\n\n")
                f.write(f"- **❌ OCR Vazio:** {ocr_analysis.get('empty_ocr', 0)} imagens ({ocr_analysis.get('empty_ocr_rate', 0):.2f}%)\n")
                f.write(f"  - *Nenhum texto foi detectado pelo OCR*\n")
                f.write(f"- **⚠️ Apenas Espaços:** {ocr_analysis.get('whitespace_only', 0)} imagens ({ocr_analysis.get('whitespace_only_rate', 0):.2f}%)\n")
                f.write(f"  - *OCR retornou apenas espaços em branco*\n")
                f.write(f"- **⚠️ Texto Curto (<3 chars):** {ocr_analysis.get('short_text', 0)} imagens ({ocr_analysis.get('short_text_rate', 0):.2f}%)\n")
                f.write(f"  - *Texto muito curto, provavelmente erro*\n\n")
                f.write("### Estatísticas de Confiança\n\n")
                f.write(f"- **Confiança Média do OCR:** {ocr_analysis.get('avg_ocr_confidence', 0):.3f}\n")
                f.write(f"- **OCR com Baixa Confiança (<0.5):** {ocr_analysis.get('low_confidence_ocr', 0)} imagens\n\n")
            
            # 3. ANÁLISE DE PARSING
            if 'date_parsing_analysis' in self.metrics_summary:
                f.write("## 📅 Análise Detalhada de Parsing de Datas\n\n")
                parse_analysis = self.metrics_summary['date_parsing_analysis']
                f.write("### Distribuição de Datas Encontradas\n\n")
                f.write(f"- **❌ Nenhuma Data:** {parse_analysis.get('no_dates_found', 0)} imagens ({parse_analysis.get('no_dates_found_rate', 0):.2f}%)\n")
                f.write(f"- **✅ Uma Data:** {parse_analysis.get('single_date', 0)} imagens ({parse_analysis.get('single_date_rate', 0):.2f}%)\n")
                f.write(f"- **🔢 Múltiplas Datas:** {parse_analysis.get('multiple_dates', 0)} imagens ({parse_analysis.get('multiple_dates_rate', 0):.2f}%)\n\n")
                f.write("### Análise de Erros de Parsing\n\n")
                f.write(f"- **⚠️ OCR OK mas Parser Falhou:** {parse_analysis.get('ocr_success_parser_fail', 0)} imagens ({parse_analysis.get('ocr_success_parser_fail_rate', 0):.2f}%)\n")
                f.write(f"  - *OCR extraiu texto mas o parser não encontrou data válida*\n")
                f.write(f"- **🎯 Parser Acertou sem OCR:** {parse_analysis.get('ocr_fail_parser_success', 0)} imagens\n")
                f.write(f"  - *Parser encontrou data mesmo com OCR vazio (raro)*\n\n")
                f.write("### ✨ Análise de Sucesso (OCR + Parser)\n\n")
                f.write(f"- **✨ OCR Válido + Parser Encontrou Data:** {parse_analysis.get('ocr_valid_parser_correct', 0)} imagens ({parse_analysis.get('ocr_valid_parser_correct_rate', 0):.2f}%)\n")
                f.write(f"  - *OCR retornou texto não-vazio E parser conseguiu extrair uma data*\n")
                f.write(f"- **🏆 OCR Válido + Parser Acertou Exato:** {parse_analysis.get('ocr_valid_parser_exact_match', 0)} imagens ({parse_analysis.get('ocr_valid_parser_exact_match_rate', 0):.2f}%)\n")
                f.write(f"  - *OCR retornou texto E parser extraiu a data CORRETA (exact match)*\n\n")
                f.write("### Estatísticas\n\n")
                f.write(f"- **Confiança Média do Parser:** {parse_analysis.get('avg_parse_confidence', 0):.3f}\n\n")
            
            # 4. ANÁLISE DE CASCATA
            if 'pipeline_cascade' in self.metrics_summary:
                f.write("## 🔄 Análise de Cascata (Propagação de Erros)\n\n")
                cascade = self.metrics_summary['pipeline_cascade']
                f.write("### Pontos de Falha da Pipeline\n\n")
                f.write(f"- **YOLO ✅ → OCR ❌:** {cascade.get('yolo_ok_ocr_fail', 0)} ({cascade.get('yolo_ok_ocr_fail_rate', 0):.2f}%)\n")
                f.write(f"  - *YOLO detectou mas OCR falhou*\n")
                f.write(f"- **YOLO ✅ → OCR ✅ → Parser ❌:** {cascade.get('yolo_ok_ocr_ok_parser_fail', 0)} ({cascade.get('yolo_ok_ocr_ok_parser_fail_rate', 0):.2f}%)\n")
                f.write(f"  - *YOLO e OCR OK mas parser não encontrou data*\n")
                f.write(f"- **🎯 Pipeline Completa com Sucesso:** {cascade.get('full_pipeline_success', 0)} imagens ({cascade.get('full_pipeline_success_rate', 0):.2f}%)\n")
                f.write(f"  - *Todas as etapas funcionaram corretamente*\n\n")
            
            # ========================================
            # MÉTRICAS TRADICIONAIS
            # ========================================
            
            # Métricas de detecção (tradicionais)
            if 'detection' in self.metrics_summary:
                f.write("## 📊 Métricas de Detecção (Apenas Sucessos)\n\n")
                detection = self.metrics_summary['detection']
                f.write(f"- **Taxa de Detecção:** {detection.get('detection_rate', 0):.2f}%\n")
                f.write(f"- **Média de Detecções/Imagem:** {detection.get('avg_detections_per_image', 0):.2f}\n")
                f.write(f"- **Confiança Média:** {detection.get('avg_confidence', 0):.3f}\n\n")
            
            # Métricas de OCR (tradicionais)
            if 'ocr' in self.metrics_summary:
                f.write("## � Métricas de OCR (Apenas Sucessos)\n\n")
                ocr = self.metrics_summary['ocr']
                f.write(f"- **Exact Match:** {ocr.get('exact_match_rate', 0):.2f}%\n")
                f.write(f"- **Partial Match:** {ocr.get('partial_match_rate', 0):.2f}%\n")
                f.write(f"- **Similaridade Média:** {ocr.get('avg_similarity', 0):.3f}\n")
                f.write(f"- **CER Médio:** {ocr.get('avg_cer', 0):.3f}\n")
                f.write(f"- **WER Médio:** {ocr.get('avg_wer', 0):.3f}\n\n")
            
            # Métricas de parsing (tradicionais)
            if 'date_parsing' in self.metrics_summary:
                f.write("## � Métricas de Parsing (Apenas Sucessos)\n\n")
                parsing = self.metrics_summary['date_parsing']
                f.write(f"- **Taxa de Datas Encontradas:** {parsing.get('date_found_rate', 0):.2f}%\n")
                f.write(f"- **Média de Datas/Imagem:** {parsing.get('avg_dates_per_image', 0):.2f}\n\n")
            
            # Métricas end-to-end (tradicionais)
            if 'end_to_end' in self.metrics_summary:
                f.write("## 🎯 Métricas End-to-End\n\n")
                e2e = self.metrics_summary['end_to_end']
                f.write(f"- **Acurácia da Pipeline:** {e2e.get('pipeline_accuracy', 0):.2f}%\n")
                f.write(f"- **Tempo Médio de Processamento:** {e2e.get('avg_processing_time', 0):.3f}s\n")
                f.write(f"- **Confiança Final Média:** {e2e.get('avg_final_confidence', 0):.3f}\n\n")
            
            # Análise de erros
            if self.error_analysis:
                f.write("## 🔍 Análise de Erros\n\n")
                
                if 'error_breakdown' in self.error_analysis:
                    f.write("### Breakdown por Etapa\n\n")
                    for error_type, count in self.error_analysis['error_breakdown'].items():
                        f.write(f"- **{error_type}:** {count}\n")
                    f.write("\n")
                
                if 'error_examples' in self.error_analysis and self.error_analysis['error_examples']:
                    f.write("### Exemplos de Erros\n\n")
                    for i, example in enumerate(self.error_analysis['error_examples'][:5], 1):
                        f.write(f"**{i}. {example['image_name']}**\n\n")
                        f.write(f"- **Ground Truth:** `{example['ground_truth']}`\n")
                        f.write(f"- **Predição:** `{example['predicted_text']}`\n")
                        f.write(f"- **CER:** {example['cer']:.3f}\n")
                        f.write(f"- **Similaridade:** {example['similarity']:.3f}\n\n")


def load_config(config_path: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Carrega configuração e sobrescreve com argumentos CLI.
    
    Args:
        config_path: Caminho do arquivo de configuração
        cli_args: Argumentos da linha de comando
        
    Returns:
        Configuração mesclada
    """
    # Carregar YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Sobrescrever com CLI args
    if cli_args.num_images is not None:
        config['dataset']['num_images'] = cli_args.num_images
    
    if cli_args.selection_mode is not None:
        config['dataset']['selection_mode'] = cli_args.selection_mode
    
    if cli_args.output is not None:
        config['output']['base_dir'] = cli_args.output
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação completa da pipeline end-to-end"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline/pipeline_evaluation.yaml',
        help='Arquivo de configuração da avaliação'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        help='Número de imagens para avaliar (sobrescreve config)'
    )
    parser.add_argument(
        '--selection-mode',
        type=str,
        choices=['all', 'first', 'random'],
        help='Modo de seleção de imagens (sobrescreve config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Diretório de saída (sobrescreve config)'
    )
    
    args = parser.parse_args()
    
    # Carregar configuração
    config = load_config(args.config, args)
    
    # Criar diretório de output
    output_dir = Path(config['output']['base_dir'])
    if config['output'].get('use_timestamp', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    if config['logging']['save_to_file']:
        log_file = output_dir / 'evaluation.log'
        logger.add(log_file, level=config['logging']['level'])
    
    # Criar avaliador
    evaluator = PipelineEvaluator(config)
    
    # Configurar diretório de debug
    evaluator.debug_output_dir = output_dir / "debug_images"
    evaluator.debug_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Diretório de debug: {evaluator.debug_output_dir}")
    
    # Inicializar pipeline
    evaluator.initialize_pipeline()
    
    # Executar avaliação
    start_time = time.time()
    results = evaluator.evaluate()
    total_time = time.time() - start_time
    
    # Salvar resultados
    evaluator.save_results(output_dir)
    
    # Salvar visualizações
    evaluator.save_visualizations(output_dir)
    
    # Imprimir resumo detalhado no console
    evaluator.print_detailed_summary()
    
    # Relatório final
    logger.info("\n" + "=" * 80)
    logger.info("✅ AVALIAÇÃO CONCLUÍDA")
    logger.info("=" * 80)
    logger.info(f"⏱️  Tempo total: {total_time:.2f}s")
    logger.info(f"📁 Resultados salvos em: {output_dir}")
    logger.info(f"� Relatório detalhado: {output_dir / 'evaluation_report.md'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
