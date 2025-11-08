"""
🚀 Script de Teste: Full Pipeline
Testa o pipeline completo YOLO → OCR → Parse com outputs detalhados de cada etapa.

Funcionalidades:
- Visualização da imagem de entrada
- Detecções YOLO com bounding boxes e segmentação
- Máscaras de segmentação (quando disponível)
- Crops originais (com fundo branco se usar segmentação)
- Crops pré-processados
- Resultados OCR em cada crop
- Parsing de datas
- Resultado final com melhor data destacada

Outputs salvos (em ordem):
- *_1_input.jpg: Imagem original
- *_2_yolo_detection.jpg: Detecções YOLO (bbox + overlay de máscara)
- *_3a_mask_*.jpg: Máscaras de segmentação (se disponível)
- *_3b_crop_*_original.jpg: Crops originais (com fundo branco se segmentado)
- *_3c_crop_*_processed.jpg: Crops após pré-processamento
- *_3d_crop_*_ocr.jpg: Visualização dos resultados OCR
- *_4_result.jpg: Resultado final com datas anotadas

Uso:
    # Processar uma imagem (com outputs detalhados)
    python scripts/pipeline/test_full_pipeline.py --image data/test.jpg
    
    # Processar diretório
    python scripts/pipeline/test_full_pipeline.py --image-dir data/test_images/
    
    # Com config customizada
    python scripts/pipeline/test_full_pipeline.py --image data/test.jpg --config config/pipeline/full_pipeline.yaml
    
    # Especificar diretório de output
    python scripts/pipeline/test_full_pipeline.py --image data/test.jpg --output outputs/meus_testes
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.full_pipeline import FullPipeline, load_pipeline_config


def test_single_image(pipeline: FullPipeline, image_path: str, output_dir: str = "outputs/pipeline_steps"):
    """
    Testa pipeline em uma única imagem com output de cada etapa.
    
    Args:
        pipeline: Instância do pipeline
        image_path: Caminho para imagem
        output_dir: Diretório para salvar outputs das etapas
    """
    logger.info(f"📸 Testando imagem: {image_path}")
    
    # Criar diretório de output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"❌ Erro ao carregar imagem: {image_path}")
        return
    
    # ========================================
    # ETAPA 1: ENTRADA
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("📥 ETAPA 1: IMAGEM DE ENTRADA")
    logger.info("="*60)
    logger.info(f"   Dimensões: {image.shape[1]}x{image.shape[0]} pixels")
    logger.info(f"   Canais: {image.shape[2]}")
    logger.info(f"   Tipo: {image.dtype}")
    
    # Salvar imagem original
    input_path = output_path / f"{image_name}_1_input.jpg"
    cv2.imwrite(str(input_path), image)
    logger.info(f"   ✅ Salvo: {input_path}")
    
    # ========================================
    # ETAPA 2: DETECÇÃO YOLO
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("🎯 ETAPA 2: DETECÇÃO YOLO")
    logger.info("="*60)
    
    # Executar detecção
    detections = pipeline._detect_regions(image)
    logger.info(f"   Detecções encontradas: {len(detections)}")
    
    # Verificar se tem segmentação
    has_segmentation = any('mask' in det and det['mask'] is not None for det in detections)
    logger.info(f"   Modo de detecção: {'Segmentação' if has_segmentation else 'BBox apenas'}")
    
    # Visualizar detecções
    yolo_viz = image.copy()
    for i, det in enumerate(detections):
        bbox = det['bbox']
        # Extrair coordenadas do dicionário estruturado
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])
        
        # Se tiver máscara, desenhar contorno da segmentação também
        if 'mask' in det and det['mask'] is not None:
            mask = det['mask']
            h, w = image.shape[:2]
            
            # Redimensionar máscara se necessário
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            # Criar overlay de máscara (semi-transparente)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(image)
            colored_mask[:, :] = (0, 255, 0)  # Verde
            
            # Aplicar máscara com transparência
            alpha = 0.3
            yolo_viz = np.where(
                mask_binary[:, :, np.newaxis] > 0,
                (yolo_viz * (1 - alpha) + colored_mask * alpha).astype(np.uint8),
                yolo_viz
            )
            
            # Desenhar contorno da máscara (POLÍGONO)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(yolo_viz, contours, -1, (0, 255, 0), 3)
            
            # Se tiver polígono no detection, desenhar os pontos
            if 'polygon' in det and det['polygon']:
                polygon_points = np.array(det['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(yolo_viz, [polygon_points], True, (255, 0, 255), 2)  # Magenta para polígono original
        
        # Desenhar bbox
        cv2.rectangle(yolo_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label
        label = f"{det['class_name']} {det['confidence']:.2%}"
        if 'mask' in det and det['mask'] is not None:
            label += " [SEG]"
        cv2.putText(yolo_viz, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logger.info(f"   Detecção {i+1}:")
        logger.info(f"      Classe: {det['class_name']}")
        logger.info(f"      Confiança: {det['confidence']:.2%}")
        logger.info(f"      BBox: [{x1}, {y1}, {x2}, {y2}]")
        if 'mask' in det and det['mask'] is not None:
            logger.info(f"      Segmentação: ✓ (máscara {mask.shape})")
    
    # Salvar visualização YOLO
    yolo_path = output_path / f"{image_name}_2_yolo_detection.jpg"
    cv2.imwrite(str(yolo_path), yolo_viz)
    logger.info(f"   ✅ Salvo: {yolo_path}")
    
    # ========================================
    # ETAPA 3: PRÉ-PROCESSAMENTO E OCR
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("🔧 ETAPA 3: PRÉ-PROCESSAMENTO E OCR")
    logger.info("="*60)
    
    ocr_results = []
    for i, detection in enumerate(detections):
        logger.info(f"\n   Processando detecção {i+1}/{len(detections)}...")
        
        # Se tiver máscara, salvar visualização da segmentação
        if 'mask' in detection and detection['mask'] is not None:
            mask = detection['mask']
            h, w = image.shape[:2]
            
            # Redimensionar máscara
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            # Salvar máscara como imagem
            mask_vis = (mask_resized * 255).astype(np.uint8)
            mask_path = output_path / f"{image_name}_3a_mask_{i+1}.jpg"
            cv2.imwrite(str(mask_path), mask_vis)
            logger.info(f"      ✅ Máscara de segmentação: {mask_path}")
        
        # Extrair crop
        crop = pipeline._extract_crop(image, detection)
        logger.info(f"      Crop: {crop.shape[1]}x{crop.shape[0]} pixels")
        
        # Salvar crop original
        crop_orig_path = output_path / f"{image_name}_3b_crop_{i+1}_original.jpg"
        cv2.imwrite(str(crop_orig_path), crop)
        logger.info(f"      ✅ Crop original: {crop_orig_path}")
        
        # Pré-processar
        if pipeline.preprocessor:
            logger.info("      Aplicando pré-processamento...")
            crop_processed = pipeline.preprocessor.process(crop)
            
            # Salvar crop processado
            crop_proc_path = output_path / f"{image_name}_3c_crop_{i+1}_processed.jpg"
            cv2.imwrite(str(crop_proc_path), crop_processed)
            logger.info(f"      ✅ Crop processado: {crop_proc_path}")
        else:
            crop_processed = crop
            logger.info("      Sem pré-processamento configurado")
        
        # OCR
        logger.info("      Executando OCR...")
        text, confidence = pipeline.ocr_engine.extract_text(crop_processed)
        
        ocr_result = {
            'detection_index': i,
            'text': text,
            'confidence': confidence,
            'bbox': detection['bbox']
        }
        ocr_results.append(ocr_result)
        
        logger.info(f"      📝 Texto: '{text}'")
        logger.info(f"      📊 Confiança: {confidence:.2%}")
        
        # Criar visualização do OCR no crop
        crop_viz = crop_processed.copy()
        if len(crop_viz.shape) == 2:
            crop_viz = cv2.cvtColor(crop_viz, cv2.COLOR_GRAY2BGR)
        
        # Adicionar texto reconhecido
        h, w = crop_viz.shape[:2]
        cv2.putText(crop_viz, f"OCR: {text}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(crop_viz, f"Conf: {confidence:.2%}", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        crop_ocr_path = output_path / f"{image_name}_3d_crop_{i+1}_ocr.jpg"
        cv2.imwrite(str(crop_ocr_path), crop_viz)
        logger.info(f"      ✅ OCR visualização: {crop_ocr_path}")
    
    # ========================================
    # ETAPA 4: PARSING DE DATAS
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("📅 ETAPA 4: PARSING DE DATAS")
    logger.info("="*60)
    
    dates = []
    for ocr_result in ocr_results:
        parsed_date, parse_confidence = pipeline.date_parser.parse(ocr_result['text'])
        
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
            logger.info(f"   ✅ Data extraída: {date_result['date_str']}")
            logger.info(f"      Texto original: '{ocr_result['text']}'")
            logger.info(f"      Confiança OCR: {ocr_result['confidence']:.2%}")
            logger.info(f"      Confiança Parse: {parse_confidence:.2%}")
            logger.info(f"      Confiança combinada: {date_result['combined_confidence']:.2%}")
        else:
            logger.warning(f"   ❌ Não foi possível extrair data de: '{ocr_result['text']}'")
    
    # Melhor data
    best_date = None
    if dates:
        best_date = max(dates, key=lambda x: x['combined_confidence'])
        logger.info(f"\n   🏆 MELHOR RESULTADO: {best_date['date_str']} (conf: {best_date['combined_confidence']:.2%})")
    
    # ========================================
    # ETAPA 5: RESULTADO FINAL
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("📊 ETAPA 5: RESULTADO FINAL")
    logger.info("="*60)
    
    # Criar visualização final
    final_viz = image.copy()
    
    for date_result in dates:
        bbox = date_result['bbox']
        # Extrair coordenadas do dicionário estruturado
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])
        
        # Cor baseada na confiança
        conf = date_result['combined_confidence']
        if conf >= 0.8:
            color = (0, 255, 0)  # Verde - alta confiança
        elif conf >= 0.6:
            color = (0, 255, 255)  # Amarelo - média confiança
        else:
            color = (0, 165, 255)  # Laranja - baixa confiança
        
        # Desenhar bbox
        thickness = 3 if date_result == best_date else 2
        cv2.rectangle(final_viz, (x1, y1), (x2, y2), color, thickness)
        
        # Label com data
        label = f"{date_result['date_str']} ({conf:.1%})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background para o texto
        cv2.rectangle(final_viz, (x1, y1-label_size[1]-10), 
                     (x1+label_size[0]+10, y1), color, -1)
        
        # Texto
        cv2.putText(final_viz, label, (x1+5, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Marcador de melhor resultado
        if date_result == best_date:
            cv2.putText(final_viz, "MELHOR", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Salvar resultado final
    result_path = output_path / f"{image_name}_4_result.jpg"
    cv2.imwrite(str(result_path), final_viz)
    logger.info(f"   ✅ Resultado final salvo: {result_path}")
    
    # Criar resultado resumido
    result = {
        'success': len(dates) > 0,
        'detections': detections,
        'ocr_results': ocr_results,
        'dates': dates,
        'best_date': best_date,
        'processing_time': 0.0,
        'image_name': image_name
    }
    
    # Exibir resultado
    print_result(result)
    
    logger.info(f"\n✅ Todos os outputs salvos em: {output_path}/")
    logger.info(f"   1️⃣  {image_name}_1_input.jpg - Imagem original")
    logger.info(f"   2️⃣  {image_name}_2_yolo_detection.jpg - Detecções YOLO")
    logger.info(f"   3️⃣  {image_name}_3a_mask_*.jpg - Máscaras de segmentação (se disponível)")
    logger.info(f"   3️⃣  {image_name}_3b_crop_*_original.jpg - Crops originais")
    logger.info(f"   3️⃣  {image_name}_3c_crop_*_processed.jpg - Crops pré-processados")
    logger.info(f"   3️⃣  {image_name}_3d_crop_*_ocr.jpg - Resultados OCR")
    logger.info(f"   4️⃣  {image_name}_4_result.jpg - Resultado final")


def test_directory(pipeline: FullPipeline, image_dir: str, pattern: str = "*.jpg"):
    """
    Testa pipeline em um diretório.
    
    Args:
        pipeline: Instância do pipeline
        image_dir: Diretório com imagens
        pattern: Padrão de arquivos
    """
    logger.info(f"📁 Testando diretório: {image_dir}")
    
    # Processar todas as imagens
    results = pipeline.process_directory(image_dir, pattern)
    
    # Resumo já é salvo automaticamente pelo pipeline
    logger.info(f"✅ Processamento concluído! {len(results)} imagens processadas")


def print_result(result: dict):
    """Imprime resultado formatado."""
    print("\n" + "="*60)
    print("📊 RESULTADO DO PIPELINE")
    print("="*60)
    
    print(f"\n🎯 Status: {'✅ Sucesso' if result['success'] else '❌ Falha'}")
    print(f"⏱️  Tempo de processamento: {result['processing_time']:.2f}s")
    
    # Detecções
    print(f"\n📍 Detecções YOLO: {len(result['detections'])}")
    for i, det in enumerate(result['detections']):
        print(f"   {i+1}. {det['class_name']} (conf: {det['confidence']:.2%})")
    
    # OCR
    print(f"\n🔍 Resultados OCR: {len(result['ocr_results'])}")
    for i, ocr in enumerate(result['ocr_results']):
        print(f"   {i+1}. '{ocr['text']}' (conf: {ocr['confidence']:.2%})")
    
    # Datas
    print(f"\n📅 Datas Extraídas: {len(result['dates'])}")
    for i, date in enumerate(result['dates']):
        print(f"   {i+1}. {date['date_str']} (conf: {date['combined_confidence']:.2%})")
    
    # Melhor data
    if result['best_date']:
        best = result['best_date']
        print(f"\n🏆 MELHOR RESULTADO:")
        print(f"   Data: {best['date_str']}")
        print(f"   Texto original: '{best['text']}'")
        print(f"   Confiança OCR: {best['ocr_confidence']:.2%}")
        print(f"   Confiança Parse: {best['parse_confidence']:.2%}")
        print(f"   Confiança combinada: {best['combined_confidence']:.2%}")
    else:
        print("\n⚠️  Nenhuma data válida encontrada")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Testa o pipeline completo YOLO → OCR → Parse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Processar uma imagem
  python scripts/pipeline/test_full_pipeline.py --image data/test.jpg
  
  # Processar diretório
  python scripts/pipeline/test_full_pipeline.py --image-dir data/test_images/
  
  # Com config customizada
  python scripts/pipeline/test_full_pipeline.py --image data/test.jpg --config config/pipeline/full_pipeline.yaml
  
  # Processar múltiplos formatos
  python scripts/pipeline/test_full_pipeline.py --image-dir data/ --pattern "*.png"
        """
    )
    
    # Argumentos
    parser.add_argument(
        '--image',
        type=str,
        help='Caminho para uma imagem'
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Diretório com imagens'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='Padrão de arquivos (default: *.jpg)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline/full_pipeline.yaml',
        help='Arquivo de configuração (default: config/pipeline/full_pipeline.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/pipeline_steps',
        help='Diretório de saída para etapas (default: outputs/pipeline_steps)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Desabilita salvamento de visualizações'
    )
    
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Salva crops das detecções'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.image and not args.image_dir:
        parser.error("Especifique --image ou --image-dir")
    
    # Carregar configuração
    logger.info(f"📋 Carregando configuração: {args.config}")
    config = load_pipeline_config(args.config)
    
    # Sobrescrever opções se especificadas
    if args.no_viz:
        config.setdefault('output', {})['save_visualizations'] = False
    
    if args.save_crops:
        config.setdefault('output', {})['save_crops'] = True
    
    # Inicializar pipeline
    logger.info("🚀 Inicializando pipeline...")
    pipeline = FullPipeline(config)
    
    # Executar testes
    if args.image:
        test_single_image(pipeline, args.image, args.output)
    else:
        test_directory(pipeline, args.image_dir, args.pattern)
    
    logger.info("✅ Teste concluído!")


if __name__ == '__main__':
    main()
