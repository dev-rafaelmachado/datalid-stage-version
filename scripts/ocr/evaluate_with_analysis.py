"""
📊 Script de Avaliação de OCR com Análise Detalhada
Executa avaliação de engine OCR e gera análise completa com gráficos e estatísticas.

Uso:
    python scripts/ocr/evaluate_with_analysis.py --engine parseq_enhanced --test-data data/ocr_test --output outputs/ocr_analysis
"""

import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ocr.evaluator import OCREvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação de OCR com análise detalhada"
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Nome do engine (tesseract, easyocr, paddleocr, parseq, parseq_enhanced, trocr)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Caminho para arquivo de configuração do engine"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Diretório com dados de teste (deve conter ground_truth.json e pasta images/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Diretório de saída para análise"
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        help="Configuração de pré-processamento (ppro-tesseract, ppro-parseq, etc.)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Pular geração de gráficos (útil se matplotlib não estiver instalado)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("📊 AVALIAÇÃO DE OCR COM ANÁLISE DETALHADA")
    logger.info("="*80)
    logger.info(f"🤖 Engine: {args.engine}")
    logger.info(f"📁 Dados de teste: {args.test_data}")
    logger.info(f"💾 Saída: {args.output}")
    logger.info("="*80)
    
    # Criar evaluator
    evaluator = OCREvaluator()
    
    # Buscar configuração do engine
    if args.config:
        engine_config_path = args.config
    else:
        # Buscar configuração padrão
        config_dir = Path(__file__).parent.parent.parent / 'config' / 'ocr'
        engine_config_path = config_dir / f'{args.engine}.yaml'
        
        if not engine_config_path.exists():
            logger.warning(f"⚠️ Configuração não encontrada: {engine_config_path}")
            engine_config_path = None
    
    # Adicionar engine
    try:
        evaluator.add_engine(args.engine, str(engine_config_path) if engine_config_path else None)
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar engine: {e}")
        return 1
    
    # Verificar dados de teste
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"❌ Diretório de teste não encontrado: {test_data_path}")
        return 1
    
    gt_path = test_data_path / "ground_truth.json"
    if not gt_path.exists():
        logger.error(f"❌ Ground truth não encontrado: {gt_path}")
        logger.info("💡 Execute: make ocr-annotate")
        return 1
    
    images_dir = test_data_path / "images"
    if not images_dir.exists():
        logger.error(f"❌ Diretório de imagens não encontrado: {images_dir}")
        return 1
    
    # Carregar ground truth
    import json
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    if 'annotations' not in gt_data:
        logger.error("❌ Ground truth inválido (falta campo 'annotations')")
        return 1
    
    ground_truth = gt_data['annotations']
    
    # Carregar preprocessador se especificado
    preprocessor = None
    if args.preprocessing:
        from src.ocr.config import load_preprocessing_config
        from src.ocr.preprocessors import ImagePreprocessor
        
        preprocessing_name = args.preprocessing if args.preprocessing.startswith('ppro-') else f'ppro-{args.preprocessing}'
        config_dir = Path(__file__).parent.parent.parent / 'config' / 'preprocessing'
        prep_config_path = config_dir / f'{preprocessing_name}.yaml'
        
        if prep_config_path.exists():
            try:
                prep_config = load_preprocessing_config(str(prep_config_path))
                preprocessor = ImagePreprocessor(prep_config)
                logger.success(f"✅ Pré-processamento configurado: {preprocessing_name}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar pré-processamento: {e}")
        else:
            logger.warning(f"⚠️ Configuração de pré-processamento não encontrada: {prep_config_path}")
    
    # Processar imagens
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        logger.error(f"❌ Nenhuma imagem encontrada em: {images_dir}")
        return 1
    
    logger.info(f"📸 Processando {len(image_files)} imagens...")
    
    # Criar diretório para debug de imagens
    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🖼️  Salvando imagens de debug em: {debug_dir}")
    
    results = []
    for img_file in image_files:
        filename = img_file.name
        
        # Verificar ground truth
        if filename not in ground_truth:
            logger.warning(f"⚠️ Ground truth não encontrado para: {filename}")
            continue
        
        expected_text = ground_truth[filename]
        
        if not expected_text or expected_text.strip() == "":
            logger.warning(f"⚠️ Ground truth vazio para: {filename}")
            continue
        
        # Carregar imagem
        image = cv2.imread(str(img_file))
        if image is None:
            logger.warning(f"⚠️ Erro ao carregar: {filename}")
            continue
        
        # Salvar imagem original
        img_debug_dir = debug_dir / Path(filename).stem
        img_debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(img_debug_dir / "00_original.png"), image)
        
        # Aplicar pré-processamento e salvar etapas
        if preprocessor:
            try:
                preprocessed = preprocessor.process(image)
                cv2.imwrite(str(img_debug_dir / "01_preprocessed.png"), preprocessed)
            except Exception as e:
                logger.warning(f"⚠️ Erro no pré-processamento: {e}")
                preprocessed = image
        else:
            preprocessed = image
        
        # Avaliar
        try:
            result = evaluator.evaluate_single(
                image,
                expected_text,
                args.engine,
                preprocessor=preprocessor,
                debug_dir=img_debug_dir  # Passar diretório de debug
            )
            result['image_file'] = filename
            results.append(result)
            
            # Log progresso
            status = '✅' if result['exact_match'] else '❌'
            logger.info(f"  {status} {filename}: '{result.get('predicted_text', '')}' vs '{expected_text}' (CER: {result['character_error_rate']:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {filename}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    if not results:
        logger.error("❌ Nenhum resultado obtido")
        return 1
    
    # Converter para DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Criar diretório de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar resultados básicos
    results_file = output_dir / f"{args.engine}_results.csv"
    df.to_csv(results_file, index=False, encoding='utf-8')
    logger.success(f"💾 Resultados salvos: {results_file}")
    
    # Salvar JSON também
    json_file = output_dir / f"{args.engine}_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"💾 Resultados JSON salvos: {json_file}")
    
    # Gerar análise detalhada com visualizações
    logger.info("\n" + "="*80)
    logger.info("🎨 GERANDO ANÁLISE DETALHADA")
    logger.info("="*80)
    
    try:
        stats = evaluator.generate_detailed_analysis(df, str(output_dir))
        
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO DA ANÁLISE")
        logger.info("="*80)
        
        basic = stats.get('basic', {})
        logger.info(f"✅ Total de amostras: {basic.get('total_samples', 0)}")
        logger.info(f"✅ Exact Match Rate: {basic.get('exact_match_rate', 0):.2%}")
        logger.info(f"📊 Average CER: {basic.get('avg_cer', 0):.3f}")
        logger.info(f"📊 Median CER: {basic.get('median_cer', 0):.3f}")
        logger.info(f"📈 Average Confidence: {basic.get('avg_confidence', 0):.2%}")
        logger.info(f"⏱️  Average Time: {basic.get('avg_processing_time', 0):.3f}s")
        logger.info(f"⏱️  Total Time: {basic.get('total_processing_time', 0):.1f}s")
        
        # Categorias de erro
        errors = stats.get('errors', {})
        if errors:
            logger.info("\n📊 Categorias de Erro:")
            for category, data in errors.items():
                if isinstance(data, dict) and 'count' in data:
                    logger.info(f"  • {category}: {data['count']} ({data.get('percentage', 0):.1f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ANÁLISE COMPLETA!")
        logger.info("="*80)
        logger.info(f"📁 Todos os arquivos foram salvos em: {output_dir}")
        logger.info(f"📄 Relatório HTML: {output_dir / 'report.html'}")
        logger.info(f"📊 Estatísticas JSON: {output_dir / 'statistics.json'}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar análise detalhada: {e}")
        import traceback
        traceback.print_exc()
        
        # Exibir resumo básico mesmo com erro
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO BÁSICO")
        logger.info("="*80)
        
        exact_matches = sum(1 for r in results if r.get('exact_match', False))
        total = len(results)
        accuracy = exact_matches / total if total > 0 else 0
        
        import numpy as np
        avg_time = np.mean([r.get('processing_time', 0) for r in results])
        avg_conf = np.mean([r.get('confidence', 0) for r in results if r.get('confidence', 0) > 0])
        avg_cer = np.mean([r.get('character_error_rate', 0) for r in results])
        
        logger.info(f"✅ Exact Match: {exact_matches}/{total} ({accuracy*100:.1f}%)")
        logger.info(f"📊 Average CER: {avg_cer:.3f}")
        logger.info(f"⏱️  Tempo médio: {avg_time:.3f}s")
        logger.info(f"📈 Confiança média: {avg_conf:.2f}")
        logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
