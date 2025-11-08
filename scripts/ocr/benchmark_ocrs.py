"""
Script para benchmark de engines OCR
Compara Tesseract, EasyOCR, PaddleOCR e TrOCR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ocr.config import load_ocr_config, load_preprocessing_config
from src.ocr.engines.easyocr import EasyOCREngine
from src.ocr.engines.openocr import OpenOCREngine
from src.ocr.engines.paddleocr import PaddleOCREngine
from src.ocr.engines.parseq import PARSeqEngine
from src.ocr.engines.tesseract import TesseractEngine
from src.ocr.engines.trocr import TrOCREngine
from src.ocr.preprocessors import OCRPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OCR engines")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/ocr_comparison.yaml",
        help="Arquivo de configuração do experimento"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ocr_benchmarks/comparison",
        help="Pasta de saída"
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Engine específico para testar (tesseract, easyocr, openocr, paddleocr, trocr, parseq). Se não especificado, testa todos."
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["tesseract", "easyocr", "openocr", "paddleocr", "trocr", "parseq"],
        help="Engines para testar (usado se --engine não for especificado)"
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["none", "minimal", "medium", "heavy"],
        default="medium",
        help="Nível de pré-processamento (none para desabilitar)"
    )
    parser.add_argument(
        "--preprocessing-config",
        type=str,
        help="Arquivo de configuração de pré-processamento customizado"
    )
    return parser.parse_args()


def load_ground_truth(gt_path: Path) -> dict:
    """Carrega ground truth"""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('annotations', {})


def calculate_metrics(predicted: str, ground_truth: str) -> dict:
    """Calcula métricas de comparação"""
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    
    # Exact match
    exact_match = pred == gt
    
    # Partial match (subsequência)
    partial_match = pred in gt or gt in pred
    
    # Character Error Rate (CER)
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
    
    cer = levenshtein_distance(pred, gt) / max(len(gt), 1) if gt else 1.0
    
    return {
        'exact_match': exact_match,
        'partial_match': partial_match,
        'cer': cer
    }


def benchmark_engine(
    engine,
    images_dir: Path,
    ground_truth: dict,
    output_dir: Path,
    preprocessor=None
):
    """Executa benchmark em um engine"""
    engine_name = engine.get_name()
    print(f"\n🔍 Testando {engine_name}...")
    
    if preprocessor:
        print(f"   📊 Pré-processamento: {preprocessor.name}")
    
    results = []
    total_time = 0
    
    image_files = sorted(images_dir.glob("*.jpg"))
    
    for img_path in tqdm(image_files, desc=f"  {engine_name}"):
        # Ler imagem
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Obter ground truth
        gt_text = ground_truth.get(img_path.name, "")
        if not gt_text:
            continue
        
        # Aplicar pré-processamento se configurado
        processed_image = image
        if preprocessor:
            try:
                processed_image = preprocessor.process(image)
            except Exception as e:
                print(f"  ⚠️  Erro no pré-processamento de {img_path.name}: {e}")
                continue
        
        # Extrair texto
        start_time = time.time()
        try:
            extracted_text, confidence = engine.extract_text(processed_image)
        except Exception as e:
            print(f"  ⚠️  Erro em {img_path.name}: {e}")
            extracted_text = ""
            confidence = 0.0
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        # Calcular métricas
        metrics = calculate_metrics(extracted_text, gt_text)
        
        results.append({
            'image': img_path.name,
            'engine': engine_name,
            'predicted': extracted_text,
            'ground_truth': gt_text,
            'confidence': confidence,
            'time': elapsed_time,
            **metrics
        })
    
    # Salvar resultados individuais
    df = pd.DataFrame(results)
    output_path = output_dir / f"{engine_name}_results.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Métricas agregadas
    metrics_summary = {
        'engine': engine_name,
        'total_images': len(results),
        'exact_match_rate': df['exact_match'].mean(),
        'partial_match_rate': df['partial_match'].mean(),
        'mean_cer': df['cer'].mean(),
        'total_time': total_time,
        'avg_time_per_image': total_time / len(results) if results else 0
    }
    
    print(f"  ✅ {engine_name}:")
    print(f"     Exact Match: {metrics_summary['exact_match_rate']:.2%}")
    print(f"     Partial Match: {metrics_summary['partial_match_rate']:.2%}")
    print(f"     CER: {metrics_summary['mean_cer']:.3f}")
    print(f"     Tempo médio: {metrics_summary['avg_time_per_image']:.3f}s")
    
    return metrics_summary, df


def create_comparison_plots(all_results: pd.DataFrame, output_dir: Path):
    """Cria gráficos de comparação"""
    print("\n📊 Gerando visualizações...")
    
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Comparação de Exact Match
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Exact Match Rate
    ax = axes[0, 0]
    summary = all_results.groupby('engine')['exact_match'].mean().sort_values(ascending=False)
    summary.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Exact Match Rate por Engine', fontsize=14, fontweight='bold')
    ax.set_ylabel('Taxa de Acerto Exato')
    ax.set_xlabel('Engine OCR')
    ax.set_ylim(0, 1)
    for i, v in enumerate(summary):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # Character Error Rate (CER)
    ax = axes[0, 1]
    summary_cer = all_results.groupby('engine')['cer'].mean().sort_values()
    summary_cer.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Character Error Rate (CER) por Engine', fontsize=14, fontweight='bold')
    ax.set_ylabel('CER (menor é melhor)')
    ax.set_xlabel('Engine OCR')
    for i, v in enumerate(summary_cer):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Tempo de processamento
    ax = axes[1, 0]
    summary_time = all_results.groupby('engine')['time'].mean().sort_values()
    summary_time.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Tempo Médio de Processamento', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tempo (segundos)')
    ax.set_xlabel('Engine OCR')
    for i, v in enumerate(summary_time):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')
    
    # Heatmap de comparação
    ax = axes[1, 1]
    comparison_data = all_results.groupby('engine').agg({
        'exact_match': 'mean',
        'partial_match': 'mean',
        'cer': 'mean',
        'time': 'mean'
    }).T
    
    sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Valor'})
    ax.set_title('Heatmap de Métricas', fontsize=14, fontweight='bold')
    ax.set_ylabel('Métrica')
    ax.set_xlabel('Engine OCR')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Gráfico salvo: {output_dir / 'comparison_summary.png'}")
    
    # 2. Distribuição de erros por engine
    fig, ax = plt.subplots(figsize=(12, 6))
    all_results.boxplot(column='cer', by='engine', ax=ax)
    ax.set_title('Distribuição do Character Error Rate por Engine', fontsize=14, fontweight='bold')
    ax.set_ylabel('CER')
    ax.set_xlabel('Engine OCR')
    plt.suptitle('')  # Remove título automático
    plt.savefig(output_dir / 'cer_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Gráfico salvo: {output_dir / 'cer_distribution.png'}")
    
    plt.close('all')


def main():
    args = parse_args()
    
    print("="*60)
    print("🏆 BENCHMARK DE ENGINES OCR")
    print("="*60)
    print()
    
    # Criar pasta de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar configuração
    if Path(args.config).exists():
        config = load_ocr_config(args.config)
        # A config já aponta para data/ocr_test/images, então vamos usar o parent
        config_path = Path(config.get('dataset', {}).get('path', 'data/ocr_test/images'))
        if config_path.name == 'images':
            dataset_path = config_path.parent
        else:
            dataset_path = config_path
    else:
        dataset_path = Path('data/ocr_test')
    
    images_dir = dataset_path / "images"
    gt_path = dataset_path / "ground_truth.json"
    
    if not images_dir.exists():
        print(f"❌ Pasta de imagens não encontrada: {images_dir}")
        print("💡 Execute primeiro: make ocr-prepare-data")
        return
    
    if not gt_path.exists():
        print(f"❌ Ground truth não encontrado: {gt_path}")
        print("💡 Anote o ground truth primeiro!")
        return
    
    # Carregar ground truth
    print(f"📂 Dataset: {dataset_path}")
    ground_truth = load_ground_truth(gt_path)
    print(f"📝 Ground truth: {len(ground_truth)} anotações")
    print()
    
    # Inicializar engines
    engines = []
    engine_configs = {
        'tesseract': 'config/ocr/tesseract.yaml',
        'easyocr': 'config/ocr/easyocr.yaml',
        'openocr': 'config/ocr/openocr.yaml',
        'paddleocr': 'config/ocr/paddleocr.yaml',
        'trocr': 'config/ocr/trocr.yaml',
        'parseq': 'config/ocr/parseq.yaml'
    }
    
    # Se --engine foi especificado, usar apenas ele
    if args.engine:
        engines_to_test = [args.engine.lower()]
        print(f"\n🎯 Testando apenas o engine: {args.engine}")
    else:
        engines_to_test = args.engines
        print(f"\n🎯 Testando múltiplos engines: {', '.join(engines_to_test)}")
    
    for engine_name in engines_to_test:
        print(f"🔧 Inicializando {engine_name}...")
        config_path = engine_configs.get(engine_name)
        
        try:
            if engine_name == 'tesseract':
                engine = TesseractEngine(load_ocr_config(config_path) if config_path else {})
            elif engine_name == 'easyocr':
                engine = EasyOCREngine(load_ocr_config(config_path) if config_path else {})
            elif engine_name == 'openocr':
                engine = OpenOCREngine(load_ocr_config(config_path) if config_path else {})
            elif engine_name == 'paddleocr':
                engine = PaddleOCREngine(load_ocr_config(config_path) if config_path else {})
            elif engine_name == 'trocr':
                engine = TrOCREngine(load_ocr_config(config_path) if config_path else {})
            elif engine_name == 'parseq':
                engine = PARSeqEngine(load_ocr_config(config_path) if config_path else {})
            else:
                print(f"  ⚠️  Engine desconhecido: {engine_name}")
                continue
            
            # Inicializar o engine (carregar modelo)
            engine.initialize()
            
            engines.append(engine)
            print(f"  ✅ {engine_name} inicializado")
        except Exception as e:
            print(f"  ❌ Erro ao inicializar {engine_name}: {e}")
    
    if not engines:
        print("❌ Nenhum engine disponível!")
        return
    
    # Configurar pré-processamento
    preprocessor = None
    if args.preprocessing != "none":
        print(f"\n🔧 Configurando pré-processamento: {args.preprocessing}")
        
        if args.preprocessing_config:
            config_path = Path(args.preprocessing_config)
        else:
            config_path = Path(f"config/preprocessing/{args.preprocessing}.yaml")
        
        if config_path.exists():
            try:
                prep_config = load_preprocessing_config(str(config_path))
                preprocessor = OCRPreprocessor(prep_config)
                print(f"  ✅ Pré-processamento {args.preprocessing} configurado")
            except Exception as e:
                print(f"  ⚠️  Erro ao carregar pré-processamento: {e}")
                print(f"  ℹ️  Continuando sem pré-processamento")
        else:
            print(f"  ⚠️  Configuração não encontrada: {config_path}")
            print(f"  ℹ️  Continuando sem pré-processamento")
    else:
        print("\nℹ️  Pré-processamento desabilitado")
    
    # Executar benchmarks
    print("\n" + "="*60)
    print("🚀 EXECUTANDO BENCHMARKS")
    print("="*60)
    
    all_summaries = []
    all_results = []
    
    for engine in engines:
        summary, results = benchmark_engine(
            engine,
            images_dir,
            ground_truth,
            output_dir,
            preprocessor=preprocessor
        )
        all_summaries.append(summary)
        all_results.append(results)
    
    # Consolidar resultados
    summary_df = pd.DataFrame(all_summaries)
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Salvar resultados consolidados
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
    all_results_df.to_csv(output_dir / 'all_results.csv', index=False, encoding='utf-8')
    
    # Criar visualizações
    create_comparison_plots(all_results_df, output_dir)
    
    # Mostrar resumo
    print("\n" + "="*60)
    print("📊 RESUMO COMPARATIVO")
    print("="*60)
    print()
    print(summary_df.to_string(index=False))
    print()
    
    # Identificar melhor engine
    best_exact = summary_df.loc[summary_df['exact_match_rate'].idxmax()]
    best_speed = summary_df.loc[summary_df['avg_time_per_image'].idxmin()]
    
    print("🏆 MELHORES ENGINES:")
    print(f"  Acurácia (Exact Match): {best_exact['engine']} ({best_exact['exact_match_rate']:.2%})")
    print(f"  Velocidade: {best_speed['engine']} ({best_speed['avg_time_per_image']:.3f}s)")
    print()
    
    print("="*60)
    print("🎉 BENCHMARK CONCLUÍDO!")
    print("="*60)
    print()
    print(f"📁 Resultados: {output_dir}")
    print(f"📊 Resumo: {output_dir / 'comparison_summary.csv'}")
    print(f"📈 Gráficos: {output_dir / 'comparison_summary.png'}")
    print()


if __name__ == "__main__":
    main()
