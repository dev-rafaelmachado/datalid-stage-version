#!/usr/bin/env python3
"""
🧪 Testa modelo YOLO no conjunto de teste

Script para avaliar modelos treinados no conjunto de teste final,
fornecendo métricas não enviesadas de performance.

Uso:
    python scripts/evaluation/test_model.py MODEL_PATH --data DATA_YAML
    
Exemplo:
    python scripts/evaluation/test_model.py \
        experiments/yolov8n_seg_baseline/weights/best.pt \
        --data data/processed/v1_segment/data.yaml \
        --output outputs/test_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("❌ Ultralytics não instalado. Instale com: pip install ultralytics")
    sys.exit(1)


def test_model(
    model_path: str,
    data_yaml: str,
    split: str = 'test',
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    save_json: bool = True,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    batch_size: int = 1,
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Testa modelo no conjunto especificado.
    
    Args:
        model_path: Caminho do modelo treinado (ex: experiments/best.pt)
        data_yaml: Caminho do data.yaml
        split: Split para testar ('test', 'val', ou 'train')
        output_dir: Diretório para salvar resultados
        save_plots: Salvar plots de visualização
        save_json: Salvar resultados em JSON
        conf_threshold: Threshold de confiança
        iou_threshold: Threshold de IoU
        batch_size: Batch size para validação (menor = menos memória)
        imgsz: Tamanho da imagem (menor = menos memória)
        
    Returns:
        Dicionário com métricas do teste
        
    Raises:
        FileNotFoundError: Se modelo ou data.yaml não encontrados
    """
    model_path = Path(model_path)
    data_yaml_path = Path(data_yaml)
    
    # Validações
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Modelo não encontrado: {model_path}")
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"❌ data.yaml não encontrado: {data_yaml_path}")
    
    logger.info("=" * 70)
    logger.info("🧪 TESTANDO MODELO YOLO")
    logger.info("=" * 70)
    logger.info(f"📦 Modelo: {model_path}")
    logger.info(f"📊 Dataset: {data_yaml}")
    logger.info(f"🎯 Split: {split}")
    logger.info(f"⚙️  Conf threshold: {conf_threshold}")
    logger.info(f"⚙️  IoU threshold: {iou_threshold}")
    logger.info(f"⚙️  Batch size: {batch_size}")
    logger.info(f"⚙️  Image size: {imgsz}")
    logger.info("=" * 70)
    
    # Configurar output
    if output_dir is None:
        output_dir = ROOT_DIR / 'outputs' / 'test_results'
    else:
        output_dir = Path(output_dir)
    
    exp_name = f"{model_path.stem}_{split}"
    
    # Carregar modelo
    logger.info("📥 Carregando modelo...")
    model = YOLO(str(model_path))
    
    # Detectar tipo de tarefa
    task_type = 'segment' if 'seg' in model_path.stem else 'detect'
    logger.info(f"🎯 Tipo de tarefa detectado: {task_type}")
    
    # Executar teste
    logger.info(f"🔍 Executando validação no conjunto '{split}'...")
    
    # Tentar com save_json, se falhar, desabilitar
    # Adicionar parâmetros para otimizar memória
    val_kwargs = {
        'data': str(data_yaml_path),
        'split': split,
        'plots': save_plots,
        'save_json': save_json,
        'project': str(output_dir),
        'name': exp_name,
        'conf': conf_threshold,
        'iou': iou_threshold,
        'batch': batch_size,  # Reduzir batch para economizar memória
        'imgsz': imgsz,       # Reduzir tamanho de imagem se necessário
        'verbose': True,
        'half': False,        # Desabilitar FP16 para evitar problemas
    }
    
    try:
        results = model.val(**val_kwargs)
    except ModuleNotFoundError as e:
        if 'faster_coco_eval' in str(e):
            logger.warning("⚠️ faster_coco_eval não disponível, desabilitando save_json")
            logger.info("💡 Para habilitar: pip install faster-coco-eval")
            val_kwargs['save_json'] = False
            results = model.val(**val_kwargs)
        else:
            raise
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.error("❌ Erro de memória GPU!")
            logger.warning("💡 Tentando novamente com batch_size=1 e imgsz menor...")
            
            # Limpar cache da GPU
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Tentar com configurações mais conservadoras
            val_kwargs['batch'] = 1
            val_kwargs['imgsz'] = min(imgsz, 320)
            val_kwargs['save_json'] = False  # Desabilitar para economizar memória
            
            logger.info(f"🔄 Retentando com batch=1, imgsz={val_kwargs['imgsz']}")
            results = model.val(**val_kwargs)
        else:
            raise
    
    # Extrair métricas
    logger.info("📊 Extraindo métricas...")
    
    test_metrics = {
        'model': str(model_path),
        'dataset': str(data_yaml),
        'split': split,
        'task': task_type,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
    }
    
    # Métricas de detecção (box)
    if hasattr(results, 'box') and results.box is not None:
        test_metrics.update({
            'box_mAP50': float(results.box.map50),
            'box_mAP50_95': float(results.box.map),
            'box_mAP75': float(results.box.map75) if hasattr(results.box, 'map75') else None,
            'box_precision': float(results.box.p),
            'box_recall': float(results.box.r),
        })
        
        # Calcular F1
        p, r = results.box.p, results.box.r
        if p > 0 and r > 0:
            test_metrics['box_f1'] = float(2 * p * r / (p + r))
        else:
            test_metrics['box_f1'] = 0.0
    
    # Métricas de segmentação (mask)
    if hasattr(results, 'masks') and results.masks is not None:
        test_metrics.update({
            'mask_mAP50': float(results.masks.map50),
            'mask_mAP50_95': float(results.masks.map),
            'mask_mAP75': float(results.masks.map75) if hasattr(results.masks, 'map75') else None,
            'mask_precision': float(results.masks.p),
            'mask_recall': float(results.masks.r),
        })
        
        # Calcular F1
        p, r = results.masks.p, results.masks.r
        if p > 0 and r > 0:
            test_metrics['mask_f1'] = float(2 * p * r / (p + r))
        else:
            test_metrics['mask_f1'] = 0.0
    
    # Métricas por classe
    if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
        class_metrics = []
        for idx, class_id in enumerate(results.box.ap_class_index):
            class_info = {
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)],
                'ap50': float(results.box.ap50[idx]) if len(results.box.ap50) > idx else None,
                'ap': float(results.box.ap[idx]) if len(results.box.ap) > idx else None,
            }
            class_metrics.append(class_info)
        
        test_metrics['per_class_metrics'] = class_metrics
    
    # Salvar resultados
    output_json = output_dir / exp_name / 'test_metrics.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    
    # Log resultados
    logger.success("✅ Teste concluído com sucesso!")
    logger.info("=" * 70)
    logger.info(f"🎯 MÉTRICAS DO CONJUNTO '{split.upper()}'")
    logger.info("=" * 70)
    
    # Box metrics
    if 'box_mAP50' in test_metrics:
        logger.info("📦 Métricas de Detecção (Box):")
        logger.info(f"  • mAP@0.5      : {test_metrics['box_mAP50']:.4f}")
        logger.info(f"  • mAP@0.5:0.95 : {test_metrics['box_mAP50_95']:.4f}")
        if test_metrics.get('box_mAP75'):
            logger.info(f"  • mAP@0.75     : {test_metrics['box_mAP75']:.4f}")
        logger.info(f"  • Precision    : {test_metrics['box_precision']:.4f}")
        logger.info(f"  • Recall       : {test_metrics['box_recall']:.4f}")
        logger.info(f"  • F1-Score     : {test_metrics['box_f1']:.4f}")
    
    # Mask metrics
    if 'mask_mAP50' in test_metrics:
        logger.info("\n🎭 Métricas de Segmentação (Mask):")
        logger.info(f"  • mAP@0.5      : {test_metrics['mask_mAP50']:.4f}")
        logger.info(f"  • mAP@0.5:0.95 : {test_metrics['mask_mAP50_95']:.4f}")
        if test_metrics.get('mask_mAP75'):
            logger.info(f"  • mAP@0.75     : {test_metrics['mask_mAP75']:.4f}")
        logger.info(f"  • Precision    : {test_metrics['mask_precision']:.4f}")
        logger.info(f"  • Recall       : {test_metrics['mask_recall']:.4f}")
        logger.info(f"  • F1-Score     : {test_metrics['mask_f1']:.4f}")
    
    # Per-class metrics
    if 'per_class_metrics' in test_metrics:
        logger.info("\n📊 Métricas por Classe:")
        for class_info in test_metrics['per_class_metrics']:
            logger.info(f"  • {class_info['class_name']}:")
            if class_info['ap50'] is not None:
                logger.info(f"    - AP@0.5      : {class_info['ap50']:.4f}")
            if class_info['ap'] is not None:
                logger.info(f"    - AP@0.5:0.95 : {class_info['ap']:.4f}")
    
    logger.info("=" * 70)
    logger.info(f"📄 Resultados salvos em: {output_json}")
    logger.info(f"📊 Visualizações em: {output_dir / exp_name}")
    logger.info("=" * 70)
    
    return test_metrics


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Testa modelo YOLO no conjunto de teste",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        '--model_path',
        type=str,
        help='Caminho do modelo treinado (.pt)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/v1_segment/data.yaml',
        help='Caminho do data.yaml'
    )
    
    # Argumentos opcionais
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val', 'train'],
        help='Split para testar'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Diretório de saída (padrão: outputs/test_results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Não salvar plots de visualização'
    )
    
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Não salvar resultados em JSON'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Threshold de confiança'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='Threshold de IoU'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size para validação (menor = menos memória GPU)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamanho da imagem (menor = menos memória GPU)'
    )
    
    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_arguments()
    
    try:
        test_metrics = test_model(
            model_path=args.model_path,
            data_yaml=args.data,
            split=args.split,
            output_dir=args.output,
            save_plots=not args.no_plots,
            save_json=not args.no_json,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            batch_size=args.batch,
            imgsz=args.imgsz
        )
        
        logger.success("✅ Teste finalizado com sucesso!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
