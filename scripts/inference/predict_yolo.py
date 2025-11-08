"""
🔮 Script de Predição YOLO
Executa predições com modelos treinados.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.core.config import config
from src.yolo import PredictionResult, YOLOConfig, YOLOPredictor

# Adicionar src ao path ANTES de qualquer import do projeto
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Predição YOLO com modelos treinados",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Modelo
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Caminho do modelo (.pt)'
    )

    # Entrada
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image',
        type=str,
        help='Imagem única para predição'
    )
    group.add_argument(
        '--directory',
        type=str,
        help='Diretório com imagens'
    )
    group.add_argument(
        '--batch',
        nargs='+',
        help='Lista de imagens'
    )

    # Configurações de predição
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold'
    )

    parser.add_argument(
        '--task',
        choices=['detect', 'segment'],
        default='detect',
        help='Tipo de tarefa'
    )

    # Saídas
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/predictions',
        help='Diretório de saída'
    )

    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Salvar imagens com predições'
    )

    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Salvar crops das detecções'
    )

    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Salvar resultados em JSON'
    )

    # Visualização
    parser.add_argument(
        '--show-conf',
        action='store_true',
        default=True,
        help='Mostrar confidence nas imagens'
    )

    parser.add_argument(
        '--show-class',
        action='store_true',
        default=True,
        help='Mostrar nomes das classes'
    )

    # Performance
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Executar benchmark de performance'
    )

    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=10,
        help='Número de runs para benchmark'
    )

    return parser.parse_args()


def setup_output_directory(output_dir: Path) -> dict:
    """Configura diretórios de saída."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        'images': output_dir / 'images',
        'crops': output_dir / 'crops',
        'json': output_dir / 'json'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_prediction_results(
    results: List[PredictionResult],
    output_dirs: dict,
    args
) -> None:
    """Salva resultados das predições."""
    logger.info(f"💾 Salvando resultados de {len(results)} predições...")

    for i, result in enumerate(results):
        if not result.has_detections:
            continue

        base_name = f"prediction_{i:04d}"
        if result.image_path:
            base_name = Path(result.image_path).stem

        # Salvar JSON
        if args.save_json:
            json_path = output_dirs['json'] / f"{base_name}.json"

            import json
            with open(json_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        # Salvar crops
        if args.save_crops and result.crops:
            import cv2

            crops_dir = output_dirs['crops'] / base_name
            crops_dir.mkdir(exist_ok=True)

            for j, crop in enumerate(result.crops):
                crop_path = crops_dir / f"crop_{j:02d}.jpg"
                cv2.imwrite(str(crop_path), crop)


def run_single_image(predictor: YOLOPredictor, args) -> List[PredictionResult]:
    """Executa predição em imagem única."""
    logger.info(f"🔮 Processando imagem: {args.image}")

    result = predictor.predict_image(
        args.image,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        return_crops=args.save_crops
    )

    logger.info(f"✅ Predição concluída: {result.num_detections} detecções "
                f"em {result.inference_time:.3f}s")

    return [result]


def run_directory(predictor: YOLOPredictor, args) -> List[PredictionResult]:
    """Executa predição em diretório."""
    logger.info(f"📁 Processando diretório: {args.directory}")

    results = predictor.predict_directory(
        args.directory,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        return_crops=args.save_crops
    )

    total_detections = sum(r.num_detections for r in results)
    logger.success(f"✅ Diretório processado: {len(results)} imagens, "
                   f"{total_detections} detecções")

    return results


def run_batch(predictor: YOLOPredictor, args) -> List[PredictionResult]:
    """Executa predição em lote."""
    logger.info(f"📦 Processando lote de {len(args.batch)} imagens...")

    results = predictor.predict_batch(
        args.batch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        return_crops=args.save_crops
    )

    total_detections = sum(r.num_detections for r in results)
    logger.success(f"✅ Lote processado: {len(results)} imagens, "
                   f"{total_detections} detecções")

    return results


def run_benchmark(predictor: YOLOPredictor, test_images: List[str], args):
    """Executa benchmark de performance."""
    logger.info(f"⚡ Executando benchmark com {len(test_images)} imagens...")

    metrics = predictor.benchmark(
        test_images,
        num_runs=args.benchmark_runs
    )

    logger.info("📊 RESULTADOS DO BENCHMARK:")
    logger.info("=" * 40)
    logger.info(f"  • FPS: {metrics['fps']:.1f}")
    logger.info(f"  • Tempo médio/lote: {metrics['avg_time_per_batch']:.3f}s")
    logger.info(f"  • Desvio padrão: {metrics['std_time']:.3f}s")
    logger.info(
        f"  • Detecções/imagem: {metrics['avg_detections_per_image']:.1f}")
    logger.info(f"  • Imagens/lote: {metrics['images_per_batch']}")
    logger.info(f"  • Runs: {metrics['num_runs']}")

    return metrics


def visualize_results(
    predictor: YOLOPredictor,
    results: List[PredictionResult],
    output_dirs: dict,
    args
) -> None:
    """Visualiza e salva imagens com predições."""
    if not args.save_images:
        return

    logger.info(f"🎨 Criando visualizações...")

    for result in results:
        if not result.image_path or not result.has_detections:
            continue

        try:
            # Nome da imagem de saída
            image_name = Path(result.image_path).name
            output_path = output_dirs['images'] / f"pred_{image_name}"

            # Criar visualização
            predictor.visualize_prediction(
                result.image_path,
                result,
                save_path=output_path,
                show_confidence=args.show_conf,
                show_class_names=args.show_class
            )

        except Exception as e:
            logger.warning(
                f"⚠️ Erro criando visualização para {result.image_path}: {str(e)}")


def main():
    """Função principal."""
    args = parse_arguments()

    logger.info("🔮 PREDIÇÃO YOLO DATALID 3.0")
    logger.info("=" * 50)

    # Validar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ Modelo não encontrado: {model_path}")
        sys.exit(1)

    try:
        # Configurar saída
        output_dir = Path(args.output_dir)
        output_dirs = setup_output_directory(output_dir)

        logger.info(f"📁 Saída configurada: {output_dir}")

        # Criar predictor
        logger.info(f"🤖 Carregando modelo: {model_path}")
        predictor = YOLOPredictor(
            str(model_path),
            task_type=args.task
        )

        # Log configurações
        logger.info("⚙️ CONFIGURAÇÕES:")
        logger.info(f"  • Modelo: {model_path.name}")
        logger.info(f"  • Tarefa: {args.task}")
        logger.info(f"  • Confidence: {args.conf}")
        logger.info(f"  • IoU: {args.iou}")
        logger.info(f"  • Salvar imagens: {'✅' if args.save_images else '❌'}")
        logger.info(f"  • Salvar crops: {'✅' if args.save_crops else '❌'}")
        logger.info(f"  • Salvar JSON: {'✅' if args.save_json else '❌'}")

        # Executar predições
        results = []

        if args.image:
            results = run_single_image(predictor, args)
        elif args.directory:
            results = run_directory(predictor, args)
        elif args.batch:
            results = run_batch(predictor, args)

        # Executar benchmark se solicitado
        if args.benchmark and results:
            # Até 10 imagens
            test_images = [r.image_path for r in results if r.image_path][:10]
            if test_images:
                benchmark_metrics = run_benchmark(predictor, test_images, args)

                # Salvar métricas de benchmark
                import json
                benchmark_path = output_dir / 'benchmark_metrics.json'
                with open(benchmark_path, 'w') as f:
                    json.dump(benchmark_metrics, f, indent=2)

        # Processar resultados
        if results:
            # Estatísticas gerais
            total_images = len(results)
            total_detections = sum(r.num_detections for r in results)
            images_with_detections = sum(
                1 for r in results if r.has_detections)
            avg_inference_time = sum(
                r.inference_time for r in results) / total_images

            logger.info("\n📊 ESTATÍSTICAS:")
            logger.info("=" * 40)
            logger.info(f"  • Total de imagens: {total_images}")
            logger.info(f"  • Total de detecções: {total_detections}")
            logger.info(
                f"  • Imagens com detecções: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
            logger.info(
                f"  • Detecções/imagem: {total_detections/total_images:.1f}")
            logger.info(f"  • Tempo médio/imagem: {avg_inference_time:.3f}s")
            logger.info(f"  • FPS médio: {1/avg_inference_time:.1f}")

            # Salvar resultados
            save_prediction_results(results, output_dirs, args)

            # Criar visualizações
            visualize_results(predictor, results, output_dirs, args)

            # Salvar resumo
            summary = {
                'model': str(model_path),
                'task': args.task,
                'conf_threshold': args.conf,
                'iou_threshold': args.iou,
                'total_images': total_images,
                'total_detections': total_detections,
                'images_with_detections': images_with_detections,
                'avg_detections_per_image': total_detections / total_images,
                'avg_inference_time': avg_inference_time,
                'avg_fps': 1 / avg_inference_time if avg_inference_time > 0 else 0
            }

            summary_path = output_dir / 'summary.json'
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.success(
                f"✅ Predições concluídas! Resultados salvos em: {output_dir}")
        else:
            logger.warning("⚠️ Nenhuma predição executada")

    except KeyboardInterrupt:
        logger.warning("⚠️ Processo interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erro na predição: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
