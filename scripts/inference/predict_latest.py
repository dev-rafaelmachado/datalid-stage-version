"""
🔮 Script de Predição com Último Modelo
Encontra automaticamente o último modelo treinado e executa predição.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import subprocess

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def find_latest_model(experiments_dir: Path = None) -> Path:
    """
    Encontra o último modelo treinado (best.pt mais recente).

    Args:
        experiments_dir: Diretório de experimentos (padrão: experiments/)

    Returns:
        Path do modelo mais recente

    Raises:
        FileNotFoundError: Se nenhum modelo for encontrado
    """
    if experiments_dir is None:
        experiments_dir = ROOT / "experiments"

    if not experiments_dir.exists():
        raise FileNotFoundError(
            f"Diretório de experimentos não encontrado: {experiments_dir}")

    # Buscar todos os arquivos best.pt
    model_files = list(experiments_dir.glob("*/weights/best.pt"))

    if not model_files:
        raise FileNotFoundError(
            f"Nenhum modelo (best.pt) encontrado em: {experiments_dir}\n"
            "Treine um modelo primeiro com: make train-quick ou make train-nano"
        )

    # Ordenar por data de modificação (mais recente primeiro)
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    latest_model = model_files[0]

    logger.info(
        f"📦 Modelo mais recente encontrado: {latest_model.parent.parent.name}")
    logger.info(f"📅 Modificado: {latest_model.stat().st_mtime}")

    return latest_model


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Predição com último modelo treinado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Entrada (obrigatório)
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Caminho da imagem para predição'
    )

    # Configurações de predição
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0.0-1.0)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold para NMS (0.0-1.0)'
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
        default=True,
        help='Salvar imagens com predições (padrão: True)'
    )

    parser.add_argument(
        '--save-json',
        action='store_true',
        default=True,
        help='Salvar resultados em JSON (padrão: True)'
    )

    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Salvar crops das detecções'
    )

    # Experimentos
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Diretório de experimentos'
    )

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_arguments()

    logger.info("🔮 PREDIÇÃO COM ÚLTIMO MODELO - DATALID 3.0")
    logger.info("=" * 50)

    try:
        # Validar imagem
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"❌ Imagem não encontrada: {image_path}")
            logger.info(f"💡 Caminho absoluto tentado: {image_path.resolve()}")
            sys.exit(1)

        logger.info(f"📸 Imagem: {image_path}")

        # Encontrar último modelo
        experiments_dir = ROOT / args.experiments_dir
        logger.info(f"🔍 Procurando modelos em: {experiments_dir}")

        model_path = find_latest_model(experiments_dir)

        logger.success(f"✅ Modelo encontrado: {model_path}")
        logger.info(f"📁 Experimento: {model_path.parent.parent.name}")

        # Detectar tipo de tarefa pelo nome do modelo
        model_name = model_path.parent.parent.name.lower()
        task = 'segment' if 'seg' in model_name else 'detect'

        logger.info(f"🎯 Tipo de tarefa detectado: {task}")

        # Preparar comando para predict_yolo.py
        predict_script = ROOT / "scripts" / "predict_yolo.py"

        cmd = [
            sys.executable,
            str(predict_script),
            '--model', str(model_path),
            '--image', str(image_path),
            '--task', task,
            '--conf', str(args.conf),
            '--iou', str(args.iou),
            '--output-dir', args.output_dir,
        ]

        if args.save_images:
            cmd.append('--save-images')
        if args.save_json:
            cmd.append('--save-json')
        if args.save_crops:
            cmd.append('--save-crops')

        # Executar predição
        logger.info("🚀 Executando predição...")
        logger.info(f"⚙️ Confidence: {args.conf}")
        logger.info(f"⚙️ IoU: {args.iou}")
        logger.info("")

        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            output_dir = Path(args.output_dir)
            logger.success("✅ Predição concluída com sucesso!")
            logger.info(f"📁 Resultados salvos em: {output_dir}")
            logger.info(f"🖼️ Visualizações: {output_dir / 'images'}")
            logger.info(f"📄 JSON: {output_dir / 'json'}")
            if args.save_crops:
                logger.info(f"✂️ Crops: {output_dir / 'crops'}")

    except FileNotFoundError as e:
        logger.error(f"❌ {str(e)}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro na execução da predição")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("⚠️ Processo interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
