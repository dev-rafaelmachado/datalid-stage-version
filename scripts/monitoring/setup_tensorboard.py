"""
📊 Setup TensorBoard para YOLO
Configura e converte logs YOLO para TensorBoard.
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    logger.warning(
        "TensorBoard não disponível. Instale com: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False


def convert_yolo_results_to_tensorboard(experiment_dir: Path, force: bool = False):
    """
    Converte results.csv do YOLO para logs do TensorBoard.

    Args:
        experiment_dir: Diretório do experimento YOLO
        force: Se True, reconverte mesmo que já exista
    """
    if not TENSORBOARD_AVAILABLE:
        logger.error("TensorBoard não instalado!")
        return False

    results_csv = experiment_dir / "results.csv"

    if not results_csv.exists():
        logger.warning(f"results.csv não encontrado em {experiment_dir}")
        return False

    # Verificar se já existe conversão
    log_dir = experiment_dir / "tensorboard_logs"
    if log_dir.exists() and any(log_dir.iterdir()) and not force:
        logger.info(
            f"⏭️  Logs do TensorBoard já existem para: {experiment_dir.name} (use --force para reconverter)")
        return False

    try:
        # Ler CSV com resultados do treinamento
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remover espaços

        # Criar writer do TensorBoard
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        logger.info(
            f"📊 Convertendo resultados para TensorBoard: {experiment_dir.name}")

        # Mapear todas as colunas para TensorBoard
        metrics_map = {
            'train/box_loss': 'Loss/train_box',
            'train/seg_loss': 'Loss/train_seg',
            'train/cls_loss': 'Loss/train_cls',
            'train/dfl_loss': 'Loss/train_dfl',
            'metrics/precision(B)': 'Metrics/precision_B',
            'metrics/recall(B)': 'Metrics/recall_B',
            'metrics/mAP50(B)': 'Metrics/mAP50_B',
            'metrics/mAP50-95(B)': 'Metrics/mAP50-95_B',
            'metrics/precision(M)': 'Metrics/precision_M',
            'metrics/recall(M)': 'Metrics/recall_M',
            'metrics/mAP50(M)': 'Metrics/mAP50_M',
            'metrics/mAP50-95(M)': 'Metrics/mAP50-95_M',
            'val/box_loss': 'Loss/val_box',
            'val/seg_loss': 'Loss/val_seg',
            'val/cls_loss': 'Loss/val_cls',
            'val/dfl_loss': 'Loss/val_dfl',
            'lr/pg0': 'LR/pg0',
            'lr/pg1': 'LR/pg1',
            'lr/pg2': 'LR/pg2',
            'epoch': 'Meta/epoch',
            'time': 'Meta/time',
        }

        # Converter cada época
        for idx, row in df.iterrows():
            epoch = int(row['epoch']) if 'epoch' in df.columns else idx
            for yolo_col, tb_name in metrics_map.items():
                if yolo_col in df.columns:
                    value = row[yolo_col]
                    if pd.notna(value):  # Ignorar NaN
                        writer.add_scalar(tb_name, float(value), epoch)

        # Adicionar informações do experimento
        writer.add_text('Config/experiment_name', experiment_dir.name)

        writer.close()
        logger.success(f"✅ Logs do TensorBoard criados em: {log_dir}")
        return True

    except Exception as e:
        logger.error(f"❌ Erro ao converter logs: {e}")
        return False


def setup_all_experiments(force: bool = False):
    """
    Converte todos os experimentos existentes.

    Args:
        force: Se True, reconverte mesmo que já existam logs
    """
    experiments_dir = ROOT_DIR / "experiments"

    if not experiments_dir.exists():
        logger.warning("Diretório de experimentos não encontrado!")
        return

    converted = 0
    skipped = 0
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "results.csv").exists():
            result = convert_yolo_results_to_tensorboard(exp_dir, force=force)
            if result:
                converted += 1
            elif not force:
                skipped += 1

    logger.info(f"✨ Convertidos {converted} experimentos para TensorBoard")
    if skipped > 0:
        logger.info(f"⏭️  Ignorados {skipped} experimentos (já convertidos)")
    logger.info("📊 Execute: make tensorboard")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup TensorBoard para YOLO")
    parser.add_argument(
        '--experiment',
        type=str,
        help='Nome do experimento específico (ou todos se não especificado)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Força reconversão mesmo que os logs já existam'
    )

    args = parser.parse_args()

    if args.experiment:
        exp_dir = ROOT_DIR / "experiments" / args.experiment
        if exp_dir.exists():
            convert_yolo_results_to_tensorboard(exp_dir, force=args.force)
        else:
            logger.error(f"Experimento não encontrado: {args.experiment}")
    else:
        setup_all_experiments(force=args.force)
