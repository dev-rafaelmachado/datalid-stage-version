"""
📊 Processamento de Dados com Fração
Cria múltiplos datasets com diferentes porcentagens dos dados para análise de curva de aprendizado.
"""
import sys
# Adicionar src ao path
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import random
import shutil
from typing import Dict, List, Optional, Tuple

import yaml
from loguru import logger

from src.core.constants import CLASS_NAMES, IMAGE_EXTENSIONS
from src.data.validators import quick_validate


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Processa dados com diferentes frações para análise de curva de aprendizado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-data',
        type=str,
        required=True,
        help='Caminho do dataset base (completo) já processado'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/fractions',
        help='Diretório de saída para datasets fracionados'
    )

    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=[0.25, 0.50, 0.75, 1.0],
        help='Frações dos dados para criar (ex: 0.25 0.50 0.75 1.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validar datasets criados'
    )

    return parser.parse_args()


def discover_dataset_structure(base_path: Path) -> Dict[str, List[Path]]:
    """Descobre estrutura do dataset base."""
    structure = {
        'train': [],
        'val': [],
        'test': []
    }

    for split in ['train', 'val', 'test']:
        images_dir = base_path / split / 'images'
        if images_dir.exists():
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(images_dir.glob(f'*{ext}'))
            structure[split] = sorted(images)

    return structure


def get_label_path(image_path: Path) -> Optional[Path]:
    """Retorna caminho do label correspondente."""
    labels_dir = image_path.parent.parent / 'labels'
    label_path = labels_dir / f"{image_path.stem}.txt"
    return label_path if label_path.exists() else None


def create_fractional_dataset(
    base_structure: Dict[str, List[Path]],
    output_path: Path,
    fraction: float,
    seed: int,
    task_type: str
) -> None:
    """
    Cria dataset com fração específica dos dados.

    Args:
        base_structure: Estrutura do dataset base
        output_path: Caminho de saída
        fraction: Fração dos dados (0.0-1.0)
        seed: Seed para aleatoriedade
        task_type: Tipo de tarefa (detect ou segment)
    """
    random.seed(seed)

    logger.info(f"📊 Criando dataset com {fraction*100:.0f}% dos dados...")

    # Criar estrutura de diretórios
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    total_copied = 0

    # Processar cada split
    for split, images in base_structure.items():
        if not images:
            continue

        # Calcular quantas imagens copiar
        total_images = len(images)

        # Para train, aplicar fração. Val e test ficam completos
        if split == 'train':
            num_to_copy = max(1, int(total_images * fraction))
        else:
            num_to_copy = total_images

        # Selecionar aleatoriamente
        selected_images = random.sample(images, num_to_copy)

        logger.info(f"  • {split}: {num_to_copy}/{total_images} imagens")

        # Copiar imagens e labels
        for img_path in selected_images:
            # Copiar imagem
            dest_img = output_path / split / 'images' / img_path.name
            shutil.copy2(img_path, dest_img)

            # Copiar label
            label_path = get_label_path(img_path)
            if label_path:
                dest_label = output_path / split / 'labels' / label_path.name
                shutil.copy2(label_path, dest_label)

            total_copied += 1

    # Criar data.yaml
    create_data_yaml(output_path, task_type, fraction)

    logger.success(f"✅ Dataset criado: {total_copied} arquivos copiados")


def create_data_yaml(dataset_path: Path, task_type: str, fraction: float) -> Path:
    """Cria arquivo data.yaml para o dataset."""
    data_yaml_path = dataset_path / "data.yaml"

    data_config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    # Adicionar task para segmentação
    if task_type == "segment":
        data_config['task'] = 'segment'

    # Adicionar metadata
    data_config['metadata'] = {
        'created_by': 'datalid_fraction_script',
        'task_type': task_type,
        'fraction': fraction,
        'description': f'Dataset com {fraction*100:.0f}% dos dados de treino para análise de curva de aprendizado'
    }

    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False,
                  allow_unicode=True, indent=2)

    logger.success(f"✅ Criado data.yaml: {data_yaml_path}")
    return data_yaml_path


def detect_task_type(base_path: Path) -> str:
    """Detecta tipo de tarefa do dataset base."""
    data_yaml = base_path / 'data.yaml'

    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('task', 'detect')

    # Se não tem data.yaml, verificar se tem polígonos nas labels
    train_labels = base_path / 'train' / 'labels'
    if train_labels.exists():
        sample_label = next(train_labels.glob('*.txt'), None)
        if sample_label:
            with open(sample_label, 'r') as f:
                first_line = f.readline().strip()
                # Se tem mais de 5 valores, provavelmente é segmentação
                values = first_line.split()
                return 'segment' if len(values) > 5 else 'detect'

    return 'detect'


def main():
    """Função principal."""
    args = parse_arguments()

    logger.info("📊 PROCESSAMENTO COM FRAÇÕES - CURVA DE APRENDIZADO")
    logger.info("=" * 60)

    # Validar paths
    base_path = Path(args.base_data)
    output_dir = Path(args.output_dir)

    if not base_path.exists():
        logger.error(f"❌ Dataset base não encontrado: {base_path}")
        sys.exit(1)

    # Detectar tipo de tarefa
    task_type = detect_task_type(base_path)
    logger.info(f"🎯 Tipo de tarefa detectado: {task_type}")

    # Descobrir estrutura do dataset base
    logger.info(f"🔍 Analisando dataset base: {base_path}")
    structure = discover_dataset_structure(base_path)

    # Log estatísticas
    total_train = len(structure['train'])
    total_val = len(structure['val'])
    total_test = len(structure['test'])

    logger.info(f"📊 Dataset base:")
    logger.info(f"  • Train: {total_train} imagens")
    logger.info(f"  • Val: {total_val} imagens")
    logger.info(f"  • Test: {total_test} imagens")

    if total_train == 0:
        logger.error("❌ Nenhuma imagem de treino encontrada")
        sys.exit(1)

    # Validar frações
    fractions = sorted([f for f in args.fractions if 0.0 < f <= 1.0])
    if not fractions:
        logger.error(
            "❌ Nenhuma fração válida fornecida (valores entre 0.0 e 1.0)")
        sys.exit(1)

    logger.info(
        f"📈 Criando datasets para frações: {[f'{f*100:.0f}%' for f in fractions]}")

    # Criar datasets para cada fração
    created_datasets = {}

    for fraction in fractions:
        fraction_name = f"fraction_{int(fraction*100):02d}"
        output_path = output_dir / fraction_name

        logger.info(f"\n{'='*60}")
        logger.info(
            f"📦 Processando fração: {fraction*100:.0f}% ({fraction_name})")
        logger.info(f"📁 Saída: {output_path}")

        create_fractional_dataset(
            structure,
            output_path,
            fraction,
            args.seed,
            task_type
        )

        created_datasets[fraction] = output_path

    # Validar datasets se solicitado
    if args.validate:
        logger.info(f"\n{'='*60}")
        logger.info("🔍 Validando datasets criados...")

        for fraction, dataset_path in created_datasets.items():
            logger.info(f"  📊 Validando {fraction*100:.0f}%...")
            is_valid = quick_validate(str(dataset_path))
            if is_valid:
                logger.success(f"    ✅ Válido")
            else:
                logger.warning(f"    ⚠️ Problemas encontrados")

    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.success("🎉 PROCESSAMENTO CONCLUÍDO!")
    logger.info(f"\n📊 DATASETS CRIADOS ({len(created_datasets)}):")

    for fraction, dataset_path in created_datasets.items():
        train_images = len(
            list((dataset_path / 'train' / 'images').glob('*.*')))
        logger.info(f"  • {fraction*100:5.0f}%: {dataset_path}")
        logger.info(f"           {train_images} imagens de treino")

    logger.info(f"\n🚀 Para treinar com cada fração, use:")
    logger.info(
        f"  make train-fraction-nano FRACTION=25  # {fractions[0]*100:.0f}%")
    logger.info(f"  make train-fraction-small FRACTION=50  # Próxima fração")
    logger.info(f"  make train-fraction-medium FRACTION=75  # etc...")

    logger.info(f"\n📈 Para comparar curvas de aprendizado:")
    logger.info(
        f"  python scripts/compare_learning_curves.py --fractions-dir {output_dir}")


if __name__ == "__main__":
    main()
