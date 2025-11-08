"""
🔄 Processamento de Dados RAW
Processa dados do Roboflow com divisão customizável.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Agora importar módulos locais
from src.core.constants import CLASS_NAMES
from src.data.converters import convert_polygons_to_bbox
from src.data.validators import (DatasetValidator, get_dataset_stats,
                                 quick_validate)


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Processar dados RAW do Roboflow")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Caminho dos dados RAW")
    parser.add_argument("--output", "-o", type=str, default="data/processed",
                        help="Caminho de saída (default: data/processed)")

    # Divisão customizável
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Proporção de treino (default: 0.7)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Proporção de validação (default: 0.2)")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Proporção de teste (default: 0.1)")

    # Processamento
    parser.add_argument("--task", choices=["detect", "segment", "both"],
                        default="segment", help="Tipo de processamento (default: segment)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para reprodutibilidade")
    parser.add_argument("--copy-mode", choices=["copy", "move", "symlink"],
                        default="copy", help="Modo de transferência de arquivos")

    # Validação
    parser.add_argument("--validate", action="store_true",
                        help="Validar dataset após processamento")
    parser.add_argument("--preview", action="store_true",
                        help="Mostrar preview antes de processar")

    return parser.parse_args()


def validate_splits(train: float, val: float, test: float) -> None:
    """Valida proporções de divisão."""
    total = train + val + test
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Soma das proporções deve ser 1.0, atual: {total:.3f}")

    if any(split <= 0 for split in [train, val, test]):
        raise ValueError("Todas as proporções devem ser > 0")

    logger.info(
        f"📊 Divisão: Treino={train:.1%}, Val={val:.1%}, Teste={test:.1%}")


def discover_raw_data(input_path: Path) -> Dict[str, List[Path]]:
    """Descobre estrutura dos dados RAW."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {input_path}")

    logger.info(f"🔍 Descobrindo estrutura em: {input_path}")

    discovered = {}

    # Verificar estrutura Roboflow padrão (train/images/, train/labels/)
    train_images_path = input_path / "train" / "images"
    if train_images_path.exists():
        logger.info("📁 Detectada estrutura Roboflow padrão")
        images = list(train_images_path.glob("*.jpg")) + \
            list(train_images_path.glob("*.png"))
        if images:
            discovered["train"] = images
            logger.info(f"  ✅ Train: {len(images)} imagens encontradas")

            # Verificar se existem labels correspondentes
            labels_path = input_path / "train" / "labels"
            if labels_path.exists():
                labels_count = len(list(labels_path.glob("*.txt")))
                logger.info(
                    f"  📄 Labels: {labels_count} arquivos .txt encontrados")
            else:
                logger.warning("  ⚠️ Pasta de labels não encontrada!")

    # Verificar outras estruturas possíveis
    else:
        # Estruturas possíveis do Roboflow
        possible_structures = [
            # Estrutura padrão com divisões
            {"train": "train", "valid": "valid", "test": "test"},
            {"train": "train", "val": "val", "test": "test"},

            # Estrutura sem divisões (tudo junto)
            {"all": "."},

            # Estrutura com imagens e labels separadas
            {"images": "images", "labels": "labels"},
        ]

        for structure in possible_structures:
            found_all = True
            temp_discovered = {}

            for key, folder in structure.items():
                folder_path = input_path / folder
                if folder_path.exists():
                    # Procurar imagens
                    images = list(folder_path.glob("*.jpg")) + \
                        list(folder_path.glob("*.png"))
                    if images:
                        temp_discovered[key] = images
                    else:
                        # Procurar em subpastas
                        for subfolder in ["images", "."]:
                            subfolder_path = folder_path / subfolder
                            if subfolder_path.exists():
                                images = list(subfolder_path.glob(
                                    "*.jpg")) + list(subfolder_path.glob("*.png"))
                                if images:
                                    temp_discovered[key] = images
                                    break
                        else:
                            found_all = False
                            break
                else:
                    found_all = False
                    break

            if found_all:
                discovered = temp_discovered
                break

    # Fallback: buscar todas as imagens recursivamente
    if not discovered:
        logger.info("🔍 Fallback: buscando imagens recursivamente")
        all_images = list(input_path.rglob("*.jpg")) + \
            list(input_path.rglob("*.png"))
        if all_images:
            discovered = {"all": all_images}
            logger.info(f"  ✅ Encontradas {len(all_images)} imagens no total")

    if not discovered:
        raise ValueError(f"Nenhuma imagem encontrada em: {input_path}")

    return discovered


def get_label_path(image_path: Path, input_path: Path) -> Optional[Path]:
    """Encontra arquivo de label correspondente à imagem."""
    # Nome do arquivo label (mesmo stem da imagem)
    label_filename = f"{image_path.stem}.txt"

    # Locais possíveis para labels, ordenados por prioridade
    possible_label_paths = [
        # 1. Estrutura Roboflow padrão: train/labels/arquivo.txt
        # Se a imagem está em train/images, label está em train/labels
        image_path.parent.parent / "labels" / label_filename,

        # 2. Estrutura com valid ou test
        image_path.parent.parent / "labels" / label_filename,

        # 3. Pasta labels no mesmo nível da pasta images atual
        image_path.parent / "labels" / label_filename,

        # 4. Mesmo diretório, extensão .txt
        image_path.with_suffix('.txt'),

        # 5. Labels na raiz do input_path
        input_path / "labels" / label_filename,

        # 6. Labels em train/labels (estrutura Roboflow)
        input_path / "train" / "labels" / label_filename,

        # 7. Labels em valid/labels
        input_path / "valid" / "labels" / label_filename,

        # 8. Labels em test/labels
        input_path / "test" / "labels" / label_filename,

        # 9. Buscar recursivamente no input_path
        # (será feito se nenhum dos anteriores funcionar)
    ]

    # Tentar os caminhos diretos primeiro
    for label_path in possible_label_paths:
        if label_path.exists() and label_path.is_file():
            return label_path

    # Fallback: buscar recursivamente
    for label_path in input_path.rglob(label_filename):
        if label_path.is_file():
            logger.debug(f"Label encontrado via busca recursiva: {label_path}")
            return label_path

    # Debug: mostrar onde procurou (somente se não encontrar e em modo debug)
    logger.trace(f"Label não encontrado para {image_path.name}. Procurado em:")
    for lp in possible_label_paths[:4]:  # Mostrar apenas os 4 primeiros
        logger.trace(f"  - {lp} (existe: {lp.exists()})")

    return None


def split_data(
    all_images: List[Path],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """Divide dados em treino, validação e teste."""
    random.seed(seed)
    images = all_images.copy()
    random.shuffle(images)

    total = len(images)
    train_count = int(total * train_split)
    val_count = int(total * val_split)
    test_count = total - train_count - val_count

    splits = {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:]
    }

    logger.info(f"📊 Divisão final:")
    logger.info(
        f"  • Treino: {len(splits['train'])} imagens ({len(splits['train'])/total:.1%})")
    logger.info(
        f"  • Validação: {len(splits['val'])} imagens ({len(splits['val'])/total:.1%})")
    logger.info(
        f"  • Teste: {len(splits['test'])} imagens ({len(splits['test'])/total:.1%})")

    return splits


def copy_files(
    src_path: Path,
    dst_path: Path,
    mode: str = "copy"
) -> None:
    """Copia arquivo com modo especificado."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src_path, dst_path)
    elif mode == "move":
        shutil.move(str(src_path), str(dst_path))
    elif mode == "symlink":
        if dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_path.absolute())


def process_dataset(
    splits: Dict[str, List[Path]],
    input_path: Path,
    output_path: Path,
    task: str,
    copy_mode: str
) -> Dict[str, Path]:
    """Processa dataset para formato YOLO."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    created_datasets = {}

    # Processar cada tipo de tarefa
    tasks_to_process = []
    if task in ["detect", "both"]:
        tasks_to_process.append("detect")
    if task in ["segment", "both"]:
        tasks_to_process.append("segment")

    for task_type in tasks_to_process:
        task_output = output_path / f"v1_{task_type}"
        logger.info(f"🔄 Processando {task_type} em: {task_output}")

        # Criar estrutura de diretórios
        for split_name in splits.keys():
            (task_output / split_name / "images").mkdir(parents=True, exist_ok=True)
            (task_output / split_name / "labels").mkdir(parents=True, exist_ok=True)

        # Processar cada split
        for split_name, images in splits.items():
            logger.info(f"  📁 Processando {split_name}: {len(images)} imagens")

            processed_count = 0
            skipped_count = 0
            skipped_images = []

            for image_path in images:
                # Copiar imagem
                dst_image = task_output / split_name / "images" / image_path.name
                copy_files(image_path, dst_image, copy_mode)

                # Encontrar e processar label
                label_path = get_label_path(image_path, input_path)

                if label_path:
                    # Verificar se label tem conteúdo
                    if label_path.stat().st_size == 0:
                        logger.debug(
                            f"    ⚠️ Label vazio ignorado: {image_path.name}")
                        skipped_count += 1
                        skipped_images.append(image_path.name)
                        continue

                    dst_label = task_output / split_name / \
                        "labels" / f"{image_path.stem}.txt"

                    if task_type == "detect":
                        # Converter polígonos para bbox se necessário
                        convert_polygons_to_bbox(label_path, dst_label)
                    else:  # segment
                        # Copiar labels de segmentação diretamente (já estão no formato correto do Roboflow)
                        copy_files(label_path, dst_label, copy_mode)

                    processed_count += 1
                else:
                    skipped_count += 1
                    skipped_images.append(image_path.name)
                    logger.debug(
                        f"    ⚠️ Label não encontrado: {image_path.name}")

            # Logging melhorado
            logger.info(
                f"    ✅ Processadas: {processed_count}/{len(images)} ({processed_count/len(images)*100:.1f}%)")

            if skipped_count > 0:
                logger.warning(
                    f"    ⚠️ Ignoradas: {skipped_count}/{len(images)} ({skipped_count/len(images)*100:.1f}%)")

                # Mostrar até 5 exemplos de imagens sem labels
                if skipped_images:
                    logger.warning(f"    📋 Exemplos de imagens sem labels:")
                    for img in skipped_images[:5]:
                        logger.warning(f"       - {img}")
                    if len(skipped_images) > 5:
                        logger.warning(
                            f"       ... e mais {len(skipped_images) - 5} imagens")

        # Criar data.yaml
        create_data_yaml(task_output, task_type)
        created_datasets[task_type] = task_output

    return created_datasets


def create_data_yaml(dataset_path: Path, task_type: str) -> Path:
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
        'created_by': 'datalid_processing_script',
        'task_type': task_type,
        'processed_date': str(Path().cwd()),
        'description': f'Dataset processado para {task_type} de datas de validade'
    }

    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False,
                  allow_unicode=True, indent=2)

    logger.success(f"✅ Criado data.yaml: {data_yaml_path}")
    return data_yaml_path


def preview_processing(discovered: Dict[str, List[Path]], splits: Dict[str, List[Path]]) -> None:
    """Mostra preview do processamento."""
    logger.info("👀 PREVIEW DO PROCESSAMENTO:")
    logger.info("=" * 50)

    # Dados descobertos
    logger.info("📁 DADOS DESCOBERTOS:")
    for key, images in discovered.items():
        logger.info(f"  • {key}: {len(images)} imagens")

    # Divisões
    logger.info("\n📊 DIVISÕES PLANEJADAS:")
    total_images = sum(len(images) for images in splits.values())
    for split_name, images in splits.items():
        percentage = len(images) / total_images * 100
        logger.info(
            f"  • {split_name}: {len(images)} imagens ({percentage:.1f}%)")

    # Confirmar
    response = input("\n❓ Continuar com o processamento? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        logger.info("❌ Processamento cancelado pelo usuário")
        exit(0)


def main():
    """Função principal."""
    args = parse_arguments()

    logger.info("🔄 PROCESSAMENTO DE DADOS RAW")
    logger.info("=" * 50)

    # Validar argumentos
    validate_splits(args.train_split, args.val_split, args.test_split)

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        # 1. Descobrir estrutura dos dados
        logger.info(f"🔍 Descobrindo dados em: {input_path}")
        discovered = discover_raw_data(input_path)

        # 2. Consolidar todas as imagens
        all_images = []
        for images_list in discovered.values():
            all_images.extend(images_list)

        # Remover duplicatas
        all_images = list(set(all_images))
        logger.info(f"📊 Total de imagens encontradas: {len(all_images)}")

        # 3. Dividir dados
        logger.info("🔀 Dividindo dados...")
        splits = split_data(
            all_images,
            args.train_split,
            args.val_split,
            args.test_split,
            args.seed
        )

        # 4. Preview se solicitado
        if args.preview:
            preview_processing(discovered, splits)

        # 5. Processar dataset
        logger.info(f"⚙️ Processando para: {output_path}")
        created_datasets = process_dataset(
            splits,
            input_path,
            output_path,
            args.task,
            args.copy_mode
        )

        # 6. Validar se solicitado
        if args.validate:
            logger.info("🔍 Validando datasets criados...")
            for task_type, dataset_path in created_datasets.items():
                logger.info(f"  📊 Validando {task_type}...")
                is_valid = quick_validate(str(dataset_path))
                if is_valid:
                    logger.success(f"    ✅ {task_type} válido")
                else:
                    logger.error(f"    ❌ {task_type} com problemas")

        # Resumo final
        logger.success("🎉 PROCESSAMENTO CONCLUÍDO!")
        logger.info("\n📊 DATASETS CRIADOS:")
        for task_type, dataset_path in created_datasets.items():
            logger.info(f"  • {task_type.upper()}: {dataset_path}")
            data_yaml = dataset_path / "data.yaml"
            if data_yaml.exists():
                logger.info(f"    📄 Config: {data_yaml}")

        logger.info(f"\n🚀 Para treinar, use:")
        for task_type, dataset_path in created_datasets.items():
            logger.info(
                f"  python scripts/train_yolo.py --preset {task_type}_small --data {dataset_path}")

    except Exception as e:
        logger.error(f"❌ Erro no processamento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
