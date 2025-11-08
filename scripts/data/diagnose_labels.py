"""
🔍 Diagnóstico de Labels
Verifica integridade dos labels do Roboflow e identifica problemas.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def analyze_label_file(label_path: Path) -> Dict:
    """Analisa um arquivo de label."""
    result = {
        'valid': True,
        'empty': False,
        'format': 'unknown',
        'objects_count': 0,
        'errors': []
    }

    # Verificar se arquivo está vazio
    if label_path.stat().st_size == 0:
        result['empty'] = True
        result['valid'] = False
        result['errors'].append("Arquivo vazio")
        return result

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                result['valid'] = False
                result['errors'].append(
                    f"Linha {line_num}: Formato inválido (< 5 valores)")
                continue

            # Verificar class_id
            try:
                class_id = int(parts[0])
            except ValueError:
                result['valid'] = False
                result['errors'].append(f"Linha {line_num}: class_id inválido")
                continue

            # Verificar coordenadas
            coords = parts[1:]

            # Determinar formato
            if len(coords) == 4:
                result['format'] = 'bbox'
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                result['format'] = 'polygon'
            else:
                result['format'] = 'unknown'
                result['valid'] = False
                result['errors'].append(
                    f"Linha {line_num}: Formato desconhecido ({len(coords)} coords)")

            result['objects_count'] += 1

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Erro ao ler arquivo: {str(e)}")

    return result


def analyze_dataset(dataset_path: Path) -> Dict:
    """Analisa um dataset completo."""
    dataset_path = Path(dataset_path)

    logger.info(f"🔍 Analisando dataset: {dataset_path}")

    results = {
        'total_images': 0,
        'total_labels': 0,
        'empty_labels': 0,
        'invalid_labels': 0,
        'format_bbox': 0,
        'format_polygon': 0,
        'format_unknown': 0,
        'missing_labels': [],
        'empty_label_files': [],
        'invalid_label_files': [],
        'splits': {}
    }

    # Verificar estrutura
    for split in ['train', 'valid', 'test', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'

        if not images_dir.exists():
            continue

        logger.info(f"\n📁 Analisando split: {split}")

        split_results = {
            'images': 0,
            'labels': 0,
            'empty': 0,
            'invalid': 0,
            'bbox': 0,
            'polygon': 0,
            'unknown': 0,
            'missing': []
        }

        # Listar imagens
        images = list(images_dir.glob('*.jpg')) + \
            list(images_dir.glob('*.png'))
        split_results['images'] = len(images)
        results['total_images'] += len(images)

        logger.info(f"  📸 {len(images)} imagens encontradas")

        # Verificar labels
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                split_results['missing'].append(image_path.name)
                results['missing_labels'].append(str(label_path))
                continue

            split_results['labels'] += 1
            results['total_labels'] += 1

            # Analisar label
            analysis = analyze_label_file(label_path)

            if analysis['empty']:
                split_results['empty'] += 1
                results['empty_labels'] += 1
                results['empty_label_files'].append(str(label_path))

            if not analysis['valid']:
                split_results['invalid'] += 1
                results['invalid_labels'] += 1
                results['invalid_label_files'].append({
                    'path': str(label_path),
                    'errors': analysis['errors']
                })

            if analysis['format'] == 'bbox':
                split_results['bbox'] += 1
                results['format_bbox'] += 1
            elif analysis['format'] == 'polygon':
                split_results['polygon'] += 1
                results['format_polygon'] += 1
            else:
                split_results['unknown'] += 1
                results['format_unknown'] += 1

        # Relatório do split
        logger.info(f"  🏷️  {split_results['labels']} labels encontrados")
        if split_results['missing']:
            logger.warning(
                f"  ⚠️  {len(split_results['missing'])} labels faltando")
        if split_results['empty']:
            logger.warning(f"  ⚠️  {split_results['empty']} labels vazios")
        if split_results['invalid']:
            logger.error(f"  ❌ {split_results['invalid']} labels inválidos")

        logger.info(
            f"  📊 Formatos: bbox={split_results['bbox']}, polygon={split_results['polygon']}")

        results['splits'][split] = split_results

    return results


def print_summary(results: Dict):
    """Imprime resumo da análise."""
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMO DA ANÁLISE")
    logger.info("="*60)

    logger.info(f"\n📸 Total de imagens: {results['total_images']}")
    logger.info(f"🏷️  Total de labels: {results['total_labels']}")

    # Calcular porcentagem
    if results['total_images'] > 0:
        coverage = results['total_labels'] / results['total_images'] * 100
        logger.info(f"📈 Cobertura: {coverage:.1f}%")

    # Problemas
    if results['empty_labels'] > 0:
        logger.warning(f"\n⚠️  Labels vazios: {results['empty_labels']}")
        logger.info("   Arquivos vazios serão ignorados no processamento")

    if results['invalid_labels'] > 0:
        logger.error(f"\n❌ Labels inválidos: {results['invalid_labels']}")
        logger.info("   Verifique os arquivos abaixo:")
        for item in results['invalid_label_files'][:5]:  # Mostrar até 5
            logger.error(f"   - {item['path']}")
            for error in item['errors'][:2]:  # Mostrar até 2 erros
                logger.error(f"     • {error}")
        if len(results['invalid_label_files']) > 5:
            logger.info(
                f"   ... e mais {len(results['invalid_label_files']) - 5} arquivos")

    if results['missing_labels']:
        logger.warning(
            f"\n⚠️  Labels faltando: {len(results['missing_labels'])}")
        logger.info("   Exemplos:")
        for label in results['missing_labels'][:5]:
            logger.warning(f"   - {label}")
        if len(results['missing_labels']) > 5:
            logger.info(
                f"   ... e mais {len(results['missing_labels']) - 5} arquivos")

    # Formatos
    logger.info("\n📊 FORMATOS DE LABELS:")
    logger.info(f"   📦 Bounding Box: {results['format_bbox']}")
    logger.info(f"   🔺 Polígono: {results['format_polygon']}")
    if results['format_unknown'] > 0:
        logger.warning(f"   ❓ Desconhecido: {results['format_unknown']}")

    # Recomendação
    if results['format_polygon'] > results['format_bbox']:
        logger.success(f"\n✅ Dataset predominantemente em formato POLIGONAL")
        logger.info("   👍 Ideal para treinamento de segmentação")
        logger.info("   💡 Use: make process INPUT=<caminho>")
    elif results['format_bbox'] > results['format_polygon']:
        logger.info(f"\n📦 Dataset predominantemente em formato BBOX")
        logger.info("   💡 Use: make process-detect INPUT=<caminho>")

    # Status geral
    logger.info("\n" + "="*60)
    if results['invalid_labels'] == 0 and len(results['missing_labels']) < results['total_images'] * 0.05:
        logger.success("✅ Dataset em boas condições!")
        logger.info("   Você pode prosseguir com o processamento")
    else:
        logger.warning("⚠️  Dataset com alguns problemas")
        logger.info("   Revise os arquivos listados acima")
    logger.info("="*60)


def main():
    """Função principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnosticar labels do dataset")
    parser.add_argument('dataset_path', type=str, help='Caminho do dataset')
    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Modo verboso')

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        logger.error(f"❌ Caminho não encontrado: {dataset_path}")
        sys.exit(1)

    logger.info("🔍 DIAGNÓSTICO DE LABELS")
    logger.info("="*60)

    results = analyze_dataset(dataset_path)
    print_summary(results)


if __name__ == "__main__":
    main()
