"""
✅ Script de Validação de Dataset
Valida integridade e formato de datasets YOLO.
"""

import argparse
import json
import sys
from pathlib import Path

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

from src.data.validators import DatasetValidator


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Valida dataset YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Caminho do dataset YOLO (pasta com data.yaml)'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        help='Salvar relatório em arquivo JSON'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Mostrar informações detalhadas'
    )
    
    parser.add_argument(
        '--fix-issues',
        action='store_true',
        help='Tentar corrigir problemas automaticamente (experimental)'
    )
    
    return parser.parse_args()


def print_validation_summary(result: dict, detailed: bool = False):
    """Imprime resumo da validação."""
    
    logger.info("📊 RESUMO DA VALIDAÇÃO")
    logger.info("=" * 40)
    
    # Status geral
    if result['valid']:
        logger.success("✅ Dataset VÁLIDO")
    else:
        logger.error("❌ Dataset INVÁLIDO")
    
    # Estatísticas gerais
    stats = result['stats']
    logger.info(f"\n📈 ESTATÍSTICAS GERAIS:")
    logger.info(f"  • Total de imagens: {stats['total_images']}")
    logger.info(f"  • Total de labels: {stats['total_labels']}")
    
    # Informações por split
    logger.info(f"\n📁 INFORMAÇÕES POR SPLIT:")
    for split_name, split_info in stats['splits'].items():
        logger.info(f"  📂 {split_name.upper()}:")
        logger.info(f"    • Imagens: {split_info['image_count']}")
        logger.info(f"    • Labels: {split_info['label_count']}")
        
        if detailed:
            if split_info['orphaned_images']:
                logger.warning(f"    • Imagens órfãs: {len(split_info['orphaned_images'])}")
            if split_info['orphaned_labels']:
                logger.warning(f"    • Labels órfãos: {len(split_info['orphaned_labels'])}")
            if split_info['corrupted_images']:
                logger.error(f"    • Imagens corrompidas: {len(split_info['corrupted_images'])}")
            if split_info['invalid_labels']:
                logger.error(f"    • Labels inválidos: {len(split_info['invalid_labels'])}")
            
            # Estatísticas de classes
            if split_info['label_stats']:
                most_common = split_info['label_stats'].most_common(3)
                logger.info(f"    • Classes mais comuns: {most_common}")
    
    # Data.yaml info
    if 'data_yaml' in stats and stats['data_yaml']:
        data_yaml = stats['data_yaml']
        logger.info(f"\n📄 DATA.YAML:")
        logger.info(f"  • Número de classes: {data_yaml.get('nc', 'N/A')}")
        logger.info(f"  • Nomes das classes: {data_yaml.get('names', 'N/A')}")
        if 'task' in data_yaml:
            logger.info(f"  • Tipo de tarefa: {data_yaml['task']}")
    
    # Erros
    if result['errors']:
        logger.error(f"\n❌ ERROS ENCONTRADOS ({len(result['errors'])}):")
        for i, error in enumerate(result['errors'], 1):
            logger.error(f"  {i}. {error}")
    
    # Avisos
    if result['warnings']:
        logger.warning(f"\n⚠️ AVISOS ({len(result['warnings'])}):")
        for i, warning in enumerate(result['warnings'], 1):
            logger.warning(f"  {i}. {warning}")
    
    # Recomendações
    if not result['valid']:
        logger.info(f"\n💡 RECOMENDAÇÕES:")
        logger.info("  1. Corrija os erros listados acima")
        logger.info("  2. Verifique se todas as imagens estão íntegras")
        logger.info("  3. Confirme que os labels seguem o formato YOLO")
        logger.info("  4. Execute novamente após as correções")


def save_report(result: dict, output_path: str):
    """Salva relatório em arquivo JSON."""
    try:
        # Converter Path objects para strings para JSON
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        serializable_result = convert_paths(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.success(f"📄 Relatório salvo em: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro salvando relatório: {str(e)}")


def suggest_fixes(result: dict):
    """Sugere correções para problemas encontrados."""
    logger.info("\n🔧 SUGESTÕES DE CORREÇÃO:")
    
    fixes = []
    
    # Verificar problemas comuns
    for error in result['errors']:
        if "não encontrado" in error.lower():
            fixes.append("• Verifique se todos os arquivos e diretórios necessários existem")
        elif "corrompida" in error.lower():
            fixes.append("• Remova ou substitua imagens corrompidas")
        elif "inválido" in error.lower():
            fixes.append("• Verifique formato dos labels (deve ser: class x_center y_center width height)")
        elif "inconsistência" in error.lower():
            fixes.append("• Ajuste o campo 'nc' em data.yaml para corresponder ao número de classes")
    
    for warning in result['warnings']:
        if "órfã" in warning.lower():
            fixes.append("• Remova imagens sem labels ou crie labels para imagens órfãs")
        elif "caminho" in warning.lower():
            fixes.append("• Ajuste os caminhos em data.yaml para apontar para diretórios corretos")
    
    # Remover duplicatas e imprimir
    unique_fixes = list(set(fixes))
    for fix in unique_fixes:
        logger.info(f"  {fix}")
    
    if not unique_fixes:
        logger.info("  • Nenhuma sugestão específica disponível")
        logger.info("  • Verifique os erros listados acima manualmente")


def main():
    """Função principal."""
    args = parse_arguments()
    
    logger.info("✅ VALIDAÇÃO DE DATASET YOLO")
    logger.info("=" * 50)
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        sys.exit(1)
    
    logger.info(f"📁 Validando dataset: {dataset_path}")
    
    # Executar validação
    try:
        validator = DatasetValidator()
        result = validator.validate_yolo_dataset(dataset_path)
        
        # Imprimir resumo
        print_validation_summary(result, detailed=args.detailed)
        
        # Salvar relatório se solicitado
        if args.output_report:
            save_report(result, args.output_report)
        
        # Sugerir correções se há problemas
        if not result['valid']:
            suggest_fixes(result)
        
        # Status de saída
        if result['valid']:
            logger.success("\n🎉 Dataset está pronto para uso!")
            sys.exit(0)
        else:
            logger.error(f"\n❌ Dataset precisa de correções ({len(result['errors'])} erros)")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Erro durante validação: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Validação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro inesperado: {str(e)}")
        sys.exit(1)
