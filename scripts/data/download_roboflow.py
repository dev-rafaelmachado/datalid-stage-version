"""
📥 Script de Download Automático do Roboflow
Baixa dataset do Roboflow usando a API.
"""

import argparse
import os
# Adicionar src ao path
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # pasta base do projeto
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# Garante que o diretório existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configurações do projeto (baseadas no código fornecido)
DEFAULT_API_KEY = "crS7dKMHZj3VlfWw40mS"
DEFAULT_WORKSPACE = "projetotransformadorii"
DEFAULT_PROJECT = "tcc_dateset_v2-zkcsu"
DEFAULT_VERSION = 6
DEFAULT_FORMAT = "yolov8"


def check_roboflow_installation():
    """Verifica se roboflow está instalado."""
    try:
        import roboflow
        logger.info("✅ Roboflow já instalado")
        return True
    except ImportError:
        logger.warning("⚠️ Roboflow não encontrado, instalando...")
        return False


def install_roboflow():
    """Instala biblioteca roboflow."""
    import subprocess
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "roboflow"])
        logger.success("✅ Roboflow instalado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro instalando roboflow: {e}")
        return False


def ensure_absolute_path(path_str: str) -> Path:
    """Garante que o caminho seja absoluto e resolva a partir do diretório do projeto."""
    path = Path(path_str)
    if not path.is_absolute():
        # Se não for absoluto, resolve a partir do diretório raiz do projeto
        path = ROOT / path
    return path.resolve()


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    format: str = "yolov8",
    output_dir: str = "../../data/raw"
) -> bool:
    """
    Baixa dataset do Roboflow.

    Args:
        api_key: Chave da API do Roboflow
        workspace: Nome do workspace
        project: Nome do projeto
        version: Versão do dataset
        format: Formato de saída (yolov8, coco, etc.)
        output_dir: Diretório de saída

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        # Importar roboflow
        from roboflow import Roboflow

        logger.info("🔗 Conectando ao Roboflow...")
        rf = Roboflow(api_key=api_key)

        logger.info(f"📂 Acessando projeto: {workspace}/{project}")
        project_obj = rf.workspace(workspace).project(project)

        logger.info(f"🔢 Selecionando versão: {version}")
        version_obj = project_obj.version(version)

        # Criar diretório de saída com caminho absoluto
        output_path = ensure_absolute_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Mudar para diretório de saída temporariamente
        original_dir = os.getcwd()
        logger.info(f"📁 Mudando para diretório: {output_path}")
        os.chdir(output_path)

        try:
            logger.info(f"📥 Baixando dataset no formato {format}...")
            dataset = version_obj.download(format)

            logger.success("✅ Dataset baixado com sucesso!")

            # Mostrar informações do dataset
            dataset_info = {
                'workspace': workspace,
                'project': project,
                'version': version,
                'format': format,
                'location': str(output_path.absolute())
            }

            logger.info("📊 INFORMAÇÕES DO DATASET:")
            for key, value in dataset_info.items():
                logger.info(f"  • {key}: {value}")

            # Verificar estrutura baixada
            downloaded_items = list(output_path.iterdir())
            logger.info(f"📁 Arquivos/pastas baixados: {len(downloaded_items)}")
            # Mostrar apenas os primeiros 10
            for item in downloaded_items[:10]:
                item_type = "📁" if item.is_dir() else "📄"
                logger.info(f"  {item_type} {item.name}")

            if len(downloaded_items) > 10:
                logger.info(f"  ... e mais {len(downloaded_items) - 10} itens")

            return True

        finally:
            # Voltar ao diretório original
            os.chdir(original_dir)

    except Exception as e:
        logger.error(f"❌ Erro baixando dataset: {str(e)}")
        return False


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Download automático de dataset do Roboflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=DEFAULT_API_KEY,
        help='Chave da API do Roboflow'
    )

    parser.add_argument(
        '--workspace',
        type=str,
        default=DEFAULT_WORKSPACE,
        help='Nome do workspace no Roboflow'
    )

    parser.add_argument(
        '--project',
        type=str,
        default=DEFAULT_PROJECT,
        help='Nome do projeto no Roboflow'
    )

    parser.add_argument(
        '--version',
        type=int,
        default=DEFAULT_VERSION,
        help='Versão do dataset'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['yolov8', 'yolov5', 'coco', 'pascal-voc', 'tensorflow'],
        default=DEFAULT_FORMAT,
        help='Formato de download'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_DIR,
        help='Diretório de saída'
    )

    parser.add_argument(
        '--force-install',
        action='store_true',
        help='Forçar instalação do roboflow'
    )

    parser.add_argument(
        '--process-after',
        action='store_true',
        help='Processar dados automaticamente após download'
    )

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_arguments()

    logger.info("📥 DOWNLOAD AUTOMÁTICO DO ROBOFLOW")
    logger.info("=" * 50)

    try:
        # 1. Verificar/instalar roboflow
        if args.force_install or not check_roboflow_installation():
            if not install_roboflow():
                logger.error("❌ Falha na instalação do roboflow")
                return 1

        # 2. Download do dataset
        success = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            format=args.format,
            output_dir=args.output
        )

        if not success:
            logger.error("❌ Falha no download do dataset")
            return 1

        # 3. Processar dados se solicitado
        if args.process_after:
            logger.info("🔄 Processando dados baixados...")

            # Importar script de processamento
            try:
                import subprocess

                # Encontrar pasta do dataset baixado
                output_path = ensure_absolute_path(args.output)
                dataset_folders = [
                    d for d in output_path.iterdir() if d.is_dir()]

                if dataset_folders:
                    # Usar primeira pasta encontrada
                    dataset_path = dataset_folders[0]

                    # Usar caminhos absolutos
                    script_path = ROOT / "scripts" / "process_raw_data.py"
                    output_processed = ROOT / "data" / "processed"

                    process_cmd = [
                        sys.executable,
                        str(script_path),
                        "--input", str(dataset_path),
                        "--output", str(output_processed),
                        "--task", "both",
                        "--validate"
                    ]

                    result = subprocess.run(
                        process_cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        logger.success("✅ Dados processados automaticamente")
                    else:
                        logger.warning("⚠️ Erro no processamento automático")
                        logger.info(
                            "💡 Execute manualmente: make process INPUT=" + str(dataset_path))
                else:
                    logger.warning(
                        "⚠️ Pasta do dataset não encontrada para processamento")

            except Exception as e:
                logger.warning(f"⚠️ Erro no processamento automático: {e}")
                logger.info(
                    "💡 Execute manualmente: make process INPUT=" + args.output)

        # Resumo final
        logger.success("🎉 DOWNLOAD CONCLUÍDO!")
        logger.info(f"\n📁 Dataset baixado em: {args.output}")
        logger.info(f"🔢 Versão: {args.version}")
        logger.info(f"📋 Formato: {args.format}")

        if not args.process_after:
            logger.info(f"\n🔄 Para processar os dados, execute:")
            logger.info(f"   make process-data INPUT={args.output}")

        return 0

    except KeyboardInterrupt:
        logger.warning("⚠️ Download interrompido pelo usuário")
        return 1
    except Exception as e:
        logger.error(f"❌ Erro no download: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
