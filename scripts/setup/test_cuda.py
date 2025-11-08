"""
🧪 Teste de Disponibilidade CUDA e GPU
Verifica se PyTorch consegue acessar a GPU.
"""

import platform
import sys
from pathlib import Path

import torch

# Adicionar root ao PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def print_header():
    """Imprime cabeçalho."""
    print("\n" + "="*60)
    print("🧪 TESTE DE CUDA E GPU")
    print("="*60)


def test_cuda():
    """Testa disponibilidade do CUDA."""
    print("\n📋 INFORMAÇÕES DO SISTEMA:")
    print(f"  • Sistema Operacional: {platform.system()} {platform.release()}")
    print(f"  • Python: {sys.version.split()[0]}")
    print(f"  • PyTorch: {torch.__version__}")

    print("\n🔍 CUDA:")
    cuda_available = torch.cuda.is_available()
    print(f"  • CUDA Disponível: {'✅ SIM' if cuda_available else '❌ NÃO'}")

    if cuda_available:
        print(f"  • Versão CUDA: {torch.version.cuda}")
        print(f"  • cuDNN: {torch.backends.cudnn.version()}")

        print("\n🎮 GPUs DETECTADAS:")
        num_gpus = torch.cuda.device_count()
        print(f"  • Quantidade: {num_gpus}")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  📱 GPU {i}: {props.name}")
            print(
                f"     • Memória Total: {props.total_memory / 1024**3:.2f} GB")
            print(f"     • Compute Capability: {props.major}.{props.minor}")

            # Teste de alocação
            try:
                print(f"\n  ⚡ Testando alocação na GPU {i}...")
                device = torch.device(f'cuda:{i}')
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = x @ y
                print(f"     ✅ Teste de multiplicação de matrizes OK!")

                # Memória usada
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"     • Memória Alocada: {allocated:.2f} GB")
                print(f"     • Memória Reservada: {cached:.2f} GB")

                # Limpar
                del x, y, z
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"     ❌ Erro no teste: {str(e)}")

        return True
    else:
        print("\n❌ CUDA NÃO DISPONÍVEL")
        print("\n💡 POSSÍVEIS CAUSAS:")
        print("  1. Driver NVIDIA não instalado")
        print("  2. PyTorch instalado sem suporte CUDA")
        print("  3. GPU não compatível")
        print("\n🔧 SOLUÇÃO:")
        print("  • Instale o driver NVIDIA: https://www.nvidia.com/drivers")
        print("  • Reinstale PyTorch com CUDA:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

        return False


def test_yolo_gpu():
    """Testa se YOLO consegue usar GPU."""
    print("\n🎯 TESTE YOLO + GPU:")
    try:
        from ultralytics import YOLO

        print("  • Ultralytics instalado: ✅")

        if torch.cuda.is_available():
            print("  • Tentando carregar modelo...")
            model = YOLO("yolov8n.pt")
            print("  • Modelo carregado: ✅")

            # Verificar device
            device = next(model.model.parameters()).device
            print(f"  • Device padrão: {device}")

            print("\n✅ YOLO pronto para usar GPU!")
        else:
            print("  ⚠️  YOLO funcionará apenas com CPU (lento)")

    except ImportError:
        print("  ❌ Ultralytics não instalado")
        print("  💡 Instale: pip install ultralytics")
    except Exception as e:
        print(f"  ❌ Erro: {str(e)}")


def print_recommendations():
    """Imprime recomendações."""
    print("\n" + "="*60)
    print("💡 RECOMENDAÇÕES PARA TREINAMENTO:")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n✅ GPU Detectada: {gpu_name}")
        print(f"✅ Memória: {total_mem:.1f}GB")

        # Recomendações baseadas na GPU
        if "1660" in gpu_name or total_mem < 8:
            print("\n📊 Configurações Recomendadas:")
            print("  • Modelo: YOLOv8n ou YOLOv8s")
            print("  • Batch Size: 16")
            print("  • Image Size: 640")
            print("  • Workers: 4")
        elif total_mem >= 8:
            print("\n📊 Configurações Recomendadas:")
            print("  • Modelo: YOLOv8s ou YOLOv8m")
            print("  • Batch Size: 32")
            print("  • Image Size: 640")
            print("  • Workers: 8")

        print("\n🚀 Pronto para treinar!")
        print("   Execute: make train-test")
    else:
        print("\n❌ Sem GPU - Treinamento será MUITO lento")
        print("💡 Considere usar Google Colab ou Kaggle")


def main():
    """Executa teste completo."""
    print_header()

    cuda_ok = test_cuda()
    test_yolo_gpu()
    print_recommendations()

    print("\n" + "="*60)

    if cuda_ok:
        print("✅ Sistema pronto para treinamento!")
        sys.exit(0)
    else:
        print("⚠️  Configure CUDA antes de treinar")
        sys.exit(1)


if __name__ == "__main__":
    main()
