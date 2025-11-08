# 🔧 Instalação e Configuração

> Guia completo para configurar o ambiente Datalid 3.0

## 📋 Índice

- [Requisitos do Sistema](#requisitos-do-sistema)
- [Instalação Rápida](#instalação-rápida)
- [Instalação Detalhada](#instalação-detalhada)
- [Instalação com Docker](#instalação-com-docker)
- [Configuração de GPU](#configuração-de-gpu)
- [Verificação da Instalação](#verificação-da-instalação)
- [Troubleshooting](#troubleshooting)

## 🖥️ Requisitos do Sistema

### Requisitos Mínimos

- **Sistema Operacional:** Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python:** 3.8 ou superior
- **RAM:** 8 GB
- **Espaço em Disco:** 5 GB livres
- **Processador:** CPU quad-core

### Requisitos Recomendados

- **Python:** 3.10+
- **RAM:** 16 GB ou mais
- **GPU:** NVIDIA com CUDA 11.7+ (8GB+ VRAM)
- **Espaço em Disco:** 10 GB livres (para modelos e cache)

### Dependências Externas

**Windows:**
- Microsoft Visual C++ 14.0+ (para alguns pacotes Python)
- [Download aqui](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip build-essential
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install python@3.10
```

## 🚀 Instalação Rápida

### Método 1: Via Git Clone

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/datalid3.0.git
cd datalid3.0

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/macOS
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Verifique a instalação
make validate-env
```

### Método 2: Via Docker (Mais Fácil)

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/datalid3.0.git
cd datalid3.0

# 2. Build e inicie com Docker Compose
docker-compose up -d

# 3. Acesse a API
# http://localhost:8000/docs
```

## 📦 Instalação Detalhada

### Passo 1: Preparar o Ambiente Python

#### Windows

```powershell
# Instalar Python 3.10 (se não tiver)
# Baixe de: https://www.python.org/downloads/

# Verificar versão
python --version

# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate

# Atualizar pip
python -m pip install --upgrade pip
```

#### Linux/macOS

```bash
# Verificar versão do Python
python3 --version

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip
```

### Passo 2: Instalar Dependências Core

```bash
# Instalar dependências principais
pip install -r requirements.txt

# Para desenvolvimento (inclui ferramentas de teste e linting)
pip install -r requirements-dev.txt
```

### Passo 3: Baixar Modelos YOLO

Os modelos YOLO devem estar na raiz do projeto:

```bash
# Modelos já incluídos no repositório:
# - yolov8n.pt (nano, mais rápido)
# - yolov8n-seg.pt (nano segmentação)
# - yolov8s-seg.pt (small segmentação)
# - yolov8m-seg.pt (medium segmentação)

# Se precisar baixar manualmente:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
```

### Passo 4: Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# .env
PROJECT_ROOT=/caminho/para/datalid3.0
PYTHONPATH=${PROJECT_ROOT}
CUDA_VISIBLE_DEVICES=0  # Se tiver GPU

# Logs
LOG_LEVEL=INFO
LOG_FILE=logs/datalid.log

# API (se for usar)
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

## 🐳 Instalação com Docker

### Opção 1: Docker Compose (Recomendado)

```bash
# Arquivo docker-compose.yml já está configurado
# Basta executar:

docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviços
docker-compose down
```

### Opção 2: Docker Build Manual

```bash
# Build da imagem
docker build -t datalid:3.0 .

# Executar container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  --name datalid \
  datalid:3.0

# Ver logs
docker logs -f datalid
```

### Docker com GPU

```bash
# Instalar NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Executar com GPU
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  datalid:3.0
```

## 🎮 Configuração de GPU

### Verificar CUDA

```bash
# Testar se CUDA está disponível
make test-cuda

# Ou manualmente:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Instalar PyTorch com CUDA

Se você tem GPU NVIDIA, instale PyTorch com suporte CUDA:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verificar
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Configurar para CPU

Se não tiver GPU, o sistema funciona normalmente (mais lento):

```bash
# PyTorch CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## ✅ Verificação da Instalação

### Teste Completo do Ambiente

```bash
# Via Makefile (recomendado)
make validate-env

# Teste de GPU
make test-cuda

# Teste rápido do pipeline
make pipeline-test IMAGE=data/ocr_test/sample.jpg
```

### Verificação Manual

```python
# test_installation.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

import cv2
print(f"OpenCV: {cv2.__version__}")

from ultralytics import YOLO
print("✅ Ultralytics YOLO OK")

import yaml
print("✅ PyYAML OK")

print("\n✅ Instalação validada com sucesso!")
```

Execute:
```bash
python test_installation.py
```

### Saída Esperada

```
Python: 3.10.x
PyTorch: 2.x.x
CUDA disponível: True
OpenCV: 4.8.x
✅ Ultralytics YOLO OK
✅ PyYAML OK

✅ Instalação validada com sucesso!
```

## 🐛 Troubleshooting

### Erro: "No module named 'src'"

**Solução:** Adicione o diretório do projeto ao PYTHONPATH

```bash
# Linux/macOS
export PYTHONPATH="${PYTHONPATH}:/caminho/para/datalid3.0"

# Windows
set PYTHONPATH=%PYTHONPATH%;C:\caminho\para\datalid3.0

# Ou adicione ao .env
echo "PYTHONPATH=/caminho/para/datalid3.0" >> .env
```

### Erro: "Microsoft Visual C++ 14.0 is required" (Windows)

**Solução:** Instale o Build Tools for Visual Studio
- [Download aqui](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Selecione "Desktop development with C++"

### Erro: "CUDA out of memory"

**Solução:** Reduza o batch size ou use modelo menor

```yaml
# config/pipeline/full_pipeline.yaml
detection:
  batch_size: 1  # Reduzir de 8 para 1
  model_path: yolov8n-seg.pt  # Usar modelo nano
```

### Erro: "Could not find libGL.so" (Linux)

**Solução:** Instale as bibliotecas gráficas

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Problemas com OCR Engines

**PARSeq não carrega:**
```bash
# Baixar manualmente
python -c "import torch; torch.hub.load('baudm/parseq', 'parseq', pretrained=True)"
```

**Tesseract não encontrado:**
```bash
# Linux
sudo apt-get install tesseract-ocr tesseract-ocr-por

# macOS
brew install tesseract tesseract-lang

# Windows
# Baixe de: https://github.com/UB-Mannheim/tesseract/wiki
# Adicione ao PATH: C:\Program Files\Tesseract-OCR
```

### Performance Lenta

1. **Use GPU**: Instale CUDA e PyTorch com suporte GPU
2. **Modelo menor**: Use `yolov8n-seg.pt` ao invés de `yolov8m-seg.pt`
3. **Reduza resolução**: Configure `max_size: 640` no preprocessing
4. **Desabilite visualizações**: `save_visualizations: false`

## 🔄 Atualização

```bash
# Atualizar código
git pull origin main

# Atualizar dependências
pip install -r requirements.txt --upgrade

# Limpar cache
make clean
```

## 🗑️ Desinstalação

```bash
# Parar Docker (se usando)
docker-compose down -v

# Remover ambiente virtual
deactivate
rm -rf venv

# Remover arquivos do projeto
cd ..
rm -rf datalid3.0
```

## 📚 Próximos Passos

Agora que você tem o ambiente configurado:

1. **[Primeiros Passos](03-FIRST-STEPS.md)** - Faça seus primeiros testes
2. **[Guia de Início Rápido](01-QUICK-START.md)** - Uso básico do sistema
3. **[Arquitetura](04-ARCHITECTURE.md)** - Entenda como funciona

## 💡 Dicas

- ✅ **Use ambiente virtual** para isolar dependências
- ✅ **Configure GPU** para melhor performance
- ✅ **Docker** é a forma mais fácil de começar
- ✅ **Verifique a instalação** antes de usar
- ✅ **Consulte o FAQ** se tiver problemas

---

**Problemas?** Consulte [Troubleshooting](22-TROUBLESHOOTING.md) ou abra uma issue no GitHub.
