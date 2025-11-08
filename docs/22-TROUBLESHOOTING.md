# 🔧 Troubleshooting

> Soluções para problemas comuns no Datalid 3.0

## 📋 Índice

- [Problemas de Instalação](#problemas-de-instalação)
- [Problemas de Detecção](#problemas-de-detecção)
- [Problemas de OCR](#problemas-de-ocr)
- [Problemas de Performance](#problemas-de-performance)
- [Problemas de API](#problemas-de-api)
- [Problemas de Treinamento](#problemas-de-treinamento)

## 🔧 Problemas de Instalação

### Erro: "No module named 'src'"

**Sintoma:**
```
ModuleNotFoundError: No module named 'src'
```

**Causa:** PYTHONPATH não configurado

**Soluções:**

1. **Adicionar ao PYTHONPATH**
```bash
# Linux/macOS
export PYTHONPATH="${PYTHONPATH}:/caminho/para/datalid3.0"

# Windows PowerShell
$env:PYTHONPATH+=";C:\caminho\para\datalid3.0"

# Ou adicionar ao .env
echo "PYTHONPATH=/caminho/para/datalid3.0" >> .env
```

2. **Executar do diretório raiz**
```bash
cd /caminho/para/datalid3.0
python -m src.pipeline.full_pipeline
```

3. **Instalar em modo editável**
```bash
pip install -e .
```

### Erro: "Microsoft Visual C++ 14.0 is required" (Windows)

**Sintoma:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solução:**
1. Baixe [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Instale com opção "Desktop development with C++"
3. Reinicie terminal
4. Execute `pip install -r requirements.txt` novamente

### Erro: "Could not find libGL.so" (Linux)

**Sintoma:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solução:**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Erro: CUDA Out of Memory (Durante Instalação)

**Sintoma:**
```
RuntimeError: CUDA out of memory
```

**Soluções:**

1. **Instalar versão CPU do PyTorch**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **Usar versão menor do modelo**
```yaml
detection:
  model_path: yolov8n-seg.pt  # nano ao invés de medium
```

### Erro: "Tesseract not found"

**Sintoma:**
```
TesseractNotFoundError: tesseract is not installed
```

**Soluções:**

```bash
# Linux
sudo apt-get install tesseract-ocr tesseract-ocr-por

# macOS
brew install tesseract tesseract-lang

# Windows
# 1. Baixe: https://github.com/UB-Mannheim/tesseract/wiki
# 2. Instale
# 3. Adicione ao PATH: C:\Program Files\Tesseract-OCR
```

## 🎯 Problemas de Detecção

### Problema: Nenhuma Região Detectada

**Sintoma:**
```
❌ Nenhuma região detectada pela YOLO
```

**Causas e Soluções:**

#### 1. Threshold de Confiança Muito Alto

```yaml
# config/pipeline/full_pipeline.yaml
detection:
  conf: 0.15  # Reduzir de 0.25 para 0.15
```

#### 2. Imagem com Resolução Muito Baixa

```yaml
detection:
  imgsz: 1280  # Aumentar de 640 para 1280
```

#### 3. Data Muito Pequena na Imagem

**Solução:** Fazer crop manual da região antes de processar

```python
import cv2

# Carregar imagem
img = cv2.imread('product.jpg')

# Crop manual da região aproximada
h, w = img.shape[:2]
crop = img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

# Processar crop
result = pipeline.process(crop)
```

#### 4. Modelo Não Treinado para Esse Tipo de Imagem

**Solução:** Treinar modelo customizado

Ver [Treinamento YOLO](13-YOLO-TRAINING.md)

### Problema: Muitas False Positives

**Sintoma:**
Detecta regiões que não são datas

**Soluções:**

#### 1. Aumentar Threshold

```yaml
detection:
  conf: 0.50  # Aumentar de 0.25 para 0.50
```

#### 2. Filtrar por Tamanho

```python
# No código
detections = [d for d in detections if 100 < d['area'] < 50000]
```

#### 3. Filtrar por Posição

```python
# Assumir que data está na metade inferior da imagem
h, w = image.shape[:2]
detections = [d for d in detections if d['bbox'][1] > h * 0.4]
```

### Problema: Segmentação Imprecisa

**Sintoma:**
Máscara de segmentação inclui muito fundo

**Soluções:**

#### 1. Usar Modelo Melhor

```yaml
detection:
  model_path: yolov8m-seg.pt  # Medium tem melhor segmentação
```

#### 2. Aumentar Resolução

```yaml
detection:
  imgsz: 1280  # Maior resolução = melhor segmentação
```

#### 3. Usar Retina Masks

```yaml
detection:
  retina_masks: true  # Máscaras em alta resolução
```

## 📝 Problemas de OCR

### Problema: Texto Extraído Incorretamente

**Sintoma:**
```
OCR: "15i03i2025" (esperado: "15/03/2025")
```

**Soluções:**

#### 1. Experimentar Outro OCR Engine

```bash
# Testar diferentes engines
python scripts/ocr/benchmark_ocr.py --image crop.jpg
```

```yaml
# Usar melhor engine
ocr:
  engine: openocr  # ou parseq, trocr
```

#### 2. Melhorar Pré-processamento

```yaml
preprocessing:
  enable: true
  clahe:
    enable: true
    clip_limit: 3.0  # Aumentar contraste
  denoise:
    enable: true
    strength: 7      # Remover mais ruído
  sharpen:
    enable: true
    amount: 2.0      # Aumentar nitidez
```

#### 3. Aumentar Resolução do Crop

```yaml
detection:
  crop_padding: 20  # Mais margem (padrão: 10)

preprocessing:
  resize:
    max_height: 96  # Maior altura (padrão: 64)
```

### Problema: OCR Muito Lento

**Sintoma:**
OCR leva > 5 segundos por imagem

**Soluções:**

#### 1. Usar Engine Mais Rápido

```yaml
ocr:
  engine: tesseract  # Mais rápido (0.3s vs 0.8s)
```

#### 2. Reduzir Resolução

```yaml
preprocessing:
  resize:
    max_height: 48  # Menor = mais rápido
```

#### 3. Desabilitar Pré-processamento Custoso

```yaml
preprocessing:
  denoise:
    enable: false  # Denoise é lento
  deskew:
    enable: false  # Deskew é lento
```

#### 4. Usar GPU

```yaml
ocr:
  openocr:
    device: cuda  # Ao invés de cpu
```

### Problema: OCR Retorna String Vazia

**Sintoma:**
```
OCR text: ""
```

**Causas e Soluções:**

#### 1. Imagem Muito Pequena

```python
# Verificar tamanho do crop
print(f"Crop size: {crop.shape}")

# Se altura < 20px, é muito pequeno
```

**Solução:** Aumentar `crop_padding` ou desabilitar resize agressivo

#### 2. Imagem Muito Escura/Clara

**Solução:** Aplicar normalização e CLAHE

```yaml
preprocessing:
  normalize: {enable: true}
  clahe: {enable: true, clip_limit: 3.0}
```

#### 3. Texto Ilegível

**Solução:** Melhorar qualidade da imagem original

## ⚡ Problemas de Performance

### Problema: Pipeline Muito Lento

**Sintoma:**
> 10 segundos por imagem

**Diagnóstico:**

```python
result = pipeline.process('image.jpg')
print(result['processing_time'])
# {
#   'detection': 5.2,  ← Problema aqui
#   'preprocessing': 0.1,
#   'ocr': 0.8,
#   'parsing': 0.02,
#   'total': 6.12
# }
```

**Soluções por Componente:**

#### Se Detecção é Lenta (> 2s):

```yaml
detection:
  model_path: yolov8n-seg.pt  # Nano (3x mais rápido)
  imgsz: 480                   # Menor resolução
  half: true                   # FP16 (GPU)
```

#### Se OCR é Lento (> 3s):

```yaml
ocr:
  engine: tesseract  # Engine mais rápido

preprocessing:
  denoise: {enable: false}  # Desabilitar
  resize: {max_height: 48}  # Menor resolução
```

#### Se Pré-processamento é Lento (> 0.5s):

```yaml
preprocessing:
  enable: false  # Desabilitar completamente
  # ou desabilitar técnicas custosas
  denoise: {enable: false}
  deskew: {enable: false}
```

### Problema: Alto Uso de Memória

**Sintoma:**
```
MemoryError: Unable to allocate array
```

**Soluções:**

#### 1. Processar em Chunks

```python
import gc

for i, image in enumerate(images):
    result = pipeline.process(image)
    
    # Limpar memória a cada 100 imagens
    if i % 100 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

#### 2. Usar Modo Streaming

```python
# Ao invés de carregar todas
images = list(Path('data').glob('*.jpg'))

# Processar sob demanda
for img_path in Path('data').glob('*.jpg'):
    result = pipeline.process(str(img_path))
    # Processar e descartar
```

#### 3. Reduzir Batch Size

```yaml
detection:
  batch_size: 1  # Processar 1 por vez
```

#### 4. Não Cachear Imagens

```yaml
# Em training
cache: false  # Não cachear (usa menos RAM)
```

### Problema: GPU Não Está Sendo Usada

**Sintoma:**
```
GPU utilization: 0%
```

**Verificar:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**Soluções:**

#### 1. Especificar Device

```yaml
detection:
  device: cuda  # ou 0, 1, etc.

ocr:
  openocr:
    device: cuda
```

#### 2. Reinstalar PyTorch com CUDA

```bash
# Para CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🌐 Problemas de API

### Problema: API Não Inicia

**Sintoma:**
```
Error: Address already in use
```

**Solução:**

```bash
# Verificar porta em uso
# Linux/macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Matar processo
kill -9 <PID>  # Linux/macOS
taskkill /F /PID <PID>  # Windows

# Ou usar porta diferente
uvicorn src.api.main:app --port 8001
```

### Problema: Upload de Arquivo Falha

**Sintoma:**
```
413 Request Entity Too Large
```

**Solução:**

Aumentar limite de upload:

```python
# src/api/main.py
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_upload_size=50 * 1024 * 1024  # 50MB
)
```

### Problema: Timeout em Requisições

**Sintoma:**
```
504 Gateway Timeout
```

**Solução:**

Aumentar timeout:

```python
# src/api/main.py
import uvicorn

uvicorn.run(
    app,
    timeout_keep_alive=300  # 5 minutos
)
```

## 🎓 Problemas de Treinamento

### Problema: Loss Não Diminui

**Sintoma:**
```
Epoch 1: loss=1.234
Epoch 2: loss=1.230
Epoch 3: loss=1.235
...
Epoch 20: loss=1.220  ← Não melhora
```

**Causas e Soluções:**

#### 1. Learning Rate Muito Alto

```yaml
lr0: 0.001  # Reduzir de 0.01 para 0.001
```

#### 2. Dataset Muito Pequeno

**Solução:** Coletar mais dados (mínimo 500 imagens)

#### 3. Anotações Incorretas

```bash
# Validar anotações
python scripts/data/validate_annotations.py
python scripts/data/visualize_annotations.py
```

#### 4. Batch Size Muito Pequeno

```yaml
batch: 16  # Aumentar de 4 para 16
```

### Problema: Overfitting

**Sintoma:**
```
Train loss: 0.2  ← Baixo
Val loss: 0.8    ← Alto (pior que train)
```

**Soluções:**

#### 1. Mais Regularização

```yaml
weight_decay: 0.001  # Aumentar
dropout: 0.1         # Adicionar
```

#### 2. Mais Augmentation

```yaml
mosaic: 1.0
mixup: 0.15
hsv_h: 0.02
hsv_s: 0.8
degrees: 20
```

#### 3. Early Stopping

```yaml
patience: 30  # Parar mais cedo
```

#### 4. Mais Dados

- Coletar mais imagens de treino
- Data augmentation offline

### Problema: CUDA Out of Memory (Treinamento)

**Sintoma:**
```
RuntimeError: CUDA out of memory
```

**Soluções:**

```yaml
# 1. Reduzir batch size
batch: 4  # Ao invés de 16

# 2. Reduzir resolução
imgsz: 480  # Ao invés de 640

# 3. Modelo menor
model: yolov8n-seg.pt  # Ao invés de yolov8m-seg.pt

# 4. Desabilitar mixed precision
amp: false

# 5. Gradient accumulation
accumulate: 4  # Acumular gradientes de 4 batches
```

## 🔍 Diagnóstico Geral

### Script de Diagnóstico

```python
# scripts/utils/diagnose.py
import sys
import torch
import cv2
from pathlib import Path

def diagnose_system():
    """Diagnóstico completo do sistema."""
    
    print("🔍 Datalid Diagnostic Tool\n")
    
    # Python
    print(f"Python: {sys.version}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # OpenCV
    print(f"OpenCV: {cv2.__version__}")
    
    # YOLO
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO: OK")
    except ImportError as e:
        print(f"❌ Ultralytics YOLO: {e}")
    
    # OCR Engines
    engines = ['openocr', 'parseq', 'tesseract', 'easyocr']
    for engine in engines:
        try:
            exec(f"from src.ocr.engines.{engine} import *")
            print(f"✅ {engine}: OK")
        except Exception as e:
            print(f"❌ {engine}: {e}")
    
    # Verificar paths
    print(f"\nPYTHONPATH: {sys.path}")
    print(f"Current dir: {Path.cwd()}")

if __name__ == '__main__':
    diagnose_system()
```

Execute:
```bash
python scripts/utils/diagnose.py
```

## 📚 Recursos Adicionais

- [FAQ](25-FAQ.md) - Perguntas frequentes
- [Instalação](02-INSTALLATION.md) - Guia de instalação
- [Documentação](README.md) - Índice completo

## 💬 Ainda com Problemas?

1. **Procure no FAQ:** [25-FAQ.md](25-FAQ.md)
2. **Revise a instalação:** [02-INSTALLATION.md](02-INSTALLATION.md)
3. **Verifique logs:** `logs/datalid.log`
4. **Execute diagnóstico:** `python scripts/utils/diagnose.py`
5. **Abra uma issue no GitHub** com:
   - Descrição do problema
   - Mensagem de erro completa
   - Output do diagnóstico
   - Versões (Python, PyTorch, CUDA)
   - Sistema operacional

---

**Problemas resolvidos?** Continue em [Pipeline Completo](11-FULL-PIPELINE.md)
