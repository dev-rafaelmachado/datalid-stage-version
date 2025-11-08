# 🎓 Treinamento YOLO

> Guia completo para treinar modelos YOLO customizados

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Configuração de Treinamento](#configuração-de-treinamento)
- [Executar Treinamento](#executar-treinamento)
- [Monitoramento](#monitoramento)
- [Validação e Testes](#validação-e-testes)
- [Fine-tuning](#fine-tuning)
- [Troubleshooting](#troubleshooting)

## 🎯 Visão Geral

Treine modelos YOLO customizados para detectar datas de validade em seus próprios dados.

**Por que treinar um modelo customizado?**
- ✅ Melhor performance em seus dados específicos
- ✅ Adaptar a contextos únicos (embalagens, produtos)
- ✅ Suportar novas classes além de datas
- ✅ Otimizar para seu caso de uso

**Tempo estimado:**
- Preparação: 2-4 horas
- Treinamento: 2-12 horas (depende do dataset e GPU)
- Validação: 1-2 horas

## 📋 Pré-requisitos

### 1. Dataset Preparado

Você precisa ter um dataset anotado no formato YOLO:

```bash
dataset/
├── images/
│   ├── train/  (700+ imagens)
│   ├── val/    (200+ imagens)
│   └── test/   (100+ imagens)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**Se ainda não tem:** Consulte [Preparação de Dados](12-DATA-PREPARATION.md)

### 2. Hardware

**Mínimo:**
- GPU NVIDIA com 6GB+ VRAM
- 16GB RAM
- 20GB espaço em disco

**Recomendado:**
- GPU NVIDIA RTX 3060+ (12GB+ VRAM)
- 32GB RAM
- 50GB espaço em disco (para checkpoints)

**CPU Only:**
- Possível, mas **muito lento** (50-100x mais lento)
- Não recomendado para treino, apenas inferência

### 3. Software

```bash
# Verificar instalação
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Instalar se necessário
pip install ultralytics torch torchvision
```

## ⚙️ Configuração de Treinamento

### 1. Criar data.yaml

```yaml
# data/dataset_v1/data.yaml

# Paths (absolutos ou relativos a este arquivo)
path: .  # Dataset root
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: expiry_date

# Metadata (opcional)
nc: 1  # Número de classes
```

### 2. Criar Training Config

```yaml
# config/yolo/training/train_config.yaml

# Modelo base
model: yolov8n-seg.pt  # nano, s, m, l, x

# Dataset
data: data/dataset_v1/data.yaml

# Hyperparameters
epochs: 100            # Número de épocas
batch: 16             # Batch size (ajuste conforme GPU)
imgsz: 640            # Tamanho da imagem
device: 0             # GPU ID (ou 'cpu')

# Otimizador
optimizer: AdamW      # SGD, Adam, AdamW
lr0: 0.01            # Learning rate inicial
lrf: 0.01            # Learning rate final (lr0 * lrf)
momentum: 0.937      # Momentum (para SGD)
weight_decay: 0.0005 # Weight decay

# Augmentation
hsv_h: 0.015         # Hue augmentation
hsv_s: 0.7           # Saturation
hsv_v: 0.4           # Value (brightness)
degrees: 15.0        # Rotação (-15° a +15°)
translate: 0.1       # Translação
scale: 0.5           # Zoom
shear: 0.0          # Shear
perspective: 0.0     # Perspectiva
flipud: 0.0         # Flip vertical
fliplr: 0.5         # Flip horizontal (50%)
mosaic: 1.0         # Mosaic augmentation
mixup: 0.1          # Mixup augmentation
copy_paste: 0.0     # Copy-paste augmentation

# Training settings
patience: 50         # Early stopping (épocas sem melhora)
save: true          # Salvar checkpoints
save_period: 10     # Salvar a cada N épocas
cache: false        # Cache imagens (mais rápido, usa mais RAM)
workers: 8          # DataLoader workers
project: experiments # Diretório de saída
name: yolov8n_seg_v1 # Nome do experimento
exist_ok: false     # Sobrescrever experimento existente

# Validation
val: true           # Validar durante treino
plots: true         # Gerar plots
verbose: true       # Logs detalhados
```

### 3. Ajustar Hiperparâmetros por Modelo

#### YOLOv8n (Nano) - Rápido
```yaml
model: yolov8n-seg.pt
epochs: 150
batch: 32
lr0: 0.01
```

#### YOLOv8s (Small) - Balanceado
```yaml
model: yolov8s-seg.pt
epochs: 120
batch: 24
lr0: 0.01
```

#### YOLOv8m (Medium) - Melhor Precisão
```yaml
model: yolov8m-seg.pt
epochs: 100
batch: 16
lr0: 0.008
```

#### YOLOv8l/x (Large/XLarge) - Máxima Precisão
```yaml
model: yolov8l-seg.pt
epochs: 80
batch: 8
lr0: 0.005
weight_decay: 0.001
```

## 🚀 Executar Treinamento

### Método 1: Python Script

```python
# scripts/training/train_yolo.py
from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO('yolov8n-seg.pt')

# Treinar
results = model.train(
    data='data/dataset_v1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='experiments',
    name='yolov8n_seg_v1',
    patience=50,
    save=True,
    plots=True,
    verbose=True
)

# Resultados
print(f"Melhor mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"Melhor epoch: {results.best_epoch}")
```

Execute:
```bash
python scripts/training/train_yolo.py
```

### Método 2: CLI Ultralytics

```bash
# Treinamento básico
yolo segment train \
  data=data/dataset_v1/data.yaml \
  model=yolov8n-seg.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=experiments \
  name=yolov8n_seg_v1

# Com configuração YAML
yolo segment train \
  config=config/yolo/training/train_config.yaml
```

### Método 3: Via Makefile

```bash
# Treino rápido (nano, 100 épocas)
make yolo-train-nano

# Treino de produção (medium, 100 épocas)
make yolo-train-medium

# Treino customizado
make yolo-train MODEL=yolov8s-seg.pt EPOCHS=150 BATCH=24
```

### Método 4: Script Avançado com Logging

```python
# scripts/training/train_advanced.py
import yaml
from pathlib import Path
from ultralytics import YOLO
from loguru import logger

def train_yolo(config_path: str):
    """Treina YOLO com configuração YAML."""
    
    # Carregar config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"🚀 Iniciando treinamento")
    logger.info(f"  Modelo: {config['model']}")
    logger.info(f"  Dataset: {config['data']}")
    logger.info(f"  Épocas: {config['epochs']}")
    logger.info(f"  Batch: {config['batch']}")
    
    # Criar modelo
    model = YOLO(config['model'])
    
    # Treinar
    try:
        results = model.train(**config)
        
        # Log resultados
        logger.success("✅ Treinamento concluído!")
        logger.info(f"  Melhor mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  Melhor epoch: {results.best_epoch}")
        logger.info(f"  Modelo salvo em: {results.save_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        raise

# Usar
train_yolo('config/yolo/training/train_config.yaml')
```

## 📊 Monitoramento

### Durante o Treinamento

O YOLO mostra progresso em tempo real:

```
Epoch    GPU_mem   box_loss   seg_loss   cls_loss  Instances       Size
  1/100      4.2G      1.234      0.876      0.543        128        640: 100%|████| 50/50
            Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
              all        200        250      0.876      0.823      0.891      0.654

Epoch    GPU_mem   box_loss   seg_loss   cls_loss  Instances       Size
  2/100      4.3G      0.987      0.723      0.432        128        640: 100%|████| 50/50
            Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
              all        200        250      0.892      0.845      0.908      0.682
...
```

**Métricas principais:**
- `box_loss` - Perda de localização (bounding box)
- `seg_loss` - Perda de segmentação (máscara)
- `cls_loss` - Perda de classificação
- `mAP50` - Mean Average Precision @ IoU 0.5 (principal)
- `mAP50-95` - mAP médio de IoU 0.5 a 0.95

### TensorBoard

```bash
# Iniciar TensorBoard
tensorboard --logdir experiments/yolov8n_seg_v1

# Acessar: http://localhost:6006
```

**Gráficos disponíveis:**
- Loss curves (train/val)
- mAP curves
- Learning rate schedule
- Precision/Recall curves
- Confusion matrix
- Exemplos de predições

### Arquivos Gerados

```
experiments/yolov8n_seg_v1/
├── weights/
│   ├── best.pt           # Melhor modelo (maior mAP)
│   ├── last.pt           # Último checkpoint
│   └── epoch_*.pt        # Checkpoints periódicos
├── results.png           # Gráficos de métricas
├── confusion_matrix.png  # Matriz de confusão
├── PR_curve.png          # Precision-Recall curve
├── F1_curve.png          # F1 score curve
├── results.csv           # Métricas por época
├── args.yaml             # Argumentos usados
└── events.out.tfevents  # TensorBoard logs
```

## ✅ Validação e Testes

### Validar Modelo

```python
from ultralytics import YOLO

# Carregar melhor modelo
model = YOLO('experiments/yolov8n_seg_v1/weights/best.pt')

# Validar no conjunto de validação
metrics = model.val(
    data='data/dataset_v1/data.yaml',
    split='val',
    imgsz=640,
    batch=16,
    plots=True
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
```

### Testar em Imagens

```python
# Predizer em imagens de teste
results = model.predict(
    source='data/dataset_v1/images/test',
    save=True,
    save_txt=True,
    save_conf=True,
    project='outputs/validation',
    name='test_predictions'
)

# Analisar resultados
for r in results:
    print(f"Imagem: {r.path}")
    print(f"Detecções: {len(r.boxes)}")
    for box in r.boxes:
        conf = box.conf.item()
        cls = int(box.cls.item())
        print(f"  Classe: {cls}, Confiança: {conf:.2%}")
```

### Via Makefile

```bash
# Validar modelo
make yolo-val MODEL=experiments/yolov8n_seg_v1/weights/best.pt

# Testar em diretório
make yolo-test MODEL=experiments/yolov8n_seg_v1/weights/best.pt \
  INPUT=data/dataset_v1/images/test
```

## 🎯 Fine-tuning

### Transfer Learning Eficiente

```python
# Começar de modelo pré-treinado do Datalid
model = YOLO('experiments/yolov8m_seg_best/weights/best.pt')

# Fine-tune com seus dados
results = model.train(
    data='data/my_custom_dataset/data.yaml',
    epochs=50,              # Menos épocas
    lr0=0.001,             # Learning rate menor
    freeze=10,             # Congelar primeiras 10 camadas
    patience=20
)
```

### Ajuste de Hiperparâmetros

```bash
# Hyperparameter tuning automático
yolo segment tune \
  data=data/dataset_v1/data.yaml \
  model=yolov8n-seg.pt \
  epochs=50 \
  iterations=100  # Número de combinações a testar
```

### Continuar Treinamento

```python
# Retomar de checkpoint
model = YOLO('experiments/yolov8n_seg_v1/weights/last.pt')

# Continuar treinando
results = model.train(
    resume=True,  # Retomar treinamento
    epochs=150    # Total de épocas (não adicional)
)
```

## 🐛 Troubleshooting

### Problema: CUDA Out of Memory

**Soluções:**
```yaml
# Reduzir batch size
batch: 8  # ou 4, 2

# Reduzir resolução
imgsz: 480  # ao invés de 640

# Desabilitar workers
workers: 0

# Não cachear imagens
cache: false
```

### Problema: Loss não Diminui

**Causas e soluções:**

1. **Learning rate muito alto**
```yaml
lr0: 0.001  # Reduzir de 0.01
```

2. **Dataset muito pequeno**
- Coletar mais dados (mínimo 500 imagens)
- Usar data augmentation mais agressivo

3. **Anotações incorretas**
- Revisar anotações manualmente
- Usar `validate_annotations.py`

### Problema: Overfitting (Val loss > Train loss)

**Soluções:**

1. **Aumentar regularização**
```yaml
weight_decay: 0.001  # Aumentar
dropout: 0.1        # Adicionar
```

2. **Mais augmentation**
```yaml
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 20
mosaic: 1.0
mixup: 0.15
```

3. **Early stopping**
```yaml
patience: 30  # Parar mais cedo
```

4. **Mais dados de treino**
- Coletar mais imagens
- Data augmentation offline

### Problema: Treinamento Muito Lento

**Soluções:**

1. **Cache de imagens**
```yaml
cache: true  # Usa mais RAM, mas acelera
# ou
cache: disk  # Usa disco (SSD recomendado)
```

2. **Mais workers**
```yaml
workers: 16  # Ajustar conforme CPU
```

3. **Mixed precision (FP16)**
```yaml
amp: true  # Automatic Mixed Precision
```

4. **Modelo menor**
```yaml
model: yolov8n-seg.pt  # Nano ao invés de medium
```

## 📈 Boas Práticas

### 1. Começar Simples

```python
# Primeiro: treino curto para validar setup
results = model.train(
    data='data/dataset_v1/data.yaml',
    epochs=5,  # Apenas 5 épocas
    batch=8,
    imgsz=640,
    cache=False
)
# Se funcionar, aumentar epochs para 100+
```

### 2. Monitorar Métricas

- **mAP50** é a métrica principal
- Validar a cada época (`val=True`)
- Usar TensorBoard para visualizar

### 3. Salvar Checkpoints

```yaml
save: true
save_period: 10  # Salvar a cada 10 épocas
```

### 4. Usar Early Stopping

```yaml
patience: 50  # Parar se 50 épocas sem melhora
```

### 5. Experimentos Organizados

```python
# Nomear experimentos claramente
name: f"yolov8n_seg_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"
```

## 📚 Referências

- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [Transfer Learning](https://docs.ultralytics.com/guides/transfer-learning/)

## 💡 Próximos Passos

- **[Avaliação](14-EVALUATION.md)** - Avaliar performance do modelo
- **[Comparação de Modelos](15-MODEL-COMPARISON.md)** - Comparar diferentes modelos
- **[Pipeline Completo](11-FULL-PIPELINE.md)** - Usar modelo treinado no pipeline

---

**Dúvidas sobre treinamento?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
