# 🎯 Detecção YOLO

> Sistema de detecção e segmentação de regiões de datas de validade usando YOLOv8

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura YOLOv8](#arquitetura-yolov8)
- [Modelos Disponíveis](#modelos-disponíveis)
- [Configuração](#configuração)
- [Uso](#uso)
- [Segmentação vs Detection](#segmentação-vs-detection)
- [Otimização](#otimização)

## 🎯 Visão Geral

O Datalid usa **YOLOv8-seg** (segmentação de instâncias) para:
- 🎯 **Localizar** regiões com datas de validade
- 🗺️ **Segmentar** contornos precisos (máscara poligonal)
- 📊 **Classificar** tipos de regiões (se necessário)
- ⚡ **Processar** em tempo real (GPU)

**Por que YOLO?**
- ✅ State-of-the-art em velocidade e precisão
- ✅ Single-stage detector (rápido)
- ✅ Suporta segmentação de instâncias
- ✅ Fácil de treinar e customizar
- ✅ Excelente documentação e comunidade

## 🏗️ Arquitetura YOLOv8

### Estrutura do Modelo

```
┌──────────────────────────────────────────┐
│           INPUT IMAGE                     │
│         (640x640x3 RGB)                   │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         BACKBONE (CSPDarknet)            │
│  - Extração hierárquica de features      │
│  - Multi-scale feature maps              │
│  - P3, P4, P5 (diferentes resoluções)    │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         NECK (PANet)                     │
│  - Feature pyramid network               │
│  - Path aggregation                      │
│  - Fusão multi-escala                    │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         DETECTION HEAD                    │
│  - Bounding boxes (x, y, w, h)           │
│  - Confidence scores                     │
│  - Class probabilities                   │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         SEGMENTATION HEAD                 │
│  - Mask coefficients (32 channels)       │
│  - Proto masks (160x160x32)              │
│  - Instance segmentation                 │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         POST-PROCESSING                   │
│  - NMS (Non-Maximum Suppression)         │
│  - Confidence filtering                  │
│  - Mask generation                       │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         OUTPUT                            │
│  - Boxes: [x1, y1, x2, y2]               │
│  - Masks: Polygon coordinates            │
│  - Scores: Confidence values             │
└──────────────────────────────────────────┘
```

### Componentes Principais

#### 1. Backbone (CSPDarknet)
```python
# Feature extraction em múltiplas escalas
P3: (80x80) - Objetos pequenos
P4: (40x40) - Objetos médios  
P5: (20x20) - Objetos grandes
```

#### 2. Neck (PANet)
```python
# Agregação de features
Bottom-up: P3 → P4 → P5 (captura contexto)
Top-down: P5 → P4 → P3 (refina detalhes)
```

#### 3. Detection Head
```python
# Para cada anchor:
- tx, ty, tw, th: Box coordinates
- objectness: Confidence score
- class_probs: [P(class1), P(class2), ...]
```

#### 4. Segmentation Head
```python
# Mask generation:
- Mask coefficients: 32 valores por instância
- Proto masks: Base masks (160x160x32)
- Final mask = Linear combination of proto masks
```

## 📦 Modelos Disponíveis

### YOLOv8 Variants

| Modelo | Parâmetros | FLOPs | mAP@50 | Velocidade (GPU) | Uso Recomendado |
|--------|-----------|-------|---------|------------------|-----------------|
| **YOLOv8n-seg** | 3.4M | 12.6G | 37.2 | **80 FPS** | 🚀 Produção (rápido) |
| **YOLOv8s-seg** | 11.8M | 42.6G | 44.6 | 60 FPS | ⚖️ Balanceado |
| **YOLOv8m-seg** | 27.3M | 110.2G | 49.9 | 35 FPS | 🎯 Alta precisão |
| **YOLOv8l-seg** | 46.0M | 220.5G | 52.3 | 20 FPS | 🔬 Pesquisa |
| **YOLOv8x-seg** | 71.8M | 344.1G | 53.4 | 12 FPS | 🏆 Máxima precisão |

### Qual Modelo Escolher?

**Para Produção (Tempo Real):**
```yaml
detection:
  model_path: yolov8n-seg.pt  # Nano
  imgsz: 640
  conf: 0.25
```

**Para Melhor Precisão:**
```yaml
detection:
  model_path: yolov8m-seg.pt  # Medium
  imgsz: 1280  # Maior resolução
  conf: 0.15   # Threshold menor
```

**Para CPU:**
```yaml
detection:
  model_path: yolov8n-seg.pt  # Nano (único viável)
  imgsz: 480   # Resolução menor
  half: false  # FP32 (sem FP16)
```

## ⚙️ Configuração

### Arquivo de Configuração

```yaml
# config/yolo/detection.yaml

detection:
  # Modelo
  model_path: yolov8n-seg.pt
  device: cuda  # cuda, cpu, mps (Apple Silicon)
  
  # Inference
  imgsz: 640            # Tamanho de entrada (múltiplo de 32)
  conf: 0.25            # Confidence threshold
  iou: 0.45             # IoU threshold para NMS
  max_det: 10           # Máximo de detecções por imagem
  
  # Segmentação
  retina_masks: true    # Máscaras em alta resolução
  
  # Performance
  half: true            # FP16 (requer GPU)
  batch_size: 1         # Batch processing
  
  # Augmentation (inference)
  augment: false        # Test-time augmentation (TTA)
  
  # Crop settings
  crop_padding: 10      # Pixels de margem no crop
  min_crop_size: 20     # Tamanho mínimo do crop
```

### Configuração Programática

```python
from ultralytics import YOLO

# Carregar modelo
model = YOLO('yolov8n-seg.pt')

# Configurar device
model.to('cuda')

# Inference com parâmetros customizados
results = model.predict(
    source='image.jpg',
    imgsz=640,
    conf=0.25,
    iou=0.45,
    max_det=10,
    retina_masks=True,
    half=True,
    verbose=False
)
```

## 🚀 Uso

### Uso Básico

```python
from src.pipeline.detection import DetectionPipeline
from src.ocr.config import load_pipeline_config

# Carregar configuração
config = load_pipeline_config('config/pipeline/full_pipeline.yaml')

# Criar pipeline de detecção
detector = DetectionPipeline(config)

# Processar imagem
result = detector.process('product_image.jpg')

# Acessar resultados
print(f"Detecções: {len(result['detections'])}")
for det in result['detections']:
    print(f"  BBox: {det['bbox']}")
    print(f"  Confiança: {det['confidence']:.2%}")
    print(f"  Área: {det['area']} pixels")
```

### Detecção em Batch

```python
# Múltiplas imagens
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.process_batch(images)

for img, result in zip(images, results):
    print(f"{img}: {len(result['detections'])} detecções")
```

### Extração de Crops

```python
import cv2
from src.yolo.utils import extract_crops

# Carregar imagem
image = cv2.imread('product.jpg')

# Detectar
result = detector.process(image)

# Extrair crops das regiões detectadas
crops = extract_crops(
    image,
    result['detections'],
    padding=10,
    use_mask=True  # Aplicar máscara de segmentação
)

# Salvar crops
for i, crop in enumerate(crops):
    cv2.imwrite(f'crop_{i}.jpg', crop['image'])
```

### Visualização

```python
from src.yolo.visualization import visualize_detections

# Criar visualização
vis_image = visualize_detections(
    image,
    result['detections'],
    show_boxes=True,
    show_masks=True,
    show_confidence=True,
    thickness=2,
    alpha=0.3  # Transparência da máscara
)

# Salvar ou exibir
cv2.imwrite('result.jpg', vis_image)
```

## 🎭 Segmentação vs Detection

### Detection (Bounding Box)

```python
# Detection apenas
model = YOLO('yolov8n.pt')  # Sem '-seg'

# Output
detection = {
    'bbox': [100, 200, 300, 400],  # x1, y1, x2, y2
    'confidence': 0.95,
    'class': 0
}
```

**Vantagens:**
- ✅ Mais rápido (~30% faster)
- ✅ Menor uso de memória
- ✅ Mais simples

**Desvantagens:**
- ❌ Bounding box retangular (pode incluir fundo)
- ❌ Menos preciso para OCR

### Segmentation (Instância)

```python
# Segmentation
model = YOLO('yolov8n-seg.pt')  # Com '-seg'

# Output
detection = {
    'bbox': [100, 200, 300, 400],
    'mask': np.array([[102, 203], [298, 205], ...]),  # Polígono
    'confidence': 0.95,
    'class': 0
}
```

**Vantagens:**
- ✅ Contorno preciso do objeto
- ✅ Remove fundo desnecessário
- ✅ Melhor para OCR (texto isolado)

**Desvantagens:**
- ❌ ~30% mais lento
- ❌ Maior uso de memória

### Quando Usar Cada Um?

**Use Detection se:**
- Velocidade é crítica
- Regiões são simples/retangulares
- Fundo não interfere no OCR

**Use Segmentation se:** ✅ (Recomendado para Datalid)
- Precisão é importante
- Regiões têm formatos irregulares
- Fundo pode atrapalhar OCR
- Quer melhor qualidade de crop

## ⚡ Otimização

### Otimização de Velocidade

#### 1. Half Precision (FP16)
```python
model = YOLO('yolov8n-seg.pt')
model.to('cuda')

# Habilitar FP16
results = model.predict(
    'image.jpg',
    half=True  # 2x mais rápido em GPUs modernas
)
```

#### 2. TensorRT (Máxima Performance)
```python
# Export para TensorRT
model.export(format='engine', half=True)

# Usar engine otimizado
model = YOLO('yolov8n-seg.engine')
results = model.predict('image.jpg')  # 3-5x mais rápido
```

#### 3. Batch Processing
```python
# Processar múltiplas imagens juntas
images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
results = model.predict(images, batch=4)  # Batch de 4
```

#### 4. Reduzir Resolução
```yaml
detection:
  imgsz: 480  # Ao invés de 640 (25% mais rápido)
```

### Otimização de Precisão

#### 1. Aumentar Resolução
```yaml
detection:
  imgsz: 1280  # Para objetos pequenos
```

#### 2. Test-Time Augmentation (TTA)
```python
results = model.predict(
    'image.jpg',
    augment=True  # Aplica múltiplas transformações e faz ensemble
)
```

#### 3. Ajustar Thresholds
```yaml
detection:
  conf: 0.15  # Reduzir para capturar mais detecções
  iou: 0.30   # Ajustar NMS para objetos próximos
```

#### 4. Multi-Scale Inference
```python
# Inferir em múltiplas escalas
results_640 = model.predict('image.jpg', imgsz=640)
results_1280 = model.predict('image.jpg', imgsz=1280)

# Combinar resultados (ensemble)
final_results = ensemble_results([results_640, results_1280])
```

### Otimização de Memória

```python
# Processar em chunks para datasets grandes
from pathlib import Path

image_dir = Path('data/images')
chunk_size = 100

for i, chunk in enumerate(image_dir.glob('*.jpg')):
    if i % chunk_size == 0:
        # Processar chunk
        results = model.predict(chunk, stream=True)
        
        # Limpar memória
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

## 📊 Métricas de Performance

### Benchmarks (GPU: RTX 3090)

| Modelo | Tamanho | FPS | mAP@50 | Memória |
|--------|---------|-----|--------|---------|
| YOLOv8n-seg | 640 | 78 | 37.2 | 2.1 GB |
| YOLOv8n-seg | 1280 | 22 | 41.5 | 3.8 GB |
| YOLOv8s-seg | 640 | 58 | 44.6 | 2.8 GB |
| YOLOv8m-seg | 640 | 34 | 49.9 | 4.2 GB |

### Benchmarks (CPU: Intel i7-10700K)

| Modelo | Tamanho | FPS | Uso CPU |
|--------|---------|-----|---------|
| YOLOv8n-seg | 640 | 2.1 | 100% |
| YOLOv8n-seg | 480 | 3.5 | 100% |

## 🐛 Troubleshooting

### Problema: CUDA Out of Memory

**Solução:**
```yaml
detection:
  batch_size: 1  # Reduzir batch
  imgsz: 480     # Reduzir resolução
  half: false    # Desabilitar FP16 se instável
```

### Problema: Detecções Faltando

**Soluções:**
1. Reduzir threshold de confiança
2. Aumentar resolução de entrada
3. Verificar se a data está bem visível na imagem
4. Considerar retreinar o modelo

### Problema: Muitas False Positives

**Soluções:**
1. Aumentar threshold de confiança
2. Ajustar NMS IoU threshold
3. Aplicar filtros de pós-processamento (tamanho, posição)

## 📚 Referências

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
- [Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)

## 💡 Próximos Passos

- **[Treinamento YOLO](13-YOLO-TRAINING.md)** - Treinar modelo customizado
- **[Pré-processamento](09-PREPROCESSING.md)** - Otimizar crops
- **[Sistema OCR](08-OCR-SYSTEM.md)** - Extrair texto das regiões

---

**Dúvidas sobre detecção YOLO?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
