# 📊 Preparação de Dados

> Guia completo para preparar e anotar dados para treinamento YOLO

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Coleta de Dados](#coleta-de-dados)
- [Anotação](#anotação)
- [Formato de Dados](#formato-de-dados)
- [Split de Dados](#split-de-dados)
- [Data Augmentation](#data-augmentation)
- [Validação](#validação)

## 🎯 Visão Geral

Para treinar um modelo YOLO customizado, você precisa de:
- **Imagens** de produtos com datas de validade
- **Anotações** (bounding boxes ou máscaras de segmentação)
- **Organização** adequada dos dados (train/val/test)

**Quantidade recomendada:**
- Mínimo: 300-500 imagens anotadas
- Ideal: 1000-2000 imagens
- Produção: 5000+ imagens

## 📸 Coleta de Dados

### Fontes de Imagens

#### 1. Captura Manual (Recomendado)
```bash
# Tirar fotos de produtos reais
- Supermercados
- Farmácias
- Distribuidoras
- Estoque próprio
```

**Dicas:**
- ✅ Variedade de produtos
- ✅ Diferentes ângulos
- ✅ Várias condições de iluminação
- ✅ Diferentes resoluções
- ✅ Incluir casos difíceis (borradas, inclinadas, reflexos)

#### 2. Dataset Público
```bash
# Procurar em:
- Kaggle
- Roboflow
- Google Dataset Search
- Papers científicos
```

#### 3. Web Scraping
```python
# Coletar de e-commerces (com permissão)
# Exemplo: produtos com datas visíveis
```

### Organização Inicial

```bash
data/raw/
├── batch1/
│   ├── produto_001.jpg
│   ├── produto_002.jpg
│   └── ...
├── batch2/
│   └── ...
└── batch3/
    └── ...
```

### Pré-seleção

Remova imagens sem datas visíveis:

```python
# scripts/data/filter_images.py
import cv2
from pathlib import Path

def has_visible_date(image_path):
    """Verifica se imagem tem data visível (manual ou automático)."""
    img = cv2.imread(str(image_path))
    # Critérios:
    # - Resolução mínima (640x480)
    # - Não muito borrada
    # - Iluminação adequada
    return True  # Implementar critérios

input_dir = Path('data/raw')
output_dir = Path('data/filtered')

for img_path in input_dir.rglob('*.jpg'):
    if has_visible_date(img_path):
        # Copiar para output
        ...
```

## 🏷️ Anotação

### Ferramentas Recomendadas

#### 1. CVAT (Computer Vision Annotation Tool)
```bash
# Instalar CVAT
docker-compose -f docker-compose.yml up -d

# Acessar: http://localhost:8080
```

**Vantagens:**
- ✅ Interface web intuitiva
- ✅ Suporta segmentação poligonal
- ✅ Anotação colaborativa
- ✅ Export para YOLO format
- ✅ Gratuito e open-source

**Tutorial:**
1. Criar projeto
2. Upload imagens
3. Criar label "expiry_date"
4. Anotar com polígonos (segmentação)
5. Export em YOLO v8 format

#### 2. LabelImg/LabelMe
```bash
pip install labelImg

# Executar
labelImg data/images data/classes.txt
```

**Vantagens:**
- ✅ Simples e rápido
- ✅ Desktop app
- ✅ Bom para bounding boxes

**Desvantagens:**
- ❌ Não suporta segmentação nativa

#### 3. Roboflow
```bash
# Web-based: https://roboflow.com
```

**Vantagens:**
- ✅ Interface excelente
- ✅ Augmentation automático
- ✅ Export direto para YOLO
- ✅ Versionamento de datasets

**Desvantagens:**
- ❌ Limite no plano gratuito

### Guia de Anotação

#### Bounding Box
```
Para detection apenas:
- Desenhar retângulo ao redor da data
- Incluir todo o texto da data
- Margem pequena (2-5 pixels)
```

#### Segmentação Poligonal (Recomendado)
```
Para segmentação:
- Desenhar polígono ao redor da data
- Seguir contorno preciso do texto
- ~8-12 pontos por polígono
- Incluir sombras/contorno se relevante
```

**Exemplo Visual:**
```
╔══════════════════════════════╗
║  PRODUTO XYZ                 ║
║                              ║
║  ┌─────────────────┐        ║
║  │ VAL 15/03/2025  │ ← Anotar essa região
║  └─────────────────┘        ║
║                              ║
║  PESO: 500g                  ║
╚══════════════════════════════╝
```

### Padrões de Qualidade

✅ **Boas anotações:**
- Cobrem todo o texto da data
- Precisas (sem muito fundo)
- Consistentes entre imagens

❌ **Anotações ruins:**
- Muito grandes (incluem muito fundo)
- Muito pequenas (cortam texto)
- Inconsistentes

### Anotação Assistida

Use modelo pré-treinado para pré-anotar:

```python
# scripts/data/pre_annotate.py
from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO('yolov8n-seg.pt')

# Prever em imagens não anotadas
results = model.predict('data/unannotated/', save_txt=True)

# Revisar e corrigir manualmente no CVAT
```

## 📦 Formato de Dados

### Estrutura YOLO

```bash
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt  # Anotação para img_001.jpg
│   │   ├── img_002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml  # Configuração do dataset
```

### Arquivo de Anotação (.txt)

#### Para Detection (Bounding Box)
```
# Format: class_id x_center y_center width height
# Valores normalizados (0-1)

0 0.5123 0.6234 0.2156 0.0823
```

#### Para Segmentation (Polygon)
```
# Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
# Valores normalizados (0-1)

0 0.4156 0.5823 0.6234 0.5845 0.6245 0.6734 0.4167 0.6712
```

**Exemplo real:**
```txt
# img_001.txt
0 0.512 0.623 0.534 0.628 0.531 0.645 0.509 0.641
```

### data.yaml

```yaml
# data.yaml

# Paths (relative to data.yaml)
path: .  # Dataset root
train: images/train  # Train images
val: images/val      # Validation images
test: images/test    # Test images (opcional)

# Classes
names:
  0: expiry_date

# Dataset info (opcional)
nc: 1  # Number of classes
```

## 📊 Split de Dados

### Proporções Recomendadas

```
Train: 70-80%  (treino)
Val:   15-20%  (validação durante treino)
Test:  10-15%  (teste final)
```

### Script de Split

```python
# scripts/data/split_dataset.py
import random
import shutil
from pathlib import Path

def split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
):
    """
    Split dataset into train/val/test.
    """
    # Validar proporções
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Criar estrutura de diretórios
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Listar todos os pares imagem-label
    images = list((source_dir / 'images').glob('*.jpg'))
    
    # Shuffle
    random.shuffle(images)
    
    # Calcular índices de split
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Fazer split
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Copiar arquivos
    for split_name, split_images in [
        ('train', train_images),
        ('val', val_images),
        ('test', test_images)
    ]:
        for img_path in split_images:
            # Copiar imagem
            shutil.copy(
                img_path,
                output_dir / 'images' / split_name / img_path.name
            )
            
            # Copiar label
            label_path = (source_dir / 'labels' / img_path.stem).with_suffix('.txt')
            if label_path.exists():
                shutil.copy(
                    label_path,
                    output_dir / 'labels' / split_name / label_path.name
                )
    
    print(f"Split completo:")
    print(f"  Train: {len(train_images)} imagens")
    print(f"  Val:   {len(val_images)} imagens")
    print(f"  Test:  {len(test_images)} imagens")

# Usar
split_dataset(
    source_dir=Path('data/annotated'),
    output_dir=Path('data/dataset_v1'),
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)
```

### Via Makefile

```bash
make data-split INPUT=data/annotated OUTPUT=data/dataset_v1
```

## 🔄 Data Augmentation

### Por que Augmentar?

- ✅ Aumentar tamanho do dataset artificialmente
- ✅ Melhorar generalização
- ✅ Reduzir overfitting
- ✅ Simular diferentes condições

### Augmentations Recomendados

```python
# Durante treinamento YOLO (automático)
augmentations = {
    # Geométricas
    'flipud': 0.5,       # Flip vertical (50%)
    'fliplr': 0.5,       # Flip horizontal
    'mosaic': 1.0,       # Mosaic (4 imgs juntas)
    'mixup': 0.1,        # Mix duas imagens
    'rotate': 15,        # Rotação (-15° a +15°)
    'translate': 0.1,    # Translação
    'scale': 0.5,        # Zoom
    'shear': 0.0,        # Shear (perspectiva)
    
    # Cor/Luz
    'hsv_h': 0.015,      # Hue
    'hsv_s': 0.7,        # Saturation
    'hsv_v': 0.4,        # Value (brightness)
    'brightness': 0.0,
    'contrast': 0.0,
    
    # Ruído
    'noise': 0.0,        # Gaussian noise
    'blur': 0.0,         # Motion blur
}
```

### Augmentation Offline

Para criar dataset aumentado permanentemente:

```python
# scripts/data/augment_dataset.py
import albumentations as A
import cv2
from pathlib import Path

# Definir pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Aplicar
for img_path in Path('data/train/images').glob('*.jpg'):
    image = cv2.imread(str(img_path))
    
    # Carregar bboxes
    label_path = Path('data/train/labels') / img_path.with_suffix('.txt').name
    bboxes, class_labels = load_yolo_labels(label_path)
    
    # Augmentar
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    # Salvar versões aumentadas
    for i in range(3):  # 3 variações
        aug_image = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        save_augmented(aug_image, img_path.stem + f'_aug{i}.jpg')
```

## ✅ Validação

### Verificar Anotações

```python
# scripts/data/validate_annotations.py
from pathlib import Path

def validate_dataset(dataset_dir):
    """Valida integridade do dataset."""
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split
        
        for img_path in img_dir.glob('*.jpg'):
            # 1. Check se label existe
            label_path = label_dir / img_path.with_suffix('.txt').name
            if not label_path.exists():
                issues.append(f"Missing label: {img_path.name}")
                continue
            
            # 2. Check formato do label
            with open(label_path) as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    
                    # Mínimo: class_id + 4 coords (bbox) ou 6+ (polygon)
                    if len(parts) < 5:
                        issues.append(
                            f"Invalid format in {label_path.name}:{line_num}"
                        )
                    
                    # Verificar se valores estão normalizados (0-1)
                    try:
                        coords = [float(x) for x in parts[1:]]
                        if not all(0 <= c <= 1 for c in coords):
                            issues.append(
                                f"Out of range coords in {label_path.name}:{line_num}"
                            )
                    except ValueError:
                        issues.append(
                            f"Non-numeric value in {label_path.name}:{line_num}"
                        )
    
    if issues:
        print(f"❌ {len(issues)} issues found:")
        for issue in issues[:10]:  # Mostrar primeiros 10
            print(f"  - {issue}")
    else:
        print("✅ Dataset validated successfully!")
    
    return len(issues) == 0

# Usar
validate_dataset(Path('data/dataset_v1'))
```

### Visualizar Anotações

```python
# scripts/data/visualize_annotations.py
import cv2
import numpy as np
from pathlib import Path

def visualize_annotation(img_path, label_path):
    """Visualiza imagem com anotações."""
    # Carregar imagem
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Carregar anotações
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            if len(coords) == 4:  # Bounding box
                x_center, y_center, width, height = coords
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            else:  # Polygon
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.polylines(img, [points], True, (0, 255, 0), 2)
    
    # Exibir
    cv2.imshow('Annotation', img)
    cv2.waitKey(0)

# Visualizar todas
dataset_dir = Path('data/dataset_v1')
for img_path in (dataset_dir / 'images/train').glob('*.jpg'):
    label_path = (dataset_dir / 'labels/train') / img_path.with_suffix('.txt').name
    if label_path.exists():
        visualize_annotation(img_path, label_path)
```

## 📊 Estatísticas do Dataset

```python
# scripts/data/dataset_stats.py
def dataset_statistics(dataset_dir):
    """Calcula estatísticas do dataset."""
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'images_per_split': {},
        'avg_annotations_per_image': 0,
        'image_sizes': []
    }
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / 'images' / split
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob('*.jpg'))
        stats['images_per_split'][split] = len(images)
        stats['total_images'] += len(images)
        
        for img_path in images:
            # Tamanho da imagem
            img = cv2.imread(str(img_path))
            stats['image_sizes'].append(img.shape[:2])
            
            # Contar anotações
            label_path = (dataset_dir / 'labels' / split / 
                         img_path.with_suffix('.txt').name)
            if label_path.exists():
                with open(label_path) as f:
                    stats['total_annotations'] += len(f.readlines())
    
    stats['avg_annotations_per_image'] = (
        stats['total_annotations'] / stats['total_images']
    )
    
    return stats
```

## 📚 Próximos Passos

- **[Treinamento YOLO](13-YOLO-TRAINING.md)** - Treinar modelo com seus dados
- **[Avaliação](14-EVALUATION.md)** - Medir qualidade do modelo
- **[Data Augmentation FAQ](docs/ai_gen/FAQ_DATA_AUGMENTATION.md)** - Perguntas comuns

---

**Dúvidas sobre preparação de dados?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
