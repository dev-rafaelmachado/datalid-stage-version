# 🚀 Pipeline Completo

> Integração end-to-end: Detecção → Pré-processamento → OCR → Parse

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Uso](#uso)
- [Configuração](#configuração)
- [Modos de Operação](#modos-de-operação)
- [Performance](#performance)
- [Casos de Uso](#casos-de-uso)

## 🎯 Visão Geral

O **Full Pipeline** é o módulo principal do Datalid que integra todos os componentes para extrair datas de validade de forma completa e automática.

**Fluxo:**
```
Imagem → YOLO → Crops → Preprocess → OCR → Parse → Data Validada
```

**Features:**
- ✅ **Detecção automática** de regiões com YOLO
- ✅ **Pré-processamento inteligente** adaptativo
- ✅ **7 engines OCR** intercambiáveis
- ✅ **Parse multi-formato** com fuzzy matching
- ✅ **Validação robusta** de datas
- ✅ **Visualizações** automáticas
- ✅ **Batch processing** para múltiplas imagens
- ✅ **Modo streaming** para processamento contínuo

## 🏗️ Arquitetura

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────┐
│                    FULL PIPELINE                         │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│  YOLO Detector  │ │ Preprocessor│ │   OCR Engine    │
│  - yolov8n-seg  │ │ - CLAHE     │ │ - OpenOCR       │
│  - Segmentation │ │ - Denoise   │ │ - PARSeq        │
│  - BBox + Mask  │ │ - Sharpen   │ │ - TrOCR         │
└─────────────────┘ └─────────────┘ └─────────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Date Parser   │
                  │ - Regex extract │
                  │ - Validation    │
                  │ - Fuzzy match   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Resultado Final │
                  │ - Melhor data   │
                  │ - Confiança     │
                  │ - Visualização  │
                  └─────────────────┘
```

### Classe Principal

```python
# src/pipeline/full_pipeline.py

class FullPipeline(PipelineBase):
    """
    Pipeline completo: YOLO → OCR → Parse.
    
    Detecta, extrai e valida datas de validade em imagens.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa pipeline com configuração."""
        self.yolo_model = YOLO(config['detection']['model_path'])
        self.preprocessor = ImagePreprocessor(config['preprocessing'])
        self.ocr_engine = self._load_ocr_engine(config['ocr'])
        self.date_parser = DateParser(config['parsing'])
    
    def process(self, image_path: str) -> Dict:
        """Processa uma imagem completa."""
        # 1. Detectar regiões
        detections = self._detect(image_path)
        
        # 2. Para cada região
        results = []
        for det in detections:
            # 2a. Extrair crop
            crop = self._extract_crop(image, det)
            
            # 2b. Pré-processar
            preprocessed = self.preprocessor.process(crop)
            
            # 2c. OCR
            text = self.ocr_engine.recognize(preprocessed)
            
            # 2d. Parse data
            parsed = self.date_parser.parse(text)
            
            results.append(parsed)
        
        # 3. Selecionar melhor data
        best_date = self._select_best_date(results)
        
        return {
            'best_date': best_date,
            'all_results': results,
            'processing_time': {...}
        }
```

## 🚀 Uso

### Uso Básico

```python
from src.pipeline.full_pipeline import FullPipeline
from src.ocr.config import load_pipeline_config

# 1. Carregar configuração
config = load_pipeline_config('config/pipeline/full_pipeline.yaml')

# 2. Inicializar pipeline
pipeline = FullPipeline(config)

# 3. Processar imagem
result = pipeline.process('product_image.jpg')

# 4. Acessar resultado
if result['best_date']:
    print(f"Data: {result['best_date']['date']}")
    print(f"Confiança: {result['best_date']['confidence']:.2%}")
    print(f"Dias até expirar: {result['best_date']['days_until_expiry']}")
else:
    print("Nenhuma data encontrada")
```

### Via Linha de Comando

```bash
# Processar uma imagem
python scripts/pipeline/run_full_pipeline.py \
  --image product.jpg \
  --config config/pipeline/full_pipeline.yaml \
  --output outputs/pipeline/

# Com visualização
python scripts/pipeline/run_full_pipeline.py \
  --image product.jpg \
  --save-visualization \
  --save-crops

# Processar diretório
python scripts/pipeline/run_full_pipeline.py \
  --input-dir data/images/ \
  --output-dir outputs/results/ \
  --batch-size 8
```

### Via Makefile

```bash
# Teste rápido
make pipeline-test IMAGE=product.jpg

# Processar diretório
make pipeline-batch INPUT_DIR=data/images/

# Com OCR específico
make pipeline-test IMAGE=product.jpg OCR_ENGINE=parseq
```

### Batch Processing

```python
from pathlib import Path

# Processar múltiplas imagens
image_dir = Path('data/images')
images = list(image_dir.glob('*.jpg'))

# Modo 1: Lista de imagens
results = pipeline.process_batch(images)

# Modo 2: Stream (para muitas imagens)
for image_path in images:
    result = pipeline.process(str(image_path))
    # Processar resultado...
    
# Modo 3: Paralelo (experimental)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(pipeline.process, img) for img in images]
    results = [f.result() for f in futures]
```

## ⚙️ Configuração

### Arquivo YAML Completo

```yaml
# config/pipeline/full_pipeline.yaml

# Nome do pipeline
name: "Datalid Full Pipeline"
version: "3.0"

# ============================================
# DETECÇÃO (YOLO)
# ============================================
detection:
  model_path: "yolov8n-seg.pt"  # Modelo YOLO
  device: "cuda"                 # cuda, cpu, mps
  imgsz: 640                     # Resolução de entrada
  conf: 0.25                     # Confidence threshold
  iou: 0.45                      # IoU para NMS
  max_det: 10                    # Máx detecções/imagem
  half: true                     # FP16 (GPU only)
  
  # Crop settings
  crop_padding: 10               # Pixels de margem
  min_crop_size: 20              # Tamanho mínimo
  use_mask: true                 # Usar máscara de segmentação

# ============================================
# PRÉ-PROCESSAMENTO
# ============================================
preprocessing:
  enable: true
  
  resize:
    enable: true
    max_height: 64
    maintain_aspect: true
  
  deskew:
    enable: false  # Apenas se necessário
  
  normalize:
    enable: true
    method: "minmax"
  
  clahe:
    enable: true
    clip_limit: 2.0
    tile_size: 8
  
  denoise:
    enable: true
    method: "bilateral"
    strength: 5
  
  sharpen:
    enable: true
    amount: 1.5
    radius: 1.0
  
  binarize:
    enable: false  # Não para DL models

# ============================================
# OCR
# ============================================
ocr:
  engine: "openocr"  # openocr, parseq, trocr, easyocr, paddleocr, tesseract
  
  # Configurações específicas do engine
  openocr:
    model: "GOT"
    device: "cuda"
    use_format: "plain"
  
  parseq:
    model: "parseq"
    device: "cuda"
    charset: "94"
  
  # Fallback (se OCR primário falhar)
  fallback:
    enable: true
    engines: ["parseq", "tesseract"]

# ============================================
# PARSING E VALIDAÇÃO
# ============================================
parsing:
  # Formatos de data aceitos
  date_formats:
    - "%d/%m/%Y"  # DD/MM/YYYY (BR)
    - "%d-%m-%Y"
    - "%d.%m.%Y"
    - "%d %m %Y"
    - "%m/%d/%Y"  # MM/DD/YYYY (US)
    - "%Y-%m-%d"  # YYYY-MM-DD (ISO)
  
  # Prefixos/sufixos conhecidos
  known_prefixes:
    - "VAL"
    - "VALIDADE"
    - "VENC"
    - "EXP"
    - "EXPIRA"
  
  # Fuzzy matching
  fuzzy_matching: true
  fuzzy_threshold: 0.85
  
  # Validação
  validation:
    min_year: 2020
    max_year: 2050
    allow_past_dates: true  # Permitir datas expiradas
    max_days_past: 365      # Máx 1 ano no passado
  
  # Scoring
  scoring_weights:
    ocr_confidence: 0.4
    format_common: 0.2
    date_reasonable: 0.2
    consistency: 0.2

# ============================================
# OUTPUT
# ============================================
output:
  # Visualizações
  save_visualizations: true
  save_crops: false
  save_preprocessed: false
  
  # Formato de saída
  save_json: true
  save_csv: false
  
  # Logs
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/pipeline.log"
```

### Configurações Predefinidas

#### Produção (Velocidade + Qualidade)

```yaml
# config/pipeline/production.yaml
detection:
  model_path: "yolov8n-seg.pt"  # Nano (rápido)
  imgsz: 640
  half: true

preprocessing:
  enable: true
  # Apenas essencial
  clahe: {enable: true}
  normalize: {enable: true}
  denoise: {enable: false}

ocr:
  engine: "openocr"  # Melhor balanceamento

output:
  save_visualizations: false  # Economizar I/O
  save_crops: false
```

#### Desenvolvimento (Máxima Informação)

```yaml
# config/pipeline/development.yaml
detection:
  model_path: "yolov8m-seg.pt"  # Medium (melhor)
  imgsz: 1280

preprocessing:
  enable: true
  # Tudo habilitado para debug
  denoise: {enable: true}
  sharpen: {enable: true}
  deskew: {enable: true}

output:
  save_visualizations: true
  save_crops: true
  save_preprocessed: true  # Debug
  log_level: "DEBUG"
```

#### Pesquisa (Máxima Precisão)

```yaml
# config/pipeline/research.yaml
detection:
  model_path: "yolov8x-seg.pt"  # Extra large
  imgsz: 1920
  conf: 0.10  # Threshold baixo

preprocessing:
  enable: true
  # Agressivo
  clahe: {clip_limit: 3.5}
  denoise: {strength: 8}
  sharpen: {amount: 2.0}

ocr:
  engine: "openocr"
  fallback:
    enable: true
    engines: ["parseq", "trocr", "tesseract"]  # Tentar todos

parsing:
  fuzzy_matching: true
  fuzzy_threshold: 0.70  # Mais permissivo
```

## 🎮 Modos de Operação

### Modo 1: Single Image

Processa uma imagem por vez.

```python
result = pipeline.process('image.jpg')
```

**Quando usar:**
- API endpoints
- Processamento interativo
- Debugging

### Modo 2: Batch

Processa múltiplas imagens eficientemente.

```python
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.process_batch(images, batch_size=8)
```

**Quando usar:**
- Processamento de dataset
- Avaliação de performance
- Treinamento de modelos downstream

**Vantagens:**
- ✅ Compartilha overhead de inicialização
- ✅ Otimizações de GPU (batch inference)
- ✅ Melhor throughput

### Modo 3: Streaming

Processa imagens sob demanda (generator).

```python
image_paths = Path('data').glob('*.jpg')

for image_path in image_paths:
    result = pipeline.process_stream(image_path)
    # Processar resultado imediatamente
    save_to_database(result)
    # Libera memória antes da próxima
```

**Quando usar:**
- Datasets muito grandes
- Memória limitada
- Processamento em tempo real

**Vantagens:**
- ✅ Baixo uso de memória
- ✅ Processamento contínuo
- ✅ Early stopping possível

### Modo 4: Pipeline Parcial

Executa apenas parte do pipeline.

```python
# Apenas detecção
detections = pipeline.detect_only('image.jpg')

# Apenas OCR (já tem crop)
text = pipeline.ocr_only(crop_image)

# Apenas parse (já tem texto)
date = pipeline.parse_only("15/03/2025")
```

**Quando usar:**
- Testes isolados
- Debugging específico
- Benchmarking de componentes

## 📊 Performance

### Benchmarks (GPU: RTX 3090)

| Configuração | Modelo YOLO | OCR Engine | Tempo/img | Throughput |
|-------------|-------------|------------|-----------|------------|
| **Rápido** | yolov8n-seg | Tesseract | 0.5s | ~2 FPS |
| **Balanceado** | yolov8n-seg | OpenOCR | 1.2s | ~0.8 FPS |
| **Qualidade** | yolov8m-seg | OpenOCR | 2.5s | ~0.4 FPS |
| **Máximo** | yolov8x-seg | Ensemble | 8.0s | ~0.12 FPS |

### Benchmarks (CPU: Intel i7-10700K)

| Configuração | Tempo/img | Throughput |
|-------------|-----------|------------|
| **Mínimo** | 5.0s | ~0.2 FPS |
| **Padrão** | 12.0s | ~0.08 FPS |

### Otimizações

```python
# 1. Usar FP16 (GPU)
config['detection']['half'] = True

# 2. Reduzir resolução
config['detection']['imgsz'] = 480

# 3. Batch processing
results = pipeline.process_batch(images, batch_size=16)

# 4. Desabilitar visualizações
config['output']['save_visualizations'] = False

# 5. Engine OCR mais rápido
config['ocr']['engine'] = 'tesseract'
```

## 💼 Casos de Uso

### Caso 1: API REST

```python
from fastapi import FastAPI, UploadFile
from src.pipeline.full_pipeline import FullPipeline

app = FastAPI()
pipeline = FullPipeline(config)

@app.post("/process")
async def process_image(file: UploadFile):
    # Salvar temporariamente
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Processar
    result = pipeline.process(image_path)
    
    # Retornar JSON
    return {
        'date': result['best_date']['date'] if result['best_date'] else None,
        'confidence': result['best_date']['confidence'] if result['best_date'] else 0,
        'days_until_expiry': result['best_date']['days_until_expiry'] if result['best_date'] else None
    }
```

### Caso 2: Processamento de Dataset

```python
import pandas as pd
from pathlib import Path

# Carregar lista de imagens
image_dir = Path('data/products')
images = list(image_dir.glob('*.jpg'))

# Processar em batch
results = []
batch_size = 16

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    batch_results = pipeline.process_batch(batch)
    results.extend(batch_results)

# Salvar em CSV
df = pd.DataFrame([
    {
        'image': r['image_path'],
        'date': r['best_date']['date'] if r['best_date'] else None,
        'confidence': r['best_date']['confidence'] if r['best_date'] else 0
    }
    for r in results
])
df.to_csv('results.csv', index=False)
```

### Caso 3: Monitoramento em Tempo Real

```python
import cv2
from src.pipeline.full_pipeline import FullPipeline

pipeline = FullPipeline(config)

# Captura de câmera/webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Processar frame
    result = pipeline.process(frame)
    
    # Exibir resultado
    if result['best_date']:
        text = f"Data: {result['best_date']['date']}"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Datalid', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🐛 Troubleshooting

### Problema: Pipeline Muito Lento

**Soluções:**
1. Usar GPU se disponível
2. Reduzir resolução (`imgsz: 480`)
3. Modelo menor (`yolov8n-seg.pt`)
4. Desabilitar visualizações
5. OCR mais rápido (Tesseract)

### Problema: Memória Insuficiente

**Soluções:**
1. Batch size menor
2. Modo streaming
3. Desabilitar cache
4. Limpar memória entre batches

```python
import gc
import torch

# A cada N imagens
if i % 100 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

### Problema: Baixa Precisão

**Soluções:**
1. Modelo maior (yolov8m-seg)
2. Resolução maior (imgsz: 1280)
3. Pré-processamento mais agressivo
4. OCR engine diferente (OpenOCR, PARSeq)
5. Ajustar thresholds

## 📚 Próximos Passos

- **[Avaliação](14-EVALUATION.md)** - Medir performance
- **[API REST](16-API-REST.md)** - Deploy em produção
- **[Otimização](21-OPTIMIZATION.md)** - Melhorar velocidade/precisão

---

**Dúvidas?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
