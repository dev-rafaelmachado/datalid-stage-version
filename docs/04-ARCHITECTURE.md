# 🏗️ Arquitetura do Sistema

> Entenda como o Datalid 3.0 é estruturado internamente

## 📐 Visão Geral Arquitetural

### Filosofia de Design

O Datalid 3.0 segue princípios de:

✅ **Modularidade**: Componentes independentes e substituíveis  
✅ **Configurabilidade**: Tudo via YAML  
✅ **Extensibilidade**: Fácil adicionar novos engines/modelos  
✅ **Testabilidade**: Componentes isolados e testáveis  
✅ **Performance**: Otimizado para produção  

---

## 🎯 Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────┐
│                       DATALID 3.0                           │
│                    (Application Layer)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   CLI/Make   │ │   API    │ │   Scripts    │
│   Interface  │ │  REST    │ │   Python     │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌────────────────────┐     ┌────────────────────┐
│  Pipeline Layer    │     │   Utility Layer    │
│  • FullPipeline    │     │   • Logging        │
│  • OCRPipeline     │     │   • Metrics        │
│  • Detection       │     │   • Visualization  │
└─────────┬──────────┘     └────────────────────┘
          │
    ┌─────┴─────┬─────────────┬──────────┐
    │           │             │          │
    ▼           ▼             ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│  YOLO  │ │  OCR   │ │   Pre    │ │  Post   │
│        │ │ Engines│ │  Process │ │ Process │
└────────┘ └────────┘ └──────────┘ └─────────┘
    │           │           │           │
    └───────────┴───────────┴───────────┘
                    │
            ┌───────┴────────┐
            │                │
            ▼                ▼
    ┌──────────────┐  ┌─────────────┐
    │   Models     │  │    Data     │
    │  (.pt files) │  │  (images)   │
    └──────────────┘  └─────────────┘
```

---

## 📦 Estrutura de Módulos

### 1. Core (`src/core/`)

**Responsabilidade**: Funcionalidades base do sistema

```
src/core/
├── __init__.py
├── constants.py      # Constantes globais
├── exceptions.py     # Exceções customizadas
├── config.py         # Classes de configuração
└── base_classes.py   # Classes abstratas base
```

**Componentes principais**:

```python
# constants.py
CLASS_NAMES = {0: "exp_date"}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
YOLO_MODELS = {...}

# exceptions.py
class ModelNotFoundError(Exception): ...
class PredictionError(Exception): ...
class OCRError(Exception): ...

# base_classes.py
class BaseModel(ABC):
    @abstractmethod
    def predict(self, image): ...
```

### 2. YOLO (`src/yolo/`)

**Responsabilidade**: Detecção e segmentação de regiões

```
src/yolo/
├── __init__.py
├── wrapper.py        # Wrapper unificado do YOLO
├── detector.py       # Detecção (bbox)
├── segmenter.py      # Segmentação (máscaras)
├── predictor.py      # Sistema de predição
├── trainer.py        # Sistema de treinamento
├── config.py         # Configurações YOLO
└── utils.py          # Utilidades
```

**Fluxo de execução**:

```python
# 1. Carregar modelo
from src.yolo import YOLOSegmenter
model = YOLOSegmenter("path/to/model.pt")

# 2. Predição
results = model.predict(
    image,
    conf=0.25,
    iou=0.7,
    return_masks=True,
    return_crops=True
)

# 3. Resultado
{
    'boxes': [[x1, y1, x2, y2], ...],
    'confidences': [0.87, ...],
    'masks': np.ndarray,  # Máscaras de segmentação
    'polygons': [[[x1,y1], [x2,y2], ...], ...],  # Contornos
    'crops': [np.ndarray, ...]  # Regiões extraídas
}
```

**Hierarquia de classes**:

```
YOLOWrapper (base)
    ├── YOLODetector (bbox)
    └── YOLOSegmenter (máscaras)
            ↓
    YOLOPredictor (inferência)
    YOLOTrainer (treinamento)
```

### 3. OCR (`src/ocr/`)

**Responsabilidade**: Extração de texto das regiões detectadas

```
src/ocr/
├── __init__.py
├── config.py                # Gerenciador de configs
├── engines/
│   ├── base.py             # Interface abstrata
│   ├── openocr.py          # OpenOCR (recomendado)
│   ├── parseq.py           # PARSeq
│   ├── parseq_enhanced.py  # PARSeq Enhanced
│   ├── trocr.py            # TrOCR
│   ├── easyocr.py          # EasyOCR
│   ├── paddleocr.py        # PaddleOCR
│   └── tesseract.py        # Tesseract
├── preprocessors.py         # Pré-processamento
├── postprocessors.py        # Parsing de datas
├── line_detector.py         # Detecção de linhas
├── normalizers.py           # Normalização geométrica
├── evaluator.py             # Avaliação de OCR
└── visualization.py         # Visualizações
```

**Interface unificada**:

```python
class OCREngineBase(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Carrega modelos"""
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            image: Imagem BGR/RGB
        Returns:
            (texto, confiança)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Nome do engine"""
        pass
```

**Uso**:

```python
# 1. Carregar engine
from src.ocr.engines import OpenOCREngine
engine = OpenOCREngine(config)
engine.initialize()

# 2. Extrair texto
text, confidence = engine.extract_text(crop_image)

# 3. Pós-processar
from src.ocr.postprocessors import DateParser
parser = DateParser()
dates = parser.parse(text)
```

### 4. Pipeline (`src/pipeline/`)

**Responsabilidade**: Orquestração end-to-end

```
src/pipeline/
├── __init__.py
├── base.py              # Interface base
├── full_pipeline.py     # Pipeline completo (YOLO→OCR→Parse)
├── detection.py         # Apenas detecção
└── ocr_pipeline.py      # Apenas OCR
```

**FullPipeline - Fluxo completo**:

```python
class FullPipeline(PipelineBase):
    def __init__(self, config: Dict):
        # Carrega YOLO model
        self.yolo_model = YOLO(config['detection']['model_path'])
        
        # Carrega OCR engine
        self.ocr_engine = self._load_ocr_engine()
        
        # Carrega preprocessador
        self.preprocessor = ImagePreprocessor(config)
        
        # Carrega parser
        self.date_parser = DateParser(config)
    
    def process(self, image: np.ndarray) -> Dict:
        # 1. Detecção YOLO
        detections = self._detect_regions(image)
        
        # 2. Para cada detecção
        ocr_results = []
        for detection in detections:
            crop = self._extract_crop(image, detection)
            
            # 3. Pré-processamento
            processed = self.preprocessor.process(crop)
            
            # 4. OCR
            text, conf = self.ocr_engine.extract_text(processed)
            ocr_results.append({'text': text, 'confidence': conf})
        
        # 5. Parsing e validação
        dates = []
        for ocr_result in ocr_results:
            parsed = self.date_parser.parse(ocr_result['text'])
            dates.extend(parsed)
        
        # 6. Melhor resultado
        best_date = max(dates, key=lambda x: x['confidence'])
        
        return {
            'success': True,
            'best_date': best_date,
            'all_dates': dates,
            'detections': detections,
            'ocr_results': ocr_results
        }
```

### 5. Data (`src/data/`)

**Responsabilidade**: Processamento e preparação de dados

```
src/data/
├── __init__.py
├── processor.py         # Processamento de datasets
├── augmentation.py      # Data augmentation
├── validator.py         # Validação de datasets
└── splitter.py          # Split train/val/test
```

### 6. API (`src/api/`)

**Responsabilidade**: Interface REST

```
src/api/
├── __init__.py
├── main.py              # FastAPI app
├── config.py            # Configurações da API
├── models.py            # Pydantic models
├── routes.py            # Endpoints
├── middleware.py        # Rate limiting, auth
└── dependencies.py      # Injeção de dependências
```

**Endpoints principais**:

```python
POST /process
    - Upload de imagem
    - Retorna data extraída

POST /batch
    - Upload de múltiplas imagens
    - Retorna lista de resultados

GET /health
    - Health check

GET /models
    - Lista modelos disponíveis

GET /engines
    - Lista engines OCR
```

### 7. Utils (`src/utils/`)

**Responsabilidade**: Utilitários compartilhados

```
src/utils/
├── __init__.py
├── image.py             # Manipulação de imagens
├── file.py              # I/O de arquivos
├── metrics.py           # Cálculo de métricas
├── visualization.py     # Plots e visualizações
└── logging.py           # Configuração de logs
```

---

## 🔄 Fluxo de Dados Detalhado

### Entrada: Imagem

```python
# Formatos suportados
- np.ndarray (BGR/RGB)
- PIL.Image
- str/Path (caminho do arquivo)
```

### Etapa 1: Detecção YOLO

```python
Input: np.ndarray (H, W, 3)
       ↓
┌──────────────────────┐
│   YOLO Inference     │
│   • Forward pass     │
│   • NMS (filtrar)    │
│   • Extract masks    │
└──────────┬───────────┘
           ↓
Output: {
    'boxes': [[x1,y1,x2,y2], ...],
    'confidences': [0.87, ...],
    'masks': np.ndarray (N, H, W),
    'polygons': [[[x1,y1], ...], ...]
}
```

### Etapa 2: Crop e Preprocessamento

```python
For each detection:
    Input: full_image, detection
           ↓
    ┌──────────────────┐
    │  Extract Crop    │
    │  • Apply mask    │
    │  • Minimal bbox  │
    └────────┬─────────┘
             ↓
    crop (H', W', 3)
             ↓
    ┌──────────────────┐
    │  Preprocess      │
    │  • Grayscale     │
    │  • Denoise       │
    │  • CLAHE         │
    │  • Deskew        │
    │  • Binarize      │
    └────────┬─────────┘
             ↓
    processed_crop (H', W', 1)
```

### Etapa 3: OCR

```python
Input: processed_crop
       ↓
┌──────────────────────┐
│   OCR Engine         │
│   • Feature extract  │
│   • Sequence model   │
│   • Decode           │
└──────────┬───────────┘
           ↓
Output: (text, confidence)
    e.g., ("VAL: 15/03/2025", 0.92)
```

### Etapa 4: Pós-processamento

```python
Input: text = "VAL: 15/03/2025"
       ↓
┌──────────────────────┐
│   Date Parser        │
│   • Regex matching   │
│   • Fuzzy matching   │
│   • Validation       │
│   • Normalization    │
└──────────┬───────────┘
           ↓
Output: {
    'date': datetime(2025, 3, 15),
    'format': 'DD/MM/YYYY',
    'confidence': 0.95,
    'original_text': "VAL: 15/03/2025"
}
```

### Saída: Resultado Estruturado

```json
{
  "success": true,
  "image_name": "produto.jpg",
  "processing_time": 1.23,
  "best_date": {
    "date": "15/03/2025",
    "confidence": 0.95,
    "format": "DD/MM/YYYY",
    "text": "VAL: 15/03/2025"
  },
  "all_dates": [...],
  "detections": [
    {
      "bbox": [120, 80, 450, 120],
      "confidence": 0.87,
      "has_mask": true,
      "polygon": [[x1,y1], [x2,y2], ...]
    }
  ],
  "ocr_results": [
    {
      "text": "VAL: 15/03/2025",
      "confidence": 0.92,
      "engine": "openocr"
    }
  ]
}
```

---

## 🔧 Sistema de Configuração

### Hierarquia de Configs

```
config/
├── pipeline/                    # Pipelines completos
│   ├── full_pipeline.yaml      # Pipeline padrão
│   └── pipeline_evaluation.yaml
│
├── yolo/                        # Modelos YOLO
│   ├── detection/
│   └── segmentation/
│       ├── yolov8n-seg.yaml
│       ├── yolov8s-seg.yaml
│       └── yolov8m-seg.yaml
│
├── ocr/                         # Engines OCR
│   ├── openocr.yaml
│   ├── parseq.yaml
│   ├── parseq_enhanced.yaml
│   ├── trocr.yaml
│   └── ...
│
└── preprocessing/               # Pré-processamentos
    ├── ppro-openocr.yaml
    ├── ppro-parseq.yaml
    └── ppro-minimal.yaml
```

### Exemplo de Config Pipeline

```yaml
# config/pipeline/full_pipeline.yaml
name: expiry_date_full_pipeline

# Detecção
detection:
  model_path: experiments/yolov8m_seg_best/weights/best.pt
  confidence: 0.25
  iou: 0.7
  device: 0  # GPU 0, ou 'cpu'

# OCR
ocr:
  engine: openocr
  config: config/ocr/openocr.yaml
  preprocessing: config/preprocessing/ppro-openocr.yaml

# Parsing
parsing:
  min_confidence: 0.5
  fuzzy_threshold: 0.8
  date_formats:
    - DD/MM/YYYY
    - MM/YYYY
    - DD MMM YYYY

# Output
output:
  save_visualizations: true
  save_crops: false
  save_intermediate: false
```

### Carregamento Dinâmico

```python
def load_pipeline_config(config_path: str) -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Resolve paths relativos
    if 'ocr' in config and 'config' in config['ocr']:
        ocr_config_path = resolve_path(config['ocr']['config'])
        with open(ocr_config_path) as f:
            config['ocr']['_loaded_config'] = yaml.safe_load(f)
    
    return config
```

---

## 🎯 Padrões de Design Utilizados

### 1. Strategy Pattern (OCR Engines)

```python
# Interface comum
class OCREngineBase(ABC):
    @abstractmethod
    def extract_text(self, image): ...

# Implementações concretas
class OpenOCREngine(OCREngineBase): ...
class PARSeqEngine(OCREngineBase): ...
class TrOCREngine(OCREngineBase): ...

# Uso polimórfico
engine = get_engine(config['engine'])  # Retorna qualquer implementação
text, conf = engine.extract_text(image)  # Funciona com qualquer uma
```

### 2. Factory Pattern (Criação de Pipelines)

```python
def create_pipeline(config: Dict) -> PipelineBase:
    pipeline_type = config.get('type', 'full')
    
    if pipeline_type == 'full':
        return FullPipeline(config)
    elif pipeline_type == 'detection_only':
        return DetectionPipeline(config)
    elif pipeline_type == 'ocr_only':
        return OCRPipeline(config)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
```

### 3. Singleton Pattern (Modelo YOLO)

```python
class YOLOModelSingleton:
    _instance = None
    _model = None
    
    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = YOLO(model_path)
        return cls._instance
    
    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)
```

### 4. Template Method (Pipeline Base)

```python
class PipelineBase(ABC):
    def process(self, image: np.ndarray) -> Dict:
        # Template method
        self._validate_input(image)
        results = self._execute(image)
        self._post_process(results)
        return results
    
    @abstractmethod
    def _execute(self, image) -> Dict:
        # Implementado pelas subclasses
        pass
    
    def _validate_input(self, image):
        # Comum para todos
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
    
    def _post_process(self, results):
        # Comum para todos
        results['timestamp'] = datetime.now()
```

---

## 📊 Gerenciamento de Estado

### Estado da Aplicação

```python
class ApplicationState:
    """Gerencia estado global da aplicação"""
    
    def __init__(self):
        self.loaded_models = {}      # Cache de modelos
        self.loaded_engines = {}     # Cache de engines OCR
        self.config = {}             # Configuração ativa
        self.metrics = {}            # Métricas de runtime
    
    def get_model(self, model_path: str):
        if model_path not in self.loaded_models:
            self.loaded_models[model_path] = YOLO(model_path)
        return self.loaded_models[model_path]
    
    def get_engine(self, engine_name: str):
        if engine_name not in self.loaded_engines:
            self.loaded_engines[engine_name] = create_engine(engine_name)
        return self.loaded_engines[engine_name]
```

### Cache e Performance

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Cache de imagens pré-processadas"""
    img = cv2.imread(image_path)
    return preprocess(img)

@lru_cache(maxsize=50)
def load_config(config_path: str) -> Dict:
    """Cache de configurações"""
    with open(config_path) as f:
        return yaml.safe_load(f)
```

---

## 🔍 Logging e Monitoramento

### Sistema de Logging

```python
from loguru import logger

# Configuração
logger.add(
    "logs/datalid_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Uso
logger.info("Pipeline iniciado")
logger.debug(f"Detecções: {len(detections)}")
logger.warning("Baixa confiança na detecção")
logger.error("Falha no OCR", exc_info=True)
```

### Métricas de Runtime

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'avg_time': 0.0,
            'detection_rate': 0.0
        }
    
    def record(self, result: Dict):
        self.metrics['total_images'] += 1
        if result['success']:
            self.metrics['successful'] += 1
        else:
            self.metrics['failed'] += 1
        
        # Atualizar média móvel
        self.metrics['avg_time'] = (
            self.metrics['avg_time'] * (self.metrics['total_images'] - 1) +
            result['processing_time']
        ) / self.metrics['total_images']
```

---

## 🚀 Otimizações de Performance

### 1. Batch Processing

```python
def process_batch(images: List[np.ndarray]) -> List[Dict]:
    # Agrupa para inferência em lote
    detections_batch = yolo_model.predict(
        images,
        batch_size=16  # Processa 16 imagens de uma vez
    )
    
    # Processa individualmente após detecção
    results = []
    for img, dets in zip(images, detections_batch):
        result = process_single(img, dets)
        results.append(result)
    
    return results
```

### 2. Lazy Loading

```python
class LazyOCREngine:
    def __init__(self, config):
        self.config = config
        self._engine = None  # Não carrega ainda
    
    @property
    def engine(self):
        if self._engine is None:
            logger.info("Carregando engine OCR...")
            self._engine = load_engine(self.config)
        return self._engine
    
    def extract_text(self, image):
        return self.engine.extract_text(image)  # Carrega só quando usado
```

### 3. Multiprocessing para Batch

```python
from multiprocessing import Pool

def process_directory_parallel(image_paths: List[str], num_workers: int = 4):
    with Pool(num_workers) as pool:
        results = pool.map(process_single_image, image_paths)
    return results
```

---

## 📈 Escalabilidade

### Horizontal Scaling (API)

```
Load Balancer
      │
      ├─── API Instance 1 (Docker)
      ├─── API Instance 2 (Docker)
      └─── API Instance 3 (Docker)
            │
            └─── Shared Model Storage (NFS/S3)
```

### Vertical Scaling (GPU)

```python
# Multi-GPU support
device_ids = [0, 1, 2, 3]  # 4 GPUs

# Distribuir carga
for i, image_batch in enumerate(batches):
    device = device_ids[i % len(device_ids)]
    results = model.predict(image_batch, device=device)
```

---

## 🎯 Próximos Passos

- **[Fluxo de Dados](06-DATA-FLOW.md)** - Veja o fluxo completo
- **[YOLO Detection](07-YOLO-DETECTION.md)** - Detalhes do YOLO
- **[OCR System](08-OCR-SYSTEM.md)** - Sistema de OCR

---

**Anterior: [← Teoria](05-THEORY.md) | Próximo: [Fluxo de Dados →](06-DATA-FLOW.md)**
