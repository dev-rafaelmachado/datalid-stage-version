# 19. YAML Configuration Reference

## 📋 Overview

This document provides a complete reference for all YAML configuration files in Datalid 3.0. Understanding these configurations allows you to customize every aspect of the system's behavior.

---

## 🎯 Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Pipeline Configuration](#pipeline-configuration)
3. [YOLO Configuration](#yolo-configuration)
4. [OCR Configuration](#ocr-configuration)
5. [Preprocessing Configuration](#preprocessing-configuration)
6. [Post-processing Configuration](#post-processing-configuration)
7. [Training Configuration](#training-configuration)
8. [Environment Variables](#environment-variables)
9. [Configuration Validation](#configuration-validation)
10. [Best Practices](#best-practices)

---

## 🏗️ Configuration Architecture

### Configuration Hierarchy

```
config/
├── pipeline/              # End-to-end pipeline configs
│   ├── full_pipeline.yaml
│   └── pipeline_evaluation.yaml
│
├── yolo/                  # YOLO detection configs
│   ├── bbox/
│   └── segmentation/
│
├── ocr/                   # OCR engine configs
│   ├── default.yaml
│   ├── tesseract.yaml
│   ├── paddleocr.yaml
│   ├── easyocr.yaml
│   ├── parseq.yaml
│   ├── trocr.yaml
│   └── openocr.yaml
│
├── preprocessing/         # Image preprocessing configs
│   ├── ppro-minimal.yaml
│   ├── ppro-tesseract.yaml
│   ├── ppro-paddleocr.yaml
│   └── ppro-parseq.yaml
│
└── experiments/           # Experimental configs
    └── ocr_comparison.yaml
```

### Loading Configurations

```python
from src.pipeline.full_pipeline import FullPipeline

# 1. Load from file
pipeline = FullPipeline.from_config("config/pipeline/full_pipeline.yaml")

# 2. Load with overrides
pipeline = FullPipeline.from_config(
    "config/pipeline/full_pipeline.yaml",
    overrides={
        "ocr": {"engine": "paddleocr"},
        "yolo": {"conf_threshold": 0.3}
    }
)

# 3. Load from dictionary
config = {
    "yolo": {"model": "yolov8s-seg.pt"},
    "ocr": {"engine": "tesseract"}
}
pipeline = FullPipeline.from_dict(config)
```

---

## 🔧 Pipeline Configuration

### Full Pipeline Configuration

**File:** `config/pipeline/full_pipeline.yaml`

```yaml
# ========================================
# FULL PIPELINE CONFIGURATION
# ========================================

# Pipeline metadata
name: "full_pipeline"
version: "3.0.0"
description: "Complete date detection and OCR pipeline"

# ========================================
# YOLO DETECTION
# ========================================
yolo:
  # Model selection
  model: "yolov8s-seg.pt"  # Options: yolov8n/s/m/l/x-seg.pt
  
  # Detection thresholds
  conf_threshold: 0.25      # Confidence threshold (0.0-1.0)
  iou_threshold: 0.7        # IoU threshold for NMS (0.0-1.0)
  
  # Processing options
  device: "cuda"            # Device: "cuda", "cpu", or device index
  half: false               # Use FP16 half-precision (GPU only)
  
  # Image processing
  imgsz: 640               # Input image size (must be multiple of 32)
  augment: false           # Test-time augmentation
  
  # Output options
  max_det: 1000            # Maximum detections per image
  classes: null            # Filter by class: [0, 1, 2] or null for all
  agnostic_nms: false      # Class-agnostic NMS
  
  # Advanced options
  batch_size: 1            # Batch size for inference
  verbose: false           # Verbose output
  
# ========================================
# OCR CONFIGURATION
# ========================================
ocr:
  # Engine selection
  engine: "openocr"        # Options: tesseract, paddleocr, easyocr, 
                          #          parseq, trocr, openocr
  
  # Load engine-specific config
  config_file: "config/ocr/openocr.yaml"
  
  # Common OCR settings
  languages: ["por", "eng"]  # Language codes
  confidence_threshold: 0.6  # Minimum OCR confidence
  
  # Processing options
  batch_size: 16           # Batch size for OCR processing
  use_gpu: true            # Use GPU if available
  
  # Text cleaning
  remove_special_chars: false
  lowercase: false
  strip_whitespace: true
  
# ========================================
# PREPROCESSING
# ========================================
preprocessing:
  # Enable/disable preprocessing
  enabled: true
  
  # Load preprocessing config
  config_file: "config/preprocessing/ppro-openocr.yaml"
  
  # Quick overrides (override config_file settings)
  overrides:
    resize:
      enabled: true
      max_dimension: 2000
    
    grayscale:
      enabled: true
    
    denoise:
      enabled: true
      method: "fastNlMeans"
    
    binarize:
      enabled: true
      method: "adaptive"

# ========================================
# POST-PROCESSING
# ========================================
post_processing:
  # Date parsing
  date_parsing:
    enabled: true
    formats:
      - "%d/%m/%Y"
      - "%d-%m-%Y"
      - "%Y-%m-%d"
      - "%d.%m.%Y"
      - "%d de %B de %Y"
    
    # Date validation
    min_year: 2000
    max_year: 2050
    allow_future: true
    
  # Text validation
  text_validation:
    enabled: true
    min_length: 3
    max_length: 100
    
  # Confidence filtering
  confidence_filtering:
    enabled: true
    min_detection_confidence: 0.25
    min_ocr_confidence: 0.6
    
  # Duplicate removal
  duplicate_removal:
    enabled: true
    iou_threshold: 0.5
    text_similarity_threshold: 0.9

# ========================================
# OUTPUT CONFIGURATION
# ========================================
output:
  # Output directory
  output_dir: "outputs/pipeline"
  
  # Save options
  save_results: true
  save_debug_images: false
  save_intermediate: false
  
  # Result format
  format: "json"           # Options: json, yaml, csv
  indent: 2                # JSON indent
  
  # Image output
  annotate_images: false
  annotation_color: [0, 255, 0]  # RGB
  annotation_thickness: 2
  
# ========================================
# LOGGING
# ========================================
logging:
  level: "INFO"            # Options: DEBUG, INFO, WARNING, ERROR
  format: "detailed"       # Options: simple, detailed
  
  # File logging
  log_to_file: false
  log_file: "logs/pipeline.log"
  
  # Performance logging
  log_timing: true
  log_memory: false

# ========================================
# PERFORMANCE
# ========================================
performance:
  # Multiprocessing
  use_multiprocessing: false
  num_workers: 4
  
  # Caching
  cache_enabled: false
  cache_dir: ".cache"
  
  # Memory management
  clear_cache_after_batch: true
  max_image_size_mb: 50
```

### Evaluation Pipeline Configuration

**File:** `config/pipeline/pipeline_evaluation.yaml`

```yaml
# Extends full_pipeline.yaml with evaluation-specific settings

# Inherit from base config
extends: "config/pipeline/full_pipeline.yaml"

# Override for evaluation
yolo:
  conf_threshold: 0.15     # Lower threshold to catch more detections
  save_conf: true          # Save confidence scores

ocr:
  engine: "parseq"         # Use high-accuracy engine
  confidence_threshold: 0.5

output:
  save_results: true
  save_debug_images: true  # Save for manual inspection
  save_intermediate: true  # Save all pipeline stages

# Evaluation-specific settings
evaluation:
  ground_truth_file: "data/evaluation/ground_truth.json"
  
  metrics:
    - "precision"
    - "recall"
    - "f1_score"
    - "accuracy"
    - "iou"
  
  confusion_matrix: true
  
  output_file: "outputs/pipeline_evaluation/results.json"
```

---

## 🎯 YOLO Configuration

### Segmentation Configuration

**File:** `config/yolo/segmentation/train.yaml`

```yaml
# ========================================
# YOLO SEGMENTATION TRAINING CONFIG
# ========================================

# Model architecture
model: "yolov8s-seg.pt"    # Base model

# Dataset
data:
  path: "data/processed/v1_segment"
  train: "images/train"
  val: "images/val"
  test: "images/test"
  
  # Class names
  names:
    0: "expiry_date"
    1: "manufacture_date"
    2: "date_field"
  
  # Number of classes
  nc: 3

# Training hyperparameters
train:
  # Epochs and batch size
  epochs: 100
  batch: 16
  
  # Image size
  imgsz: 640
  
  # Optimization
  optimizer: "SGD"           # Options: SGD, Adam, AdamW
  lr0: 0.01                  # Initial learning rate
  lrf: 0.01                  # Final learning rate (lr0 * lrf)
  momentum: 0.937            # SGD momentum
  weight_decay: 0.0005       # Weight decay
  
  # Scheduler
  warmup_epochs: 3.0         # Warmup epochs
  warmup_momentum: 0.8       # Warmup momentum
  warmup_bias_lr: 0.1        # Warmup bias learning rate
  
  # Loss weights
  box: 7.5                   # Box loss weight
  cls: 0.5                   # Class loss weight
  dfl: 1.5                   # Distribution focal loss weight
  
  # Data augmentation
  hsv_h: 0.015               # HSV-Hue augmentation
  hsv_s: 0.7                 # HSV-Saturation augmentation
  hsv_v: 0.4                 # HSV-Value augmentation
  degrees: 0.0               # Rotation (+/- deg)
  translate: 0.1             # Translation (+/- fraction)
  scale: 0.5                 # Scale (+/- gain)
  shear: 0.0                 # Shear (+/- deg)
  perspective: 0.0           # Perspective (+/- fraction)
  flipud: 0.0                # Vertical flip probability
  fliplr: 0.5                # Horizontal flip probability
  mosaic: 1.0                # Mosaic augmentation probability
  mixup: 0.0                 # Mixup augmentation probability
  copy_paste: 0.0            # Copy-paste augmentation probability

# Validation
val:
  # Validation frequency
  val_period: 1              # Validate every N epochs
  
  # Validation settings
  conf: 0.001                # Confidence threshold
  iou: 0.6                   # IoU threshold for NMS
  
  # Metrics
  plots: true                # Save plots
  save_json: true            # Save results to JSON

# Hardware
device: "0"                  # GPU device: "0", "0,1,2,3", or "cpu"
workers: 8                   # Number of worker threads

# Checkpoints
save: true                   # Save checkpoints
save_period: -1              # Save every N epochs (-1 = only last)
exist_ok: false              # Overwrite existing project

# Output
project: "experiments"       # Project directory
name: "yolov8s-seg"          # Experiment name

# Other
patience: 50                 # Early stopping patience
verbose: true                # Verbose output
seed: 0                      # Random seed
deterministic: true          # Deterministic training
```

### Inference Configuration

**File:** `config/yolo/inference.yaml`

```yaml
# YOLO inference configuration

model:
  # Model path
  weights: "experiments/yolov8s-seg/weights/best.pt"
  
  # Device
  device: "cuda"
  half: false

inference:
  # Thresholds
  conf: 0.25
  iou: 0.7
  
  # Image size
  imgsz: 640
  
  # Processing
  augment: false
  agnostic_nms: false
  max_det: 1000
  
  # Output
  save_txt: false
  save_conf: false
  save_crop: false
  hide_labels: false
  hide_conf: false
  
  # Visualization
  line_thickness: 2
  visualize: false

batch:
  # Batch processing
  batch_size: 1
  stream: false
```

---

## 📖 OCR Configuration

### Tesseract Configuration

**File:** `config/ocr/tesseract.yaml`

```yaml
# ========================================
# TESSERACT OCR CONFIGURATION
# ========================================

engine: "tesseract"

# Tesseract-specific settings
tesseract:
  # Tesseract executable path (optional)
  tesseract_cmd: null        # Auto-detect if null
  
  # Language
  lang: "por+eng"            # Language codes separated by +
  
  # Page Segmentation Mode (PSM)
  # 0  = Orientation and script detection (OSD) only
  # 1  = Automatic page segmentation with OSD
  # 3  = Fully automatic page segmentation (default)
  # 4  = Assume single column of text
  # 5  = Assume single uniform block of vertically aligned text
  # 6  = Assume single uniform block of text
  # 7  = Treat image as single text line
  # 8  = Treat image as single word
  # 10 = Treat image as single character
  # 11 = Sparse text. Find as much text as possible
  # 13 = Raw line. Treat image as single text line
  psm: 6
  
  # OCR Engine Mode (OEM)
  # 0 = Legacy engine only
  # 1 = Neural nets LSTM engine only
  # 2 = Legacy + LSTM engines
  # 3 = Default, based on what is available
  oem: 1
  
  # Custom configuration
  config: ""                 # Additional Tesseract config string
  
  # Timeout
  timeout: 30                # Timeout in seconds

# Processing options
processing:
  # Confidence threshold
  min_confidence: 60         # 0-100
  
  # Text filtering
  filter_empty: true
  filter_short: true
  min_text_length: 3
  
  # Whitelist/Blacklist
  whitelist: null            # Characters to accept (null = all)
  blacklist: null            # Characters to reject

# Output options
output:
  # Output format
  format: "text"             # Options: text, dict, data
  
  # Include metadata
  include_confidence: true
  include_bbox: true
```

### PaddleOCR Configuration

**File:** `config/ocr/paddleocr.yaml`

```yaml
# ========================================
# PADDLEOCR CONFIGURATION
# ========================================

engine: "paddleocr"

# PaddleOCR-specific settings
paddleocr:
  # Language
  lang: "pt"                 # pt, en, ch, etc.
  
  # GPU settings
  use_gpu: true
  gpu_mem: 2000              # GPU memory limit (MB)
  
  # Detection model
  det_model_dir: null        # Path to detection model (null = download)
  det_algorithm: "DB"        # Detection algorithm
  det_limit_side_len: 960    # Max side length for detection
  det_limit_type: "max"      # Limit type: max, min
  
  # Detection thresholds
  det_db_thresh: 0.3         # Binary threshold
  det_db_box_thresh: 0.6     # Box threshold
  det_db_unclip_ratio: 1.5   # Unclip ratio
  
  # Recognition model
  rec_model_dir: null        # Path to recognition model
  rec_algorithm: "CRNN"      # Recognition algorithm
  rec_image_shape: "3, 32, 320"  # Recognition image shape
  rec_batch_num: 6           # Recognition batch size
  
  # Angle classification
  use_angle_cls: true        # Use angle classifier
  cls_model_dir: null        # Path to angle classifier model
  cls_thresh: 0.9            # Angle classification threshold
  cls_batch_num: 6           # Angle classification batch size
  
  # Other options
  use_space_char: true       # Use space character
  drop_score: 0.5            # Drop detection below this score
  
  # Performance
  use_mp: false              # Use multiprocessing
  total_process_num: 1       # Number of processes
  
  # Output
  show_log: false            # Show log

# Processing options
processing:
  # Text filtering
  min_confidence: 0.6
  filter_empty: true
  min_text_length: 2
  
  # Post-processing
  remove_whitespace: false
  lowercase: false

# Advanced options
advanced:
  # DET model advanced options
  det_db_score_mode: "fast"  # fast, slow
  det_east_score_thresh: 0.8
  det_east_cover_thresh: 0.1
  det_east_nms_thresh: 0.2
  
  # REC model advanced options
  rec_char_dict_path: null
  use_pdserving: false
  warmup: false
```

### EasyOCR Configuration

**File:** `config/ocr/easyocr.yaml`

```yaml
# ========================================
# EASYOCR CONFIGURATION
# ========================================

engine: "easyocr"

# EasyOCR-specific settings
easyocr:
  # Languages
  languages: ["pt", "en"]    # List of language codes
  
  # GPU settings
  gpu: true                  # Use GPU
  
  # Model directory
  model_storage_directory: null  # Path to store models (null = default)
  download_enabled: true     # Allow downloading models
  
  # Recognition settings
  decoder: "beamsearch"      # Options: greedy, beamsearch, wordbeamsearch
  beamWidth: 5               # Beam width for beamsearch
  
  # Detection settings
  text_threshold: 0.7        # Text confidence threshold
  low_text: 0.4              # Low text threshold
  link_threshold: 0.4        # Link threshold
  canvas_size: 2560          # Max canvas size
  mag_ratio: 1.5             # Magnification ratio
  
  # Other options
  paragraph: false           # Combine results into paragraph
  batch_size: 1              # Batch size
  workers: 0                 # Number of workers
  allowlist: null            # Allowed characters
  blocklist: null            # Blocked characters
  detail: 1                  # Detail level: 0, 1
  
  # Advanced
  rotation_info: null        # Rotation information
  contrast_ths: 0.1          # Contrast threshold
  adjust_contrast: 0.5       # Contrast adjustment
  filter_ths: 0.003          # Filter threshold
  y_ths: 0.5                 # Y-axis threshold
  x_ths: 1.0                 # X-axis threshold
  slope_ths: 0.1             # Slope threshold

# Processing options
processing:
  min_confidence: 0.6
  filter_empty: true
  min_text_length: 2
```

### PARSeq Configuration

**File:** `config/ocr/parseq.yaml`

```yaml
# ========================================
# PARSEQ CONFIGURATION
# ========================================

engine: "parseq"

# PARSeq-specific settings
parseq:
  # Model
  model_path: "models/parseq/parseq-base.ckpt"
  charset: "94charset"       # Character set
  
  # Device
  device: "cuda"
  
  # Processing
  batch_size: 16
  img_size: [32, 128]        # [height, width]
  
  # Decoding
  max_label_length: 25
  decode_ar: true            # Autoregressive decoding
  refine_iters: 1            # Refinement iterations
  
  # Data augmentation (inference)
  keep_ratio: true           # Keep aspect ratio
  pad: true                  # Pad images

# Processing options
processing:
  min_confidence: 0.7
  filter_empty: true
  min_text_length: 2
  
  # Enhanced features
  ensemble: false            # Use ensemble of models
  num_models: 3              # Number of models in ensemble
```

### TrOCR Configuration

**File:** `config/ocr/trocr.yaml`

```yaml
# ========================================
# TROCR CONFIGURATION
# ========================================

engine: "trocr"

# TrOCR-specific settings
trocr:
  # Model
  model_name: "microsoft/trocr-base-handwritten"
  # Options:
  # - microsoft/trocr-base-handwritten
  # - microsoft/trocr-large-handwritten
  # - microsoft/trocr-base-printed
  # - microsoft/trocr-large-printed
  
  # Processor
  processor_name: "microsoft/trocr-base-handwritten"
  
  # Device
  device: "cuda"
  
  # Generation parameters
  max_length: 512
  num_beams: 4
  early_stopping: true
  
  # Batch processing
  batch_size: 8
  
  # Memory optimization
  use_cache: true
  low_cpu_mem_usage: true

# Processing options
processing:
  min_confidence: 0.8
  filter_empty: true
  min_text_length: 2
```

### OpenOCR Configuration

**File:** `config/ocr/openocr.yaml`

```yaml
# ========================================
# OPENOCR CONFIGURATION
# ========================================

engine: "openocr"

# OpenOCR-specific settings
openocr:
  # Model
  model_size: "base"         # Options: small, base, large
  model_path: null           # Custom model path
  
  # Device
  device: "cuda"
  use_fp16: false            # Use FP16 precision
  
  # Processing
  batch_size: 16
  image_size: [32, 128]
  
  # Recognition
  max_seq_length: 25
  beam_width: 5
  
  # Language
  language: "latin"          # Character set: latin, arabic, chinese, etc.

# Processing options
processing:
  min_confidence: 0.7
  filter_empty: true
  min_text_length: 2
```

---

## 🖼️ Preprocessing Configuration

### Minimal Preprocessing

**File:** `config/preprocessing/ppro-minimal.yaml`

```yaml
# ========================================
# MINIMAL PREPROCESSING
# ========================================
# Fast preprocessing for good quality images

preprocessing:
  enabled: true
  
  # Pipeline steps (executed in order)
  steps:
    # 1. Resize (if needed)
    - name: "resize"
      enabled: true
      params:
        max_dimension: 2000    # Max width or height
        maintain_aspect: true
        interpolation: "LANCZOS"  # NEAREST, BILINEAR, BICUBIC, LANCZOS
    
    # 2. Normalize
    - name: "normalize"
      enabled: true
      params:
        method: "standard"     # standard, minmax, robust
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
```

### Standard Preprocessing (Tesseract)

**File:** `config/preprocessing/ppro-tesseract.yaml`

```yaml
# ========================================
# STANDARD PREPROCESSING FOR TESSERACT
# ========================================

preprocessing:
  enabled: true
  
  steps:
    # 1. Resize
    - name: "resize"
      enabled: true
      params:
        max_dimension: 2000
        maintain_aspect: true
        interpolation: "LANCZOS"
    
    # 2. Convert to grayscale
    - name: "grayscale"
      enabled: true
      params:
        method: "luminosity"   # luminosity, average, desaturation
    
    # 3. Denoise
    - name: "denoise"
      enabled: true
      params:
        method: "fastNlMeans"  # fastNlMeans, bilateral, gaussian
        h: 10                  # Filter strength
        templateWindowSize: 7
        searchWindowSize: 21
    
    # 4. Binarization
    - name: "binarize"
      enabled: true
      params:
        method: "adaptive"     # otsu, adaptive, sauvola
        block_size: 11
        c: 2
    
    # 5. Deskew
    - name: "deskew"
      enabled: true
      params:
        angle_range: [-10, 10]
        angle_step: 0.1
    
    # 6. Remove borders
    - name: "border_removal"
      enabled: true
      params:
        threshold: 50
        kernel_size: 5
```

### Aggressive Preprocessing

**File:** `config/preprocessing/ppro-aggressive.yaml`

```yaml
# ========================================
# AGGRESSIVE PREPROCESSING
# ========================================
# For poor quality or degraded images

preprocessing:
  enabled: true
  
  steps:
    # 1. Resize with upscaling
    - name: "resize"
      enabled: true
      params:
        max_dimension: 3000
        min_dimension: 1500    # Upscale if smaller
        maintain_aspect: true
        interpolation: "LANCZOS"
    
    # 2. Super resolution (optional, slow)
    - name: "super_resolution"
      enabled: false
      params:
        scale: 2
        model: "ESRGAN"
    
    # 3. Grayscale
    - name: "grayscale"
      enabled: true
      params:
        method: "luminosity"
    
    # 4. Strong denoising
    - name: "denoise"
      enabled: true
      params:
        method: "bilateral"
        d: 9
        sigmaColor: 75
        sigmaSpace: 75
    
    # 5. Contrast enhancement
    - name: "contrast_enhancement"
      enabled: true
      params:
        method: "CLAHE"        # CLAHE, histogram_eq, adaptive
        clip_limit: 2.0
        tile_grid_size: [8, 8]
    
    # 6. Sharpening
    - name: "sharpen"
      enabled: true
      params:
        kernel_size: 3
        sigma: 1.0
        amount: 1.5
    
    # 7. Binarization
    - name: "binarize"
      enabled: true
      params:
        method: "sauvola"
        window_size: 15
        k: 0.2
    
    # 8. Morphological operations
    - name: "morphology"
      enabled: true
      params:
        operations:
          - type: "opening"
            kernel_size: 2
          - type: "closing"
            kernel_size: 2
    
    # 9. Deskew
    - name: "deskew"
      enabled: true
      params:
        angle_range: [-20, 20]
        angle_step: 0.1
    
    # 10. Border removal
    - name: "border_removal"
      enabled: true
      params:
        threshold: 30
        kernel_size: 5
```

---

## 🔧 Post-processing Configuration

### Date Parsing Configuration

```yaml
# ========================================
# DATE PARSING AND VALIDATION
# ========================================

post_processing:
  # Date format recognition
  date_formats:
    # Standard formats
    - format: "%d/%m/%Y"
      priority: 1
      regex: '\d{2}/\d{2}/\d{4}'
      
    - format: "%d-%m-%Y"
      priority: 2
      regex: '\d{2}-\d{2}-\d{4}'
      
    - format: "%Y-%m-%d"
      priority: 3
      regex: '\d{4}-\d{2}-\d{2}'
      
    - format: "%d.%m.%Y"
      priority: 4
      regex: '\d{2}\.\d{2}\.\d{4}'
    
    # Extended formats (Portuguese)
    - format: "%d de %B de %Y"
      priority: 5
      regex: '\d{1,2}\s+de\s+\w+\s+de\s+\d{4}'
      locale: "pt_BR"
      
    - format: "%d %B %Y"
      priority: 6
      regex: '\d{1,2}\s+\w+\s+\d{4}'
      
    # Abbreviated formats
    - format: "%d/%m/%y"
      priority: 7
      regex: '\d{2}/\d{2}/\d{2}'
      century_cutoff: 50  # 00-50 = 2000-2050, 51-99 = 1951-1999
  
  # Date validation
  validation:
    # Year range
    min_year: 1900
    max_year: 2100
    
    # Allow future dates
    allow_future: true
    max_future_years: 50
    
    # Allow past dates
    allow_past: true
    min_past_years: 100
    
    # Sanity checks
    check_month_range: true   # 1-12
    check_day_range: true     # 1-31 (with month validation)
    check_leap_year: true
  
  # Extraction patterns
  extraction:
    # Context keywords
    keywords:
      expiry: ["validade", "vencimento", "expira", "válido até", "vence"]
      manufacture: ["fabricação", "produção", "fab", "mfg"]
      issue: ["emissão", "data", "issue"]
    
    # Context radius (characters before/after date)
    context_before: 30
    context_after: 10
    
    # Multiple dates
    prefer_latest: false      # If multiple dates, prefer latest
    return_all: true          # Return all found dates
```

---

## 🔬 Training Configuration

### Data Augmentation

```yaml
# ========================================
# DATA AUGMENTATION CONFIGURATION
# ========================================

augmentation:
  # Geometric transformations
  geometric:
    # Rotation
    rotate:
      enabled: true
      angle_range: [-15, 15]
      probability: 0.5
      
    # Flipping
    flip:
      enabled: true
      horizontal: 0.5        # Probability
      vertical: 0.0
      
    # Scaling
    scale:
      enabled: true
      scale_range: [0.8, 1.2]
      probability: 0.5
      
    # Translation
    translate:
      enabled: true
      translate_range: [-0.1, 0.1]  # Fraction of image size
      probability: 0.5
      
    # Shear
    shear:
      enabled: true
      shear_range: [-5, 5]
      probability: 0.3
      
    # Perspective
    perspective:
      enabled: true
      distortion_scale: 0.2
      probability: 0.3
  
  # Color transformations
  color:
    # Brightness
    brightness:
      enabled: true
      factor_range: [0.7, 1.3]
      probability: 0.5
      
    # Contrast
    contrast:
      enabled: true
      factor_range: [0.7, 1.3]
      probability: 0.5
      
    # Saturation
    saturation:
      enabled: true
      factor_range: [0.7, 1.3]
      probability: 0.5
      
    # Hue
    hue:
      enabled: true
      delta_range: [-0.05, 0.05]
      probability: 0.5
  
  # Noise and blur
  noise:
    # Gaussian noise
    gaussian:
      enabled: true
      mean: 0
      std_range: [0, 25]
      probability: 0.3
      
    # Salt and pepper
    salt_pepper:
      enabled: true
      amount: 0.01
      probability: 0.2
      
    # Blur
    blur:
      enabled: true
      kernel_size_range: [3, 7]
      probability: 0.3
  
  # Advanced augmentations
  advanced:
    # Mosaic
    mosaic:
      enabled: true
      probability: 0.5
      
    # Mixup
    mixup:
      enabled: false
      alpha: 0.5
      probability: 0.3
      
    # Copy-paste
    copy_paste:
      enabled: false
      probability: 0.3
```

---

## 🌍 Environment Variables

### Core Environment Variables

```bash
# ========================================
# DATALID ENVIRONMENT VARIABLES
# ========================================

# Core settings
DATALID_ENV=production                    # development, production, testing
DATALID_DEBUG=false                       # Enable debug mode

# Paths
DATALID_ROOT=/path/to/datalid3.0         # Project root
DATALID_CONFIG_DIR=config                # Configuration directory
DATALID_DATA_DIR=data                    # Data directory
DATALID_OUTPUT_DIR=outputs               # Output directory
DATALID_MODELS_DIR=models                # Models directory

# YOLO settings
YOLO_MODEL=yolov8s-seg.pt                # Default YOLO model
YOLO_DEVICE=cuda                         # Device: cuda, cpu, or index
YOLO_CONF=0.25                           # Confidence threshold
YOLO_IOU=0.7                             # IOU threshold

# OCR settings
OCR_ENGINE=openocr                       # Default OCR engine
OCR_LANGUAGES=por,eng                    # Comma-separated languages
OCR_CONFIDENCE=0.6                       # Confidence threshold
OCR_BATCH_SIZE=16                        # Batch size

# Preprocessing
PREPROCESSING_ENABLED=true               # Enable preprocessing
PREPROCESSING_CONFIG=ppro-openocr.yaml   # Config file

# Performance
NUM_WORKERS=4                            # Number of worker threads
BATCH_SIZE=16                            # Default batch size
USE_GPU=true                             # Use GPU if available
GPU_ID=0                                 # GPU device ID

# Caching
CACHE_ENABLED=false                      # Enable caching
CACHE_DIR=.cache                         # Cache directory

# Logging
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE=false                        # Log to file
LOG_FILE=logs/datalid.log               # Log file path

# API settings (if using API)
API_HOST=0.0.0.0                        # API host
API_PORT=8000                           # API port
API_WORKERS=4                           # Number of API workers
API_KEY=your_secret_key_here            # API key for authentication

# Database (if using)
DB_HOST=localhost                        # Database host
DB_PORT=5432                            # Database port
DB_NAME=datalid                         # Database name
DB_USER=datalid                         # Database user
DB_PASSWORD=password                     # Database password
```

### Loading Environment Variables

```python
# Load from .env file
from dotenv import load_dotenv
import os

load_dotenv()

# Access variables
model_path = os.getenv("YOLO_MODEL", "yolov8s-seg.pt")
ocr_engine = os.getenv("OCR_ENGINE", "openocr")
debug_mode = os.getenv("DATALID_DEBUG", "false").lower() == "true"
```

---

## ✅ Configuration Validation

### Validation Schema

```python
"""
Configuration validation using Pydantic.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union

class YOLOConfig(BaseModel):
    """YOLO configuration schema."""
    model: str = Field(..., description="Model file path")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.7, ge=0.0, le=1.0)
    device: str = Field("cuda", regex="^(cuda|cpu|\\d+)$")
    imgsz: int = Field(640, multiple_of=32)
    
    @validator("model")
    def validate_model(cls, v):
        """Validate model file exists."""
        from pathlib import Path
        if not Path(v).exists():
            raise ValueError(f"Model file not found: {v}")
        return v

class OCRConfig(BaseModel):
    """OCR configuration schema."""
    engine: str = Field(..., regex="^(tesseract|paddleocr|easyocr|parseq|trocr|openocr)$")
    languages: List[str] = Field(default_factory=lambda: ["por", "eng"])
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    batch_size: int = Field(16, ge=1, le=128)
    use_gpu: bool = True

class PipelineConfig(BaseModel):
    """Full pipeline configuration schema."""
    name: str
    version: str
    yolo: YOLOConfig
    ocr: OCRConfig
    # ... other fields

# Validate configuration
def validate_config(config_dict: dict) -> PipelineConfig:
    """Validate configuration dictionary."""
    try:
        config = PipelineConfig(**config_dict)
        return config
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise
```

---

## 💡 Best Practices

### 1. Configuration Organization

```yaml
# ✅ Good: Organized, commented, logical grouping
model:
  # Core settings
  weights: "best.pt"
  device: "cuda"
  
  # Thresholds
  conf: 0.25
  iou: 0.7

# ❌ Bad: Disorganized, no comments
weights: "best.pt"
conf: 0.25
device: "cuda"
iou: 0.7
```

### 2. Use Inheritance

```yaml
# base_config.yaml
base: &base
  device: "cuda"
  batch_size: 16

# specific_config.yaml
<<: *base
conf: 0.3  # Override specific value
```

### 3. Environment-Specific Configs

```
config/
├── base.yaml            # Base configuration
├── development.yaml     # Development overrides
├── production.yaml      # Production overrides
└── testing.yaml         # Testing overrides
```

### 4. Validate Before Use

```python
from src.utils.config import load_and_validate_config

# Always validate
config = load_and_validate_config("config/pipeline/full_pipeline.yaml")

# Don't use raw YAML without validation
# config = yaml.safe_load(open("config.yaml"))  # ❌
```

### 5. Document Defaults

```yaml
# Always document what each field does and its default
threshold: 0.25  # Confidence threshold (0.0-1.0), default: 0.25
```

---

## 📚 Related Documentation

- [02. Installation](02-INSTALLATION.md) - Setup and configuration
- [11. Full Pipeline](11-FULL-PIPELINE.md) - Pipeline usage
- [13. YOLO Training](13-YOLO-TRAINING.md) - Training configurations
- [21. Optimization](21-OPTIMIZATION.md) - Performance tuning

---

**Next Steps:**
- [Configure your first pipeline](11-FULL-PIPELINE.md)
- [Optimize performance](21-OPTIMIZATION.md)
- [See examples](23-EXAMPLES.md)
