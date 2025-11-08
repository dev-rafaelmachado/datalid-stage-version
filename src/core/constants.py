"""
📏 Constantes Globais do Sistema
Define todas as constantes usadas no projeto.
"""

from typing import Dict, List, Tuple

import cv2

# ========================================
# CONSTANTES DE DATASET
# ========================================

# Classes do modelo
CLASS_NAMES = {
    0: "exp_date"
}

CLASS_COLORS = {
    0: (0, 255, 0),  # Verde para data de validade
}

# Extensões de arquivo suportadas
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
LABEL_EXTENSIONS = {'.txt', '.json', '.xml', '.yaml', '.yml'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

# ========================================
# CONSTANTES DE YOLO
# ========================================

# Tamanhos de entrada padrão
IMAGE_SIZES = {
    'small': 416,
    'medium': 512,
    'large': 640,
    'xlarge': 1024
}

# Modelos YOLO disponíveis
YOLO_MODELS = {
    'detection': {
        'nano': 'yolov8n.pt',
        'small': 'yolov8s.pt',
        'medium': 'yolov8m.pt',
        'large': 'yolov8l.pt',
        'xlarge': 'yolov8x.pt'
    },
    'segmentation': {
        'nano': 'yolov8n-seg.pt',
        'small': 'yolov8s-seg.pt',
        'medium': 'yolov8m-seg.pt',
        'large': 'yolov8l-seg.pt',
        'xlarge': 'yolov8x-seg.pt'
    }
}

# Configurações recomendadas por tamanho de modelo
MODEL_CONFIGS = {
    'nano': {
        'batch_size': 32,
        'workers': 4,
        'memory_gb': 2
    },
    'small': {
        'batch_size': 16,
        'workers': 4,
        'memory_gb': 4
    },
    'medium': {
        'batch_size': 8,
        'workers': 4,
        'memory_gb': 6
    },
    'large': {
        'batch_size': 4,
        'workers': 2,
        'memory_gb': 8
    },
    'xlarge': {
        'batch_size': 2,
        'workers': 2,
        'memory_gb': 12
    }
}

# ========================================
# CONSTANTES DE OCR
# ========================================

# Configurações OCR
OCR_ENGINES = {
    'tesseract': {
        'config': '--oem 3 --psm 6',
        'languages': ['por', 'eng']
    },
    'easyocr': {
        'languages': ['pt', 'en'],
        'gpu': True
    },
    'paddleocr': {
        'lang': 'pt',
        'use_gpu': True
    }
}

# Padrões de data
DATE_PATTERNS = [
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',           # DD/MM/YYYY ou D/M/YYYY
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',           # DD-MM-YYYY ou D-M-YYYY
    r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',         # DD.MM.YYYY ou D.M.YYYY
    r'\b\d{1,2}/\d{1,2}/\d{2}\b',           # DD/MM/YY ou D/M/YY
    r'\b\d{4}/\d{1,2}/\d{1,2}\b',           # YYYY/MM/DD
    r'\b\d{4}-\d{1,2}-\d{1,2}\b',           # YYYY-MM-DD
]

# Meses em português (para OCR)
MONTHS_PT = [
    'janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
    'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro',
    'jan', 'fev', 'mar', 'abr', 'mai', 'jun',
    'jul', 'ago', 'set', 'out', 'nov', 'dez'
]

# ========================================
# CONSTANTES DE VISUALIZAÇÃO
# ========================================

# Cores em BGR (OpenCV)
COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'cyan': (255, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'pink': (203, 192, 255),
    'lime': (0, 255, 0),
    'navy': (128, 0, 0),
    'teal': (128, 128, 0)
}

# Configurações de fonte
FONT_CONFIGS = {
    'small': {
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'scale': 0.5,
        'thickness': 1
    },
    'medium': {
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'scale': 0.7,
        'thickness': 2
    },
    'large': {
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'scale': 1.0,
        'thickness': 2
    }
}

# ========================================
# CONSTANTES DE API
# ========================================

# Status codes customizados
API_STATUS_CODES = {
    'SUCCESS': 200,
    'BAD_REQUEST': 400,
    'NOT_FOUND': 404,
    'INTERNAL_ERROR': 500,
    'MODEL_ERROR': 502,
    'TIMEOUT': 504
}

# Limites de arquivo
FILE_LIMITS = {
    'max_size_mb': 10,
    'max_width': 4096,
    'max_height': 4096,
    'min_width': 32,
    'min_height': 32
}

# ========================================
# CONSTANTES DE MÉTRICAS
# ========================================

# Thresholds para métricas
METRIC_THRESHOLDS = {
    'iou_levels': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'confidence_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'map_50': 0.5,
    'map_50_95': [0.5, 0.95]
}

# ========================================
# CONSTANTES DE PROCESSAMENTO
# ========================================

# Configurações de paralelismo
PROCESSING_CONFIGS = {
    'max_workers': 4,
    'chunk_size': 100,
    'timeout_seconds': 300,
    'retry_attempts': 3
}

# Configurações de cache
CACHE_CONFIGS = {
    'enable_cache': True,
    'cache_size_mb': 512,
    'cache_ttl_hours': 24
}

# ========================================
# CONSTANTES DE HARDWARE
# ========================================

# Configurações por tipo de GPU
GPU_CONFIGS = {
    'GTX_1660': {
        'memory_gb': 6,
        'recommended_batch': 16,
        'max_image_size': 640
    },
    'RTX_3060': {
        'memory_gb': 12,
        'recommended_batch': 32,
        'max_image_size': 1024
    },
    'RTX_4090': {
        'memory_gb': 24,
        'recommended_batch': 64,
        'max_image_size': 1280
    }
}

# Configurações para CPU
CPU_CONFIGS = {
    'batch_size': 1,
    'workers': 2,
    'image_size': 416
}

# ========================================
# MENSAGENS DE ERRO PADRONIZADAS
# ========================================

ERROR_MESSAGES = {
    'file_not_found': "Arquivo não encontrado: {path}",
    'invalid_format': "Formato de arquivo inválido: {format}. Suportados: {supported}",
    'model_not_loaded': "Modelo não foi carregado. Execute load_model() primeiro.",
    'gpu_not_available': "GPU não disponível. Usando CPU (mais lento).",
    'insufficient_memory': "Memória insuficiente. Reduza o batch_size.",
    'invalid_image': "Imagem inválida ou corrompida: {path}",
    'empty_dataset': "Dataset vazio. Verifique o caminho: {path}",
    'invalid_split': "Divisão inválida. Train+Val+Test deve somar 1.0"
}

# ========================================
# CONFIGURAÇÕES DE DATASET SPLITS
# ========================================

# Splits padrão (personalizáveis)
DEFAULT_SPLITS = {
    'train': 0.7,
    'val': 0.2,
    'test': 0.1
}

# Splits alternativos pré-definidos
PRESET_SPLITS = {
    'balanced': {'train': 0.6, 'val': 0.2, 'test': 0.2},
    'research': {'train': 0.8, 'val': 0.1, 'test': 0.1},
    'production': {'train': 0.7, 'val': 0.3, 'test': 0.0},
    'quick_test': {'train': 0.5, 'val': 0.25, 'test': 0.25}
}

# ========================================
# CONFIGURAÇÕES DE LOGGING
# ========================================

LOG_FORMATS = {
    'simple': "{time:HH:mm:ss} | {level} | {message}",
    'detailed': "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    'json': '{"time": "{time:YYYY-MM-DD HH:mm:ss}", "level": "{level}", "message": "{message}"}'
}

LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}
