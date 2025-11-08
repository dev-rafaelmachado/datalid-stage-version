"""
🔧 Configurações Globais do Sistema
Centraliza todas as configurações do projeto.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configurações globais do sistema."""

    # ========================================
    # PATHS DO PROJETO
    # ========================================
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    SRC_DIR: Path = ROOT_DIR / "src"
    DATA_DIR: Path = ROOT_DIR / "data"
    CONFIG_DIR: Path = ROOT_DIR / "config"
    EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
    MODELS_DIR: Path = ROOT_DIR / "models"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

    # Dados
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # ========================================
    # CONFIGURAÇÕES DE DATASET
    # ========================================
    # Divisão padrão dos dados (personalizável)
    TRAIN_SPLIT: float = Field(default=0.7, ge=0.1, le=0.9)
    VAL_SPLIT: float = Field(default=0.2, ge=0.1, le=0.5)
    TEST_SPLIT: float = Field(default=0.1, ge=0.05, le=0.5)

    # Formatos suportados
    SUPPORTED_IMAGE_FORMATS: List[str] = [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    SUPPORTED_LABEL_FORMATS: List[str] = [".txt", ".json", ".xml"]

    # ========================================
    # CONFIGURAÇÕES DE TREINAMENTO
    # ========================================
    DEFAULT_MODEL_SIZE: str = "yolov8s"  # n, s, m, l, x
    DEFAULT_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_IMG_SIZE: int = 640
    DEFAULT_DEVICE: str = "0"  # GPU 0 ou "cpu"

    # Configurações de Learning Rate
    DEFAULT_LR0: float = 0.01
    DEFAULT_LRF: float = 0.01
    DEFAULT_MOMENTUM: float = 0.937
    DEFAULT_WEIGHT_DECAY: float = 0.0005
    DEFAULT_OPTIMIZER: str = "SGD"

    # Configurações de Treinamento
    DEFAULT_PATIENCE: int = 50
    DEFAULT_WORKERS: int = 4
    DEFAULT_CACHE: bool = False
    DEFAULT_SAVE: bool = True
    DEFAULT_SAVE_PERIOD: int = 10
    DEFAULT_EXIST_OK: bool = True

    # Loss gains
    DEFAULT_BOX_GAIN: float = 7.5
    DEFAULT_CLS_GAIN: float = 0.5
    DEFAULT_DFL_GAIN: float = 1.5

    # Data Augmentation
    DEFAULT_AUGMENTATIONS: bool = True
    DEFAULT_HSV_H: float = 0.015
    DEFAULT_HSV_S: float = 0.7
    DEFAULT_HSV_V: float = 0.4
    DEFAULT_DEGREES: float = 10.0
    DEFAULT_TRANSLATE: float = 0.1
    DEFAULT_SCALE: float = 0.5
    DEFAULT_SHEAR: float = 0.0
    DEFAULT_PERSPECTIVE: float = 0.0
    DEFAULT_FLIPUD: float = 0.0
    DEFAULT_FLIPLR: float = 0.5
    DEFAULT_MOSAIC: float = 1.0
    DEFAULT_MIXUP: float = 0.1
    DEFAULT_COPY_PASTE: float = 0.0

    # Validation
    DEFAULT_VAL: bool = True
    DEFAULT_PLOTS: bool = True
    DEFAULT_RECT: bool = False

    # ========================================
    # CONFIGURAÇÕES DE DETECÇÃO
    # ========================================
    DETECTION_CONFIDENCE: float = Field(default=0.25, ge=0.01, le=1.0)
    DETECTION_IOU_THRESHOLD: float = Field(default=0.7, ge=0.1, le=1.0)

    # Classes do dataset
    CLASS_NAMES: Dict[int, str] = {0: "exp_date"}
    NUM_CLASSES: int = 1

    # ========================================
    # CONFIGURAÇÕES DE OCR
    # ========================================
    OCR_ENGINE: str = "tesseract"  # tesseract, easyocr, paddleocr
    OCR_LANGUAGES: List[str] = ["por", "eng"]  # Português e Inglês
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

    # ========================================
    # CONFIGURAÇÕES DE API
    # ========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # ========================================
    # PRESETS DE CONFIGURAÇÃO
    # ========================================
    TRAINING_PRESETS: Dict[str, Dict] = {
        'nano': {
            'model': 'yolov8n.pt',
            'epochs': 100,
            'batch': 32,
            'patience': 50,
            'save_period': -1,
            'workers': 4,
        },
        'small': {
            'model': 'yolov8s.pt',
            'epochs': 120,
            'batch': 16,
            'patience': 40,
            'save_period': 10,
            'workers': 4,
        },
        'medium': {
            'model': 'yolov8m.pt',
            'epochs': 150,
            'batch': 8,
            'patience': 50,
            'save_period': 10,
            'workers': 4,
        },
        'nano_seg': {
            'model': 'yolov8n-seg.pt',
            'epochs': 100,
            'batch': 8,
            'patience': 50,
            'save_period': -1,
            'workers': 4,
        },
        'small_seg': {
            'model': 'yolov8s-seg.pt',
            'epochs': 120,
            'batch': 6,
            'patience': 40,
            'save_period': 10,
            'workers': 4,
        },
        'medium_seg': {
            'model': 'yolov8m-seg.pt',
            'epochs': 150,
            'batch': 4,
            'patience': 50,
            'save_period': 10,
            'workers': 4,
        }
    }

    # ========================================
    # CONFIGURAÇÕES DE LOGGING
    # ========================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    # ========================================
    # CONFIGURAÇÕES DE VISUALIZAÇÃO
    # ========================================
    BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Verde
    BBOX_THICKNESS: int = 2
    FONT_SIZE: float = 0.8
    FONT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Branco

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CORE_",  # Use CORE_ prefix to avoid conflicts with API
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars not defined in schema
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validar splits
        total_split = self.TRAIN_SPLIT + self.VAL_SPLIT + self.TEST_SPLIT
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(
                f"Splits devem somar 1.0. Atual: {total_split:.3f} "
                f"(train={self.TRAIN_SPLIT}, val={self.VAL_SPLIT}, test={self.TEST_SPLIT})"
            )

    def create_directories(self) -> None:
        """Cria diretórios necessários se não existirem."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EXPERIMENTS_DIR,
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_splits(self) -> Tuple[float, float, float]:
        """Retorna splits configurados."""
        return self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT

    def update_splits(self, train: float, val: float, test: float) -> None:
        """Atualiza divisão dos dados."""
        if abs(train + val + test - 1.0) > 0.01:
            raise ValueError(
                f"Splits devem somar 1.0. Recebido: {train + val + test:.3f}")

        self.TRAIN_SPLIT = train
        self.VAL_SPLIT = val
        self.TEST_SPLIT = test

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Config":
        """Carrega configurações de arquivo YAML."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def save_to_yaml(self, yaml_path: Path) -> None:
        """Salva configurações em arquivo YAML."""
        config_dict = self.dict()

        # Converter Path para string
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False,
                      allow_unicode=True)

    def get_training_preset(self, preset_name: str) -> Dict:
        """Obtém preset de treinamento."""
        if preset_name not in self.TRAINING_PRESETS:
            raise ValueError(
                f"Preset '{preset_name}' não encontrado. Disponíveis: {list(self.TRAINING_PRESETS.keys())}")
        return self.TRAINING_PRESETS[preset_name].copy()

    def merge_config(self, base_config: Dict, overrides: Dict) -> Dict:
        """Mescla configurações."""
        merged = base_config.copy()
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged


# Instância global
config = Config()

# Criar diretórios necessários
config.create_directories()
