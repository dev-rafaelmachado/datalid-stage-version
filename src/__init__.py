"""
📦 Datalid 3.0 - Sistema de Detecção de Datas de Validade
Sistema completo para detecção e processamento de datas de validade em produtos.
"""

__version__ = "3.0.0"
__author__ = "Datalid Team"

# Importações principais
from . import api, core, data, ocr, pipeline, utils, yolo

__all__ = [
    'core',
    'data', 
    'yolo',
    'ocr',
    'pipeline',
    'api',
    'utils'
]