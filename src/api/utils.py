"""
🛠️ Utilitários da API
Funções auxiliares para validação, conversão, etc.
"""

import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

# ========================================
# VALIDAÇÃO
# ========================================

def validate_file_size(file_size: int, max_size: int) -> bool:
    """
    Valida tamanho de arquivo.
    
    Args:
        file_size: Tamanho em bytes
        max_size: Tamanho máximo em bytes
        
    Returns:
        True se válido
    """
    return file_size <= max_size


def validate_image_format(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Valida formato de imagem.
    
    Args:
        filename: Nome do arquivo
        allowed_extensions: Lista de extensões permitidas
        
    Returns:
        True se válido
    """
    ext = Path(filename).suffix.lower()
    return ext in [e.lower() for e in allowed_extensions]


def validate_batch_size(num_files: int, max_batch: int) -> bool:
    """
    Valida tamanho de batch.
    
    Args:
        num_files: Número de arquivos
        max_batch: Máximo permitido
        
    Returns:
        True se válido
    """
    return num_files <= max_batch


# ========================================
# CONVERSÃO DE IMAGENS
# ========================================

def decode_image(content: bytes) -> np.ndarray:
    """
    Decodifica imagem de bytes para numpy array.
    
    Args:
        content: Bytes da imagem
        
    Returns:
        Imagem BGR (numpy array)
        
    Raises:
        ValueError: Se não conseguir decodificar
    """
    try:
        # Tentar com OpenCV primeiro (mais rápido)
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            # Fallback para PIL
            pil_image = Image.open(io.BytesIO(content))
            # Converter para RGB se necessário
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Converter para numpy array (RGB)
            image = np.array(pil_image)
            # Converter RGB para BGR (OpenCV usa BGR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if image is None:
            raise ValueError("Não foi possível decodificar a imagem")
        
        return image
        
    except Exception as e:
        logger.error(f"Erro ao decodificar imagem: {e}")
        raise ValueError(f"Erro ao decodificar imagem: {str(e)}")


def encode_image(image: np.ndarray, format: str = 'jpg', quality: int = 95) -> bytes:
    """
    Codifica imagem numpy array para bytes.
    
    Args:
        image: Imagem BGR (numpy array)
        format: Formato ('jpg', 'png')
        quality: Qualidade (1-100, apenas para JPEG)
        
    Returns:
        Bytes da imagem
    """
    ext = f'.{format.lower()}'
    
    if ext == '.jpg' or ext == '.jpeg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        encode_params = []
    
    success, buffer = cv2.imencode(ext, image, encode_params)
    
    if not success:
        raise ValueError(f"Erro ao codificar imagem no formato {format}")
    
    return buffer.tobytes()


def resize_image(
    image: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Redimensiona imagem.
    
    Args:
        image: Imagem original
        max_width: Largura máxima
        max_height: Altura máxima
        keep_aspect: Manter aspect ratio
        
    Returns:
        Imagem redimensionada
    """
    h, w = image.shape[:2]
    
    if max_width is None and max_height is None:
        return image
    
    if keep_aspect:
        # Calcular novo tamanho mantendo aspect ratio
        scale = 1.0
        
        if max_width and w > max_width:
            scale = min(scale, max_width / w)
        
        if max_height and h > max_height:
            scale = min(scale, max_height / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = max_width or w
        new_h = max_height or h
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


# ========================================
# MANIPULAÇÃO DE ARQUIVOS
# ========================================

def get_file_size_mb(content: bytes) -> float:
    """
    Retorna tamanho de arquivo em MB.
    
    Args:
        content: Conteúdo do arquivo
        
    Returns:
        Tamanho em MB
    """
    return len(content) / (1024 * 1024)


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza nome de arquivo.
    
    Args:
        filename: Nome original
        
    Returns:
        Nome sanitizado
    """
    # Remover caracteres problemáticos
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limitar tamanho
    name = Path(filename).stem[:100]
    ext = Path(filename).suffix
    
    return f"{name}{ext}"


def create_temp_file(content: bytes, suffix: str = '.jpg') -> Path:
    """
    Cria arquivo temporário.
    
    Args:
        content: Conteúdo do arquivo
        suffix: Extensão do arquivo
        
    Returns:
        Path do arquivo temporário
    """
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()
    
    return Path(temp_file.name)


# ========================================
# FORMATAÇÃO DE DADOS
# ========================================

def format_bytes(size_bytes: int) -> str:
    """
    Formata tamanho em bytes para string legível.
    
    Args:
        size_bytes: Tamanho em bytes
        
    Returns:
        String formatada (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def format_duration(seconds: float) -> str:
    """
    Formata duração em segundos para string legível.
    
    Args:
        seconds: Duração em segundos
        
    Returns:
        String formatada (ex: "1.5s", "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m"


# ========================================
# HASH E CACHE
# ========================================

import hashlib


def compute_file_hash(content: bytes) -> str:
    """
    Computa hash SHA256 de arquivo.
    
    Args:
        content: Conteúdo do arquivo
        
    Returns:
        Hash hexadecimal
    """
    return hashlib.sha256(content).hexdigest()


def compute_image_hash(image: np.ndarray) -> str:
    """
    Computa hash de imagem numpy array.
    
    Args:
        image: Imagem
        
    Returns:
        Hash hexadecimal
    """
    # Usar array bytes
    return hashlib.sha256(image.tobytes()).hexdigest()


# ========================================
# LOGGING
# ========================================

def log_request_info(
    method: str,
    path: str,
    client_ip: str,
    request_id: str,
    file_size: Optional[float] = None
) -> None:
    """
    Loga informações de requisição.
    
    Args:
        method: Método HTTP
        path: Caminho da URL
        client_ip: IP do cliente
        request_id: ID da requisição
        file_size: Tamanho do arquivo (MB)
    """
    msg = f"[{request_id}] {method} {path} from {client_ip}"
    
    if file_size is not None:
        msg += f" | File: {file_size:.2f}MB"
    
    logger.info(msg)


def log_processing_result(
    request_id: str,
    success: bool,
    num_dates: int,
    processing_time: float
) -> None:
    """
    Loga resultado de processamento.
    
    Args:
        request_id: ID da requisição
        success: Sucesso?
        num_dates: Número de datas encontradas
        processing_time: Tempo de processamento
    """
    status = "✓" if success else "✗"
    msg = (
        f"[{request_id}] {status} "
        f"Dates: {num_dates} | "
        f"Time: {processing_time:.2f}s"
    )
    
    if success:
        logger.info(msg)
    else:
        logger.warning(msg)


# ========================================
# ERROR HANDLING
# ========================================

def create_error_response(
    code: str,
    message: str,
    details: Optional[dict] = None,
    request_id: Optional[str] = None
):
    """
    Cria response de erro padronizado.
    
    Args:
        code: Código do erro
        message: Mensagem
        details: Detalhes adicionais
        request_id: ID da requisição
        
    Returns:
        Dict com erro
    """
    from datetime import datetime
    
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details
        },
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id
    }


__all__ = [
    # Validação
    "validate_file_size",
    "validate_image_format",
    "validate_batch_size",
    
    # Conversão
    "decode_image",
    "encode_image",
    "resize_image",
    
    # Arquivos
    "get_file_size_mb",
    "sanitize_filename",
    "create_temp_file",
    
    # Formatação
    "format_bytes",
    "format_duration",
    
    # Hash
    "compute_file_hash",
    "compute_image_hash",
    
    # Logging
    "log_request_info",
    "log_processing_result",
    
    # Errors
    "create_error_response",
]
