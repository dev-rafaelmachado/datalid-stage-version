"""
🔌 Cliente Python para a API Datalid
Facilita a integração com a API de detecção de datas de validade.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger


class DatalidClient:
    """
    Cliente Python para a API Datalid.
    
    Exemplos:
        >>> client = DatalidClient("http://localhost:8000")
        >>> result = client.process_image("produto.jpg")
        >>> print(result['best_date']['date'])
        '2025-12-31'
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Inicializa o cliente.
        
        Args:
            base_url: URL base da API
            api_key: Chave de API (se autenticação estiver ativada)
            timeout: Timeout para requisições (segundos)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Headers
        self.headers = {}
        if api_key:
            self.headers['X-API-Key'] = api_key
        
        logger.info(f"🔌 Cliente Datalid criado: {base_url}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Faz requisição HTTP."""
        url = f"{self.base_url}{endpoint}"
        
        # Adicionar headers
        if 'headers' in kwargs:
            kwargs['headers'].update(self.headers)
        else:
            kwargs['headers'] = self.headers.copy()
        
        # Timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        
        # Fazer requisição
        response = requests.request(method, url, **kwargs)
        
        # Log
        logger.debug(f"{method} {url} → {response.status_code}")
        
        return response
    
    # ========================================
    # HEALTH & INFO
    # ========================================
    
    def health(self) -> Dict[str, Any]:
        """
        Verifica saúde da API.
        
        Returns:
            Status de saúde
            
        Exemplo:
            >>> health = client.health()
            >>> print(health['status'])
            'healthy'
        """
        response = self._request('GET', '/health')
        response.raise_for_status()
        return response.json()
    
    def info(self) -> Dict[str, Any]:
        """
        Obtém informações da API.
        
        Returns:
            Informações detalhadas
            
        Exemplo:
            >>> info = client.info()
            >>> print(info['version'])
            '1.0.0'
        """
        response = self._request('GET', '/info')
        response.raise_for_status()
        return response.json()
    
    def is_ready(self) -> bool:
        """
        Verifica se API está pronta.
        
        Returns:
            True se pronta
        """
        try:
            health = self.health()
            return health['status'] == 'healthy'
        except Exception:
            return False
    
    # ========================================
    # PROCESSAMENTO
    # ========================================
    
    def process_image(
        self,
        image_path: Union[str, Path],
        detection_confidence: Optional[float] = None,
        detection_iou: Optional[float] = None,
        ocr_engine: Optional[str] = None,
        return_visualization: bool = False,
        return_crops: bool = False,
        return_full_ocr: bool = False,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Processa uma imagem.
        
        Args:
            image_path: Caminho da imagem
            detection_confidence: Confiança mínima (0.0-1.0)
            detection_iou: IoU threshold (0.0-1.0)
            ocr_engine: Engine de OCR a usar
            return_visualization: Retornar imagem com visualizações
            return_crops: Retornar crops das detecções
            return_full_ocr: Retornar todos os resultados OCR
            save_results: Salvar resultados no servidor
            
        Returns:
            Resultado do processamento
            
        Exemplo:
            >>> result = client.process_image(
            ...     "produto.jpg",
            ...     detection_confidence=0.3,
            ...     ocr_engine="openocr"
            ... )
            >>> print(result['best_date']['date'])
            '2025-12-31'
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        
        # Preparar dados
        files = {'file': open(image_path, 'rb')}
        
        data = {}
        if detection_confidence is not None:
            data['detection_confidence'] = detection_confidence
        if detection_iou is not None:
            data['detection_iou'] = detection_iou
        if ocr_engine is not None:
            data['ocr_engine'] = ocr_engine
        
        data['return_visualization'] = return_visualization
        data['return_crops'] = return_crops
        data['return_full_ocr'] = return_full_ocr
        data['save_results'] = save_results
        
        try:
            response = self._request(
                'POST',
                '/process',
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        finally:
            files['file'].close()
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        detection_confidence: Optional[float] = None,
        detection_iou: Optional[float] = None,
        ocr_engine: Optional[str] = None,
        return_visualization: bool = False,
        return_crops: bool = False,
        return_full_ocr: bool = False,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Processa múltiplas imagens.
        
        Args:
            image_paths: Lista de caminhos de imagens
            detection_confidence: Confiança mínima
            detection_iou: IoU threshold
            ocr_engine: Engine de OCR
            return_visualization: Retornar visualizações
            return_crops: Retornar crops
            return_full_ocr: Retornar todos OCR
            save_results: Salvar resultados
            
        Returns:
            Resultados do batch
            
        Exemplo:
            >>> results = client.process_batch([
            ...     "produto1.jpg",
            ...     "produto2.jpg",
            ...     "produto3.jpg"
            ... ])
            >>> print(f"Processadas: {results['successful']}/{results['total_images']}")
        """
        # Validar arquivos
        valid_paths = []
        for path in image_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Imagem não encontrada (ignorando): {path}")
                continue
            valid_paths.append(path)
        
        if not valid_paths:
            raise ValueError("Nenhuma imagem válida fornecida")
        
        # Preparar dados
        files = [('files', open(path, 'rb')) for path in valid_paths]
        
        data = {}
        if detection_confidence is not None:
            data['detection_confidence'] = detection_confidence
        if detection_iou is not None:
            data['detection_iou'] = detection_iou
        if ocr_engine is not None:
            data['ocr_engine'] = ocr_engine
        
        data['return_visualization'] = return_visualization
        data['return_crops'] = return_crops
        data['return_full_ocr'] = return_full_ocr
        data['save_results'] = save_results
        
        try:
            response = self._request(
                'POST',
                '/process/batch',
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        finally:
            for _, f in files:
                f.close()
    
    def process_url(
        self,
        url: str,
        detection_confidence: Optional[float] = None,
        detection_iou: Optional[float] = None,
        ocr_engine: Optional[str] = None,
        return_visualization: bool = False,
        return_crops: bool = False,
        return_full_ocr: bool = False,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Processa imagem de uma URL.
        
        Args:
            url: URL da imagem
            detection_confidence: Confiança mínima
            detection_iou: IoU threshold
            ocr_engine: Engine de OCR
            return_visualization: Retornar visualizações
            return_crops: Retornar crops
            return_full_ocr: Retornar todos OCR
            save_results: Salvar resultados
            
        Returns:
            Resultado do processamento
            
        Exemplo:
            >>> result = client.process_url("https://example.com/produto.jpg")
            >>> print(result['best_date']['date'])
        """
        data = {'url': url}
        
        if detection_confidence is not None:
            data['detection_confidence'] = detection_confidence
        if detection_iou is not None:
            data['detection_iou'] = detection_iou
        if ocr_engine is not None:
            data['ocr_engine'] = ocr_engine
        
        data['return_visualization'] = return_visualization
        data['return_crops'] = return_crops
        data['return_full_ocr'] = return_full_ocr
        data['save_results'] = save_results
        
        response = self._request(
            'POST',
            '/process/url',
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================
    # JOBS (API v2 - Processamento Assíncrono)
    # ========================================
    
    def create_job(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Cria um job de processamento assíncrono.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Info do job (job_id, status_url)
            
        Exemplo:
            >>> job = client.create_job("produto.jpg")
            >>> print(job['job_id'])
            'abc123...'
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        
        files = {'file': open(image_path, 'rb')}
        
        try:
            response = self._request('POST', '/v2/jobs', files=files)
            response.raise_for_status()
            return response.json()
        finally:
            files['file'].close()
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Consulta status de um job.
        
        Args:
            job_id: ID do job
            
        Returns:
            Informações do job
            
        Exemplo:
            >>> job = client.get_job("abc123...")
            >>> print(job['status'])
            'processing'
        """
        response = self._request('GET', f'/v2/jobs/{job_id}', timeout=10)
        response.raise_for_status()
        return response.json()
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        max_wait: float = 300.0
    ) -> Dict[str, Any]:
        """
        Aguarda conclusão de um job.
        
        Args:
            job_id: ID do job
            poll_interval: Intervalo entre consultas (segundos)
            max_wait: Tempo máximo de espera (segundos)
            
        Returns:
            Resultado do job
            
        Raises:
            TimeoutError: Se exceder max_wait
            
        Exemplo:
            >>> job = client.create_job("produto.jpg")
            >>> result = client.wait_for_job(job['job_id'])
            >>> print(result['best_date']['date'])
        """
        import time
        start_time = time.time()
        
        while True:
            job = self.get_job(job_id)
            status = job['status']
            
            if status == 'completed':
                return job['result']
            
            elif status == 'failed':
                error = job.get('error', 'Unknown error')
                raise Exception(f"Job falhou: {error}")
            
            elif status == 'cancelled':
                raise Exception("Job foi cancelado")
            
            # Verificar timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job não completou em {max_wait}s")
            
            # Aguardar
            time.sleep(poll_interval)
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Lista jobs.
        
        Args:
            status: Filtrar por status (opcional)
            limit: Número máximo de jobs
            
        Returns:
            Lista de jobs
            
        Exemplo:
            >>> jobs = client.list_jobs(status='completed', limit=10)
            >>> print(f"Jobs completados: {len(jobs)}")
        """
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        response = self._request('GET', '/v2/jobs', params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> None:
        """
        Cancela um job.
        
        Args:
            job_id: ID do job
            
        Exemplo:
            >>> client.cancel_job("abc123...")
        """
        response = self._request('DELETE', f'/v2/jobs/{job_id}', timeout=10)
        response.raise_for_status()
    
    def process_image_async(
        self,
        image_path: Union[str, Path],
        wait: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processa imagem de forma assíncrona.
        
        Args:
            image_path: Caminho da imagem
            wait: Aguardar conclusão do job
            **kwargs: Argumentos para wait_for_job
            
        Returns:
            Resultado do processamento (se wait=True) ou info do job
            
        Exemplo:
            >>> # Processar e aguardar
            >>> result = client.process_image_async("produto.jpg")
            >>> 
            >>> # Apenas criar job
            >>> job = client.process_image_async("produto.jpg", wait=False)
            >>> # ... fazer outras coisas ...
            >>> result = client.wait_for_job(job['job_id'])
        """
        job = self.create_job(image_path)
        
        if wait:
            return self.wait_for_job(job['job_id'], **kwargs)
        else:
            return job
    
    # ========================================
    # MÉTRICAS (API v2)
    # ========================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas do sistema.
        
        Returns:
            Métricas agregadas
            
        Exemplo:
            >>> metrics = client.get_metrics()
            >>> print(f"Uptime: {metrics['uptime_seconds']}s")
            >>> print(f"Total de imagens: {metrics['total_images_processed']}")
            >>> print(f"Datas encontradas: {metrics['total_dates_found']}")
        """
        response = self._request('GET', '/v2/metrics', timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas detalhadas por endpoint.
        
        Returns:
            Métricas por endpoint
            
        Exemplo:
            >>> metrics = client.get_endpoint_metrics()
            >>> for endpoint, data in metrics.items():
            ...     print(f"{endpoint}: {data['count']} calls, {data['avg_time']:.2f}s avg")
        """
        response = self._request('GET', '/v2/metrics/endpoints', timeout=10)
        response.raise_for_status()
        return response.json()
    
    # ========================================
    # HELPERS
    # ========================================
    
    def get_expiry_date(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Optional[datetime]:
        """
        Obtém data de validade de uma imagem.
        
        Args:
            image_path: Caminho da imagem
            **kwargs: Argumentos para process_image
            
        Returns:
            Data de validade ou None
            
        Exemplo:
            >>> date = client.get_expiry_date("produto.jpg")
            >>> if date:
            ...     print(f"Validade: {date.strftime('%d/%m/%Y')}")
        """
        result = self.process_image(image_path, **kwargs)
        
        if result['best_date'] and result['best_date']['date']:
            date_str = result['best_date']['date']
            return datetime.strptime(date_str, '%Y-%m-%d')
        
        return None
    
    def is_expired(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Optional[bool]:
        """
        Verifica se produto está expirado.
        
        Args:
            image_path: Caminho da imagem
            **kwargs: Argumentos para process_image
            
        Returns:
            True se expirado, False se válido, None se não encontrou data
            
        Exemplo:
            >>> if client.is_expired("produto.jpg"):
            ...     print("⚠️ Produto expirado!")
        """
        result = self.process_image(image_path, **kwargs)
        
        if result['best_date']:
            return result['best_date'].get('is_expired')
        
        return None
    
    def days_until_expiry(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Optional[int]:
        """
        Retorna dias até expiração.
        
        Args:
            image_path: Caminho da imagem
            **kwargs: Argumentos para process_image
            
        Returns:
            Número de dias (negativo se expirado) ou None
            
        Exemplo:
            >>> days = client.days_until_expiry("produto.jpg")
            >>> if days:
            ...     print(f"Expira em {days} dias")
        """
        result = self.process_image(image_path, **kwargs)
        
        if result['best_date']:
            return result['best_date'].get('days_until_expiry')
        
        return None
    
    # ========================================
    # CONTEXT MANAGER
    # ========================================
    
    def __enter__(self):
        """Suporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do context."""
        pass
    
    def __repr__(self) -> str:
        return f"DatalidClient(base_url='{self.base_url}')"


# ========================================
# EXEMPLO DE USO
# ========================================

if __name__ == "__main__":
    from rich.console import Console
    
    console = Console()
    
    # Criar cliente
    client = DatalidClient("http://localhost:8000")
    
    # Verificar se API está pronta
    if not client.is_ready():
        console.print("❌ API não está pronta!", style="bold red")
        exit(1)
    
    console.print("✅ API pronta!", style="bold green")
    
    # Informações
    info = client.info()
    console.print(f"\n📚 {info['name']} v{info['version']}", style="bold cyan")
    console.print(f"Engines OCR: {', '.join(info['ocr_engines'])}")
    
    # Processar imagem de exemplo
    test_image = Path("data/sample.jpg")
    
    if test_image.exists():
        console.print(f"\n📷 Processando: {test_image}", style="bold")
        
        result = client.process_image(
            test_image,
            return_full_ocr=True
        )
        
        console.print(f"\n✓ Status: {result['status']}", style="green")
        console.print(f"✓ Detecções: {len(result['detections'])}")
        
        if result['best_date']:
            date = result['best_date']
            console.print(f"\n📅 Data encontrada: {date['date']}", style="bold green")
            console.print(f"   Confiança: {date['confidence']:.1%}")
            console.print(f"   Dias até expirar: {date['days_until_expiry']}")
    else:
        console.print(f"\n⚠️ Imagem não encontrada: {test_image}", style="yellow")
