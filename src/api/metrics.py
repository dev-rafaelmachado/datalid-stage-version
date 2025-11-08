"""
📊 Sistema de Métricas e Monitoramento
Coleta e expõe métricas da API (Prometheus-compatible).
"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel

# ========================================
# MODELS
# ========================================

class MetricPoint(BaseModel):
    """Ponto de métrica."""
    timestamp: datetime
    value: float


class Metric(BaseModel):
    """Métrica agregada."""
    name: str
    description: str
    type: str  # counter, gauge, histogram
    value: float
    unit: str = ""


class SystemMetrics(BaseModel):
    """Métricas do sistema."""
    uptime_seconds: float
    total_requests: int
    total_errors: int
    active_connections: int
    total_images_processed: int
    total_dates_found: int
    average_processing_time: float
    requests_per_minute: float
    error_rate: float


# ========================================
# METRICS COLLECTOR
# ========================================

class MetricsCollector:
    """Coletor de métricas."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Contadores
        self.total_requests = 0
        self.total_errors = 0
        self.total_images_processed = 0
        self.total_dates_found = 0
        
        # Gauges
        self.active_connections = 0
        
        # Histogramas
        self.processing_times: List[float] = []
        self.request_times: List[float] = []
        
        # Request tracking (para rate)
        self.request_timestamps: List[float] = []
        self.max_timestamps = 1000
        
        # Métricas por endpoint
        self.endpoint_metrics: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "errors": 0, "total_time": 0.0}
        )
        
        logger.info("📊 MetricsCollector inicializado")
    
    def record_request(self, endpoint: str, duration: float, error: bool = False):
        """
        Registra uma requisição.
        
        Args:
            endpoint: Endpoint chamado
            duration: Duração em segundos
            error: Se houve erro
        """
        self.total_requests += 1
        self.request_times.append(duration)
        
        # Timestamp para rate
        current_time = time.time()
        self.request_timestamps.append(current_time)
        
        # Limpar timestamps antigos (> 1 minuto)
        cutoff_time = current_time - 60
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > cutoff_time
        ]
        
        # Métricas por endpoint
        self.endpoint_metrics[endpoint]["count"] += 1
        self.endpoint_metrics[endpoint]["total_time"] += duration
        
        if error:
            self.total_errors += 1
            self.endpoint_metrics[endpoint]["errors"] += 1
    
    def record_processing(self, duration: float, dates_found: int):
        """
        Registra um processamento de imagem.
        
        Args:
            duration: Duração em segundos
            dates_found: Número de datas encontradas
        """
        self.total_images_processed += 1
        self.total_dates_found += dates_found
        self.processing_times.append(duration)
        
        # Manter apenas últimas N medições
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def increment_connections(self):
        """Incrementa conexões ativas."""
        self.active_connections += 1
    
    def decrement_connections(self):
        """Decrementa conexões ativas."""
        self.active_connections = max(0, self.active_connections - 1)
    
    def get_metrics(self) -> SystemMetrics:
        """
        Obtém métricas do sistema.
        
        Returns:
            Métricas agregadas
        """
        uptime = time.time() - self.start_time
        
        # Calcular médias
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        # Requests por minuto (últimos 60 segundos)
        requests_per_minute = len(self.request_timestamps)
        
        # Taxa de erro
        error_rate = (
            (self.total_errors / self.total_requests * 100)
            if self.total_requests > 0 else 0.0
        )
        
        return SystemMetrics(
            uptime_seconds=uptime,
            total_requests=self.total_requests,
            total_errors=self.total_errors,
            active_connections=self.active_connections,
            total_images_processed=self.total_images_processed,
            total_dates_found=self.total_dates_found,
            average_processing_time=avg_processing_time,
            requests_per_minute=requests_per_minute,
            error_rate=error_rate
        )
    
    def get_prometheus_metrics(self) -> str:
        """
        Retorna métricas em formato Prometheus.
        
        Returns:
            String com métricas formatadas
        """
        metrics = self.get_metrics()
        
        lines = [
            "# HELP datalid_uptime_seconds Uptime da API em segundos",
            "# TYPE datalid_uptime_seconds gauge",
            f"datalid_uptime_seconds {metrics.uptime_seconds}",
            "",
            "# HELP datalid_requests_total Total de requisições",
            "# TYPE datalid_requests_total counter",
            f"datalid_requests_total {metrics.total_requests}",
            "",
            "# HELP datalid_errors_total Total de erros",
            "# TYPE datalid_errors_total counter",
            f"datalid_errors_total {metrics.total_errors}",
            "",
            "# HELP datalid_connections_active Conexões ativas",
            "# TYPE datalid_connections_active gauge",
            f"datalid_connections_active {metrics.active_connections}",
            "",
            "# HELP datalid_images_processed_total Total de imagens processadas",
            "# TYPE datalid_images_processed_total counter",
            f"datalid_images_processed_total {metrics.total_images_processed}",
            "",
            "# HELP datalid_dates_found_total Total de datas encontradas",
            "# TYPE datalid_dates_found_total counter",
            f"datalid_dates_found_total {metrics.total_dates_found}",
            "",
            "# HELP datalid_processing_time_seconds Tempo médio de processamento",
            "# TYPE datalid_processing_time_seconds gauge",
            f"datalid_processing_time_seconds {metrics.average_processing_time}",
            "",
            "# HELP datalid_requests_per_minute Requisições por minuto",
            "# TYPE datalid_requests_per_minute gauge",
            f"datalid_requests_per_minute {metrics.requests_per_minute}",
            "",
            "# HELP datalid_error_rate_percent Taxa de erro em porcentagem",
            "# TYPE datalid_error_rate_percent gauge",
            f"datalid_error_rate_percent {metrics.error_rate}",
        ]
        
        return "\n".join(lines)
    
    def get_endpoint_metrics(self) -> Dict[str, Dict]:
        """
        Retorna métricas por endpoint.
        
        Returns:
            Dict com métricas de cada endpoint
        """
        result = {}
        
        for endpoint, data in self.endpoint_metrics.items():
            count = data["count"]
            errors = data["errors"]
            total_time = data["total_time"]
            
            result[endpoint] = {
                "count": count,
                "errors": errors,
                "avg_time": total_time / count if count > 0 else 0.0,
                "error_rate": (errors / count * 100) if count > 0 else 0.0
            }
        
        return result


# ========================================
# SINGLETON
# ========================================

_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """
    Retorna instância singleton do coletor de métricas.
    
    Returns:
        MetricsCollector
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "SystemMetrics",
    "Metric",
    "MetricPoint"
]
