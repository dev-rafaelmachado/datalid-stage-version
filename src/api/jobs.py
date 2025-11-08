"""
⚙️ Sistema de Jobs/Tarefas
Processamento assíncrono em background com acompanhamento de status.
"""

import asyncio
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import BaseModel

from .schemas import ProcessImageResponse
from .service import get_service

# ========================================
# ENUMS
# ========================================

class JobStatus(str, Enum):
    """Status de um job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ========================================
# MODELS
# ========================================

class Job(BaseModel):
    """Representa um job de processamento."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[ProcessImageResponse] = None
    error: Optional[str] = None
    
    class Config:
        use_enum_values = True


# ========================================
# JOB MANAGER
# ========================================

class JobManager:
    """Gerencia jobs de processamento em background."""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.max_jobs = 1000  # Limitar número de jobs armazenados
        logger.info("⚙️ JobManager inicializado")
    
    def create_job(self) -> str:
        """
        Cria um novo job.
        
        Returns:
            ID do job
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            message="Job criado"
        )
        
        self.jobs[job_id] = job
        logger.info(f"📋 Job criado: {job_id}")
        
        # Limpar jobs antigos se necessário
        self._cleanup_old_jobs()
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Obtém informações de um job.
        
        Args:
            job_id: ID do job
            
        Returns:
            Job ou None se não encontrado
        """
        return self.jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[ProcessImageResponse] = None,
        error: Optional[str] = None
    ):
        """
        Atualiza informações de um job.
        
        Args:
            job_id: ID do job
            status: Novo status
            progress: Progresso (0.0 - 1.0)
            message: Mensagem de status
            result: Resultado do processamento
            error: Mensagem de erro
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job não encontrado: {job_id}")
            return
        
        # Atualizar campos
        if status is not None:
            job.status = status
            
            if status == JobStatus.PROCESSING and job.started_at is None:
                job.started_at = datetime.now()
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.now()
        
        if progress is not None:
            job.progress = progress
        
        if message is not None:
            job.message = message
        
        if result is not None:
            job.result = result
        
        if error is not None:
            job.error = error
        
        logger.debug(f"Job {job_id} atualizado: {status} - {message}")
    
    def cancel_job(self, job_id: str):
        """
        Cancela um job.
        
        Args:
            job_id: ID do job
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job não encontrado: {job_id}")
            return
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            logger.warning(f"Não é possível cancelar job já finalizado: {job_id}")
            return
        
        self.update_job(
            job_id,
            status=JobStatus.CANCELLED,
            message="Job cancelado pelo usuário"
        )
        
        logger.info(f"Job cancelado: {job_id}")
    
    def delete_job(self, job_id: str):
        """
        Remove um job.
        
        Args:
            job_id: ID do job
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Job removido: {job_id}")
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> list[Job]:
        """
        Lista jobs.
        
        Args:
            status: Filtrar por status (opcional)
            
        Returns:
            Lista de jobs
        """
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Ordenar por data de criação (mais recente primeiro)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs
    
    async def process_job_async(
        self,
        job_id: str,
        image_data: bytes,
        image_name: str = "job_image",
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Processa um job de forma assíncrona.
        
        Args:
            job_id: ID do job
            image_data: Dados da imagem
            image_name: Nome da imagem
            options: Opções de processamento
        """
        try:
            # Atualizar status
            self.update_job(
                job_id,
                status=JobStatus.PROCESSING,
                progress=0.0,
                message="Iniciando processamento..."
            )
            
            # Decodificar imagem
            from .utils import decode_image
            image = decode_image(image_data)
            
            self.update_job(job_id, progress=0.2, message="Imagem decodificada")
            
            # Processar
            service = get_service()
            
            self.update_job(job_id, progress=0.3, message="Detectando regiões...")
            
            result = await asyncio.to_thread(
                service.process_image,
                image=image,
                image_name=image_name,
                request_id=job_id
            )
            
            self.update_job(job_id, progress=0.9, message="Finalizando...")
            
            # Job concluído
            self.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message="Processamento concluído com sucesso",
                result=result
            )
            
            logger.success(f"✅ Job {job_id} concluído")
        
        except Exception as e:
            logger.error(f"❌ Erro no job {job_id}: {e}")
            
            self.update_job(
                job_id,
                status=JobStatus.FAILED,
                message="Erro no processamento",
                error=str(e)
            )
    
    def _cleanup_old_jobs(self):
        """Remove jobs antigos se exceder o limite."""
        if len(self.jobs) > self.max_jobs:
            # Remover jobs mais antigos e finalizados
            completed_jobs = [
                (job_id, job) for job_id, job in self.jobs.items()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            ]
            
            # Ordenar por data de conclusão
            completed_jobs.sort(key=lambda x: x[1].completed_at or datetime.min)
            
            # Remover os mais antigos
            num_to_remove = len(self.jobs) - self.max_jobs
            for job_id, _ in completed_jobs[:num_to_remove]:
                del self.jobs[job_id]
            
            logger.info(f"🧹 {num_to_remove} jobs antigos removidos")


# ========================================
# SINGLETON
# ========================================

_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """
    Retorna instância singleton do JobManager.
    
    Returns:
        JobManager
    """
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


__all__ = ["Job", "JobStatus", "JobManager", "get_job_manager"]
