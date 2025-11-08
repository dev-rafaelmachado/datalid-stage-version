"""
🛣️ Rotas Adicionais da API
Endpoints para jobs, métricas, WebSocket, etc.
"""

from typing import List, Optional

from fastapi import (APIRouter, BackgroundTasks, File, HTTPException,
                     UploadFile, WebSocket, status)
from fastapi.responses import PlainTextResponse
from loguru import logger

from .jobs import Job, JobStatus, get_job_manager
from .metrics import get_metrics_collector
from .schemas import ProcessImageResponse
from .websocket import handle_websocket

# ========================================
# ROUTER
# ========================================

router_v2 = APIRouter(prefix="/v2", tags=["API v2"])


# ========================================
# WEBSOCKET
# ========================================

@router_v2.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket para processamento em tempo real.
    
    **Protocolo:**
    
    Cliente -> Servidor:
    ```json
    {
        "type": "process",
        "image": "base64_encoded_image",
        "options": {
            "return_visualization": false,
            "return_crops": false
        }
    }
    ```
    
    Servidor -> Cliente:
    ```json
    {
        "type": "progress",
        "step": "detection",
        "progress": 0.5,
        "message": "Detectando regiões..."
    }
    ```
    
    ```json
    {
        "type": "result",
        "data": { ... }
    }
    ```
    """
    await handle_websocket(websocket)


# ========================================
# JOBS (Processamento Assíncrono)
# ========================================

@router_v2.post(
    "/jobs",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Criar job de processamento",
    description="Cria um job para processar imagem de forma assíncrona"
)
async def create_processing_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Imagem para processar")
):
    """
    Cria um job para processar uma imagem de forma assíncrona.
    
    **Retorna:**
    - ID do job
    - URL para consultar status
    
    **Fluxo:**
    1. Criar job
    2. Iniciar processamento em background
    3. Retornar ID do job imediatamente
    4. Cliente consulta status via GET /jobs/{job_id}
    """
    job_manager = get_job_manager()
    
    # Criar job
    job_id = job_manager.create_job()
    
    # Ler dados da imagem
    image_data = await file.read()
    
    # Iniciar processamento em background
    background_tasks.add_task(
        job_manager.process_job_async,
        job_id=job_id,
        image_data=image_data,
        image_name=file.filename
    )
    
    return {
        "job_id": job_id,
        "status": "accepted",
        "message": "Job criado. Use GET /v2/jobs/{job_id} para consultar status",
        "status_url": f"/v2/jobs/{job_id}"
    }


@router_v2.get(
    "/jobs/{job_id}",
    response_model=Job,
    summary="Consultar status de job",
    description="Obtém informações sobre um job de processamento"
)
async def get_job_status(job_id: str):
    """
    Consulta o status de um job.
    
    **Status possíveis:**
    - `pending`: Job criado mas ainda não iniciado
    - `processing`: Job em processamento
    - `completed`: Job concluído com sucesso
    - `failed`: Job falhou
    - `cancelled`: Job cancelado
    """
    job_manager = get_job_manager()
    
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job não encontrado: {job_id}"
        )
    
    return job


@router_v2.get(
    "/jobs",
    response_model=List[Job],
    summary="Listar jobs",
    description="Lista todos os jobs ou filtra por status"
)
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 100
):
    """
    Lista jobs.
    
    **Parâmetros:**
    - `status`: Filtrar por status (opcional)
    - `limit`: Número máximo de jobs a retornar
    """
    job_manager = get_job_manager()
    
    jobs = job_manager.list_jobs(status=status)
    
    # Limitar resultados
    jobs = jobs[:limit]
    
    return jobs


@router_v2.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancelar job",
    description="Cancela um job em execução ou remove um job finalizado"
)
async def cancel_job(job_id: str):
    """
    Cancela um job.
    
    **Nota:** Jobs já finalizados (completed/failed) serão apenas removidos.
    """
    job_manager = get_job_manager()
    
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job não encontrado: {job_id}"
        )
    
    # Cancelar ou remover
    if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        job_manager.cancel_job(job_id)
    else:
        job_manager.delete_job(job_id)
    
    return None


# ========================================
# MÉTRICAS E MONITORAMENTO
# ========================================

@router_v2.get(
    "/metrics",
    summary="Métricas do sistema",
    description="Retorna métricas agregadas da API"
)
async def get_metrics():
    """
    Retorna métricas do sistema.
    
    **Métricas incluídas:**
    - Uptime
    - Total de requisições
    - Total de erros
    - Conexões ativas
    - Imagens processadas
    - Datas encontradas
    - Tempo médio de processamento
    - Requisições por minuto
    - Taxa de erro
    """
    collector = get_metrics_collector()
    metrics = collector.get_metrics()
    
    return metrics


@router_v2.get(
    "/metrics/endpoints",
    summary="Métricas por endpoint",
    description="Retorna métricas detalhadas por endpoint"
)
async def get_endpoint_metrics():
    """
    Retorna métricas detalhadas por endpoint.
    
    Para cada endpoint mostra:
    - Número de chamadas
    - Número de erros
    - Tempo médio de resposta
    - Taxa de erro
    """
    collector = get_metrics_collector()
    metrics = collector.get_endpoint_metrics()
    
    return metrics


@router_v2.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    summary="Métricas Prometheus",
    description="Expõe métricas em formato Prometheus"
)
async def get_prometheus_metrics():
    """
    Expõe métricas em formato Prometheus.
    
    Compatível com Prometheus scraper.
    """
    collector = get_metrics_collector()
    metrics_text = collector.get_prometheus_metrics()
    
    return metrics_text


# ========================================
# ADMIN
# ========================================

@router_v2.post(
    "/admin/reload",
    summary="Recarregar modelos",
    description="Recarrega os modelos (requer autenticação)"
)
async def reload_models():
    """
    Recarrega os modelos de ML.
    
    **Nota:** Este endpoint pode causar downtime temporário.
    """
    from .service import get_service
    
    try:
        service = get_service()
        service.initialize()
        
        return {
            "status": "success",
            "message": "Modelos recarregados com sucesso"
        }
    
    except Exception as e:
        logger.error(f"Erro ao recarregar modelos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao recarregar modelos: {str(e)}"
        )


@router_v2.post(
    "/admin/clear-cache",
    summary="Limpar cache",
    description="Limpa cache de jobs e resultados"
)
async def clear_cache():
    """
    Limpa cache de jobs e resultados temporários.
    """
    job_manager = get_job_manager()
    
    # Remover jobs finalizados
    completed_jobs = [
        j.job_id for j in job_manager.list_jobs()
        if j.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    ]
    
    for job_id in completed_jobs:
        job_manager.delete_job(job_id)
    
    return {
        "status": "success",
        "message": f"{len(completed_jobs)} jobs removidos",
        "cleared": len(completed_jobs)
    }


__all__ = ["router_v2"]
