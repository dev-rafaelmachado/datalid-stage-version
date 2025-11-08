"""
🔌 WebSocket para Processamento Assíncrono
Permite processamento em tempo real com feedback de progresso.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .service import get_service
from .utils import decode_image

# ========================================
# GERENCIADOR DE CONEXÕES
# ========================================

class ConnectionManager:
    """Gerencia conexões WebSocket ativas."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Aceita conexão WebSocket."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"🔌 Cliente conectado: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove conexão."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"🔌 Cliente desconectado: {client_id}")
    
    async def send_message(self, client_id: str, message: dict):
        """Envia mensagem para cliente específico."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Envia mensagem para todos os clientes."""
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar broadcast: {e}")
                disconnected.append(client_id)
        
        # Remover conexões desconectadas
        for client_id in disconnected:
            self.disconnect(client_id)


# Singleton
manager = ConnectionManager()


# ========================================
# HANDLERS DE WEBSOCKET
# ========================================

async def handle_websocket(websocket: WebSocket):
    """
    Handler principal do WebSocket.
    
    Protocolos de mensagem:
    
    Cliente -> Servidor:
    - {"type": "process", "image": "base64...", "options": {...}}
    - {"type": "ping"}
    
    Servidor -> Cliente:
    - {"type": "connected", "client_id": "..."}
    - {"type": "progress", "step": "...", "progress": 0.5}
    - {"type": "result", "data": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    """
    client_id = str(uuid.uuid4())
    
    await manager.connect(websocket, client_id)
    
    # Enviar confirmação de conexão
    await manager.send_message(client_id, {
        "type": "connected",
        "client_id": client_id,
        "message": "Conectado ao Datalid API"
    })
    
    try:
        while True:
            # Receber mensagem
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                # Processar comando
                if message_type == "ping":
                    await manager.send_message(client_id, {"type": "pong"})
                
                elif message_type == "process":
                    await handle_process_request(websocket, client_id, message)
                
                else:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": f"Tipo de mensagem desconhecido: {message_type}"
                    })
            
            except json.JSONDecodeError:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "Mensagem inválida (JSON)"
                })
            
            except Exception as e:
                logger.error(f"Erro ao processar mensagem: {e}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
        manager.disconnect(client_id)


async def handle_process_request(
    websocket: WebSocket,
    client_id: str,
    message: dict
):
    """
    Processa requisição de processamento de imagem.
    
    Args:
        websocket: Conexão WebSocket
        client_id: ID do cliente
        message: Mensagem com dados da requisição
    """
    try:
        # Enviar progresso: iniciando
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "initializing",
            "progress": 0.0,
            "message": "Iniciando processamento..."
        })
        
        # Extrair dados
        image_base64 = message.get("image")
        if not image_base64:
            raise ValueError("Campo 'image' (base64) é obrigatório")
        
        options = message.get("options", {})
        
        # Decodificar imagem
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "decoding",
            "progress": 0.1,
            "message": "Decodificando imagem..."
        })
        
        import base64
        image_bytes = base64.b64decode(image_base64)
        image = decode_image(image_bytes)
        
        # Processar com feedback de progresso
        service = get_service()
        
        # Detecção
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "detection",
            "progress": 0.3,
            "message": "Detectando regiões de data..."
        })
        
        await asyncio.sleep(0.1)  # Simular processamento
        
        # OCR
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "ocr",
            "progress": 0.6,
            "message": "Extraindo texto..."
        })
        
        await asyncio.sleep(0.1)
        
        # Parsing
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "parsing",
            "progress": 0.8,
            "message": "Analisando datas..."
        })
        
        # Processar imagem
        result = service.process_image(
            image=image,
            image_name=f"ws_{client_id}",
            request_id=client_id
        )
        
        # Enviar resultado
        await manager.send_message(client_id, {
            "type": "progress",
            "step": "completed",
            "progress": 1.0,
            "message": "Processamento concluído!"
        })
        
        await manager.send_message(client_id, {
            "type": "result",
            "data": result.model_dump(mode='json')
        })
    
    except Exception as e:
        logger.error(f"Erro ao processar via WebSocket: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e)
        })


__all__ = ["handle_websocket", "manager"]
