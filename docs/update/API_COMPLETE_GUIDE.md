# 🚀 Datalid API - Guia Completo

API REST para detecção e extração de datas de validade de produtos usando Deep Learning e OCR.

## 📋 Índice

- [Instalação](#instalação)
- [Quick Start](#quick-start)
- [Endpoints](#endpoints)
- [Cliente Python](#cliente-python)
- [WebSocket](#websocket)
- [Jobs Assíncronos](#jobs-assíncronos)
- [Autenticação](#autenticação)
- [Configuração](#configuração)
- [Docker](#docker)
- [Exemplos](#exemplos)

## 🔧 Instalação

### Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

### Instalação Rápida

```bash
# Clonar repositório
git clone <repo>
cd datalid3.0

# Instalar dependências
pip install -r requirements.txt

# Iniciar servidor
python -m src.api.main
```

Servidor estará disponível em: http://localhost:8000

Documentação interativa: http://localhost:8000/docs

## ⚡ Quick Start

### Via cURL

```bash
# Processar uma imagem
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg" \
  -F "return_visualization=false"
```

### Via Python

```python
from scripts.api.client import DatalidClient

# Criar cliente
client = DatalidClient("http://localhost:8000")

# Processar imagem
result = client.process_image("produto.jpg")

# Acessar resultados
if result['best_date']:
    print(f"Data de validade: {result['best_date']['date']}")
    print(f"Dias até expirar: {result['best_date']['days_until_expiry']}")
```

## 📡 Endpoints

### API v1

#### `GET /health`
Verifica status de saúde da API.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "processing_service": {"status": "healthy", "message": "Pronto"},
    "pipeline": {"status": "healthy", "message": "Carregado"}
  }
}
```

#### `GET /info`
Informações detalhadas sobre a API.

**Response:**
```json
{
  "name": "Datalid API",
  "version": "1.0.0",
  "models": {...},
  "ocr_engines": ["tesseract", "openocr", "parseq", ...],
  "limits": {
    "max_file_size_mb": 10,
    "max_batch_size": 50
  }
}
```

#### `POST /process`
Processa uma única imagem.

**Parameters:**
- `file` (required): Arquivo de imagem
- `detection_confidence` (optional): Confiança mínima (0.0-1.0)
- `detection_iou` (optional): IoU threshold (0.0-1.0)
- `ocr_engine` (optional): Engine de OCR
- `return_visualization` (optional): Retornar imagem com visualizações
- `return_crops` (optional): Retornar crops das detecções
- `return_full_ocr` (optional): Retornar todos os resultados OCR
- `save_results` (optional): Salvar resultados no servidor

**Response:**
```json
{
  "status": "success",
  "message": "1 data(s) de validade encontrada(s)",
  "detections": [
    {
      "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 100},
      "confidence": 0.95,
      "class_name": "exp_date"
    }
  ],
  "dates": [
    {
      "date": "2025-12-31",
      "confidence": 0.89,
      "format": "DD/MM/YYYY",
      "is_expired": false,
      "days_until_expiry": 421
    }
  ],
  "best_date": {
    "date": "2025-12-31",
    "confidence": 0.89
  },
  "metrics": {
    "total_time": 1.234,
    "detection_time": 0.456,
    "ocr_time": 0.678,
    "num_detections": 1,
    "num_dates_found": 1
  }
}
```

#### `POST /process/batch`
Processa múltiplas imagens.

**Parameters:**
- `files` (required): Lista de arquivos de imagem
- Outros parâmetros iguais a `/process`

**Response:**
```json
{
  "status": "success",
  "total_images": 5,
  "successful": 5,
  "failed": 0,
  "results": [
    {
      "filename": "produto1.jpg",
      "success": true,
      "result": {...}
    },
    ...
  ],
  "total_time": 5.678
}
```

#### `POST /process/url`
Processa imagem de uma URL.

**Parameters:**
- `url` (required): URL da imagem
- Outros parâmetros iguais a `/process`

### API v2 (Recursos Avançados)

#### `WS /v2/ws`
WebSocket para processamento em tempo real com feedback de progresso.

#### `POST /v2/jobs`
Cria job de processamento assíncrono.

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "accepted",
  "message": "Job criado",
  "status_url": "/v2/jobs/abc123..."
}
```

#### `GET /v2/jobs/{job_id}`
Consulta status de um job.

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "completed",
  "progress": 1.0,
  "message": "Processamento concluído",
  "result": {...}
}
```

#### `GET /v2/jobs`
Lista jobs.

**Parameters:**
- `status` (optional): Filtrar por status
- `limit` (optional): Número máximo de jobs

#### `DELETE /v2/jobs/{job_id}`
Cancela/remove um job.

#### `GET /v2/metrics`
Métricas do sistema.

**Response:**
```json
{
  "uptime_seconds": 3600,
  "total_requests": 1234,
  "total_errors": 5,
  "total_images_processed": 500,
  "total_dates_found": 450,
  "average_processing_time": 1.23,
  "requests_per_minute": 20.5,
  "error_rate": 0.4
}
```

#### `GET /v2/metrics/prometheus`
Métricas em formato Prometheus.

## 🐍 Cliente Python

### Instalação

O cliente está incluído no pacote:

```python
from scripts.api.client import DatalidClient
```

### Exemplos

```python
# Criar cliente
client = DatalidClient("http://localhost:8000", api_key="sua-api-key")

# Verificar saúde
if client.is_ready():
    print("✅ API pronta!")

# Processar imagem
result = client.process_image(
    "produto.jpg",
    detection_confidence=0.3,
    ocr_engine="openocr"
)

# Processar batch
results = client.process_batch([
    "produto1.jpg",
    "produto2.jpg",
    "produto3.jpg"
])

# Processar de URL
result = client.process_url("https://example.com/produto.jpg")

# Helpers
date = client.get_expiry_date("produto.jpg")
is_expired = client.is_expired("produto.jpg")
days = client.days_until_expiry("produto.jpg")

# Jobs assíncronos
job = client.create_job("produto.jpg")
result = client.wait_for_job(job['job_id'])

# Ou de forma mais simples
result = client.process_image_async("produto.jpg")

# Métricas
metrics = client.get_metrics()
print(f"Imagens processadas: {metrics['total_images_processed']}")
```

## 🔌 WebSocket

### JavaScript

```javascript
const ws = new WebSocket('ws://localhost:8000/v2/ws');

ws.onopen = () => {
  // Enviar imagem
  const imageBase64 = '...';
  ws.send(JSON.stringify({
    type: 'process',
    image: imageBase64,
    options: {
      return_visualization: false
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`Progresso: ${data.progress * 100}%`);
    console.log(`Etapa: ${data.step}`);
  }
  
  if (data.type === 'result') {
    console.log('Resultado:', data.data);
  }
};
```

### Python

```python
import asyncio
import websockets
import json
import base64

async def process_via_websocket():
    uri = "ws://localhost:8000/v2/ws"
    
    async with websockets.connect(uri) as websocket:
        # Enviar imagem
        with open("produto.jpg", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        await websocket.send(json.dumps({
            "type": "process",
            "image": image_base64
        }))
        
        # Receber mensagens
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'progress':
                print(f"Progresso: {data['progress']*100:.0f}%")
            
            if data['type'] == 'result':
                print("Resultado:", data['data'])
                break

asyncio.run(process_via_websocket())
```

## ⏱️ Jobs Assíncronos

Para processar imagens sem bloquear a requisição:

```python
# Criar job
response = requests.post(
    "http://localhost:8000/v2/jobs",
    files={"file": open("produto.jpg", "rb")}
)
job_id = response.json()['job_id']

# Consultar status
while True:
    response = requests.get(f"http://localhost:8000/v2/jobs/{job_id}")
    job = response.json()
    
    if job['status'] == 'completed':
        result = job['result']
        break
    
    time.sleep(1)
```

## 🔐 Autenticação

### API Key

Configurar no `.env`:

```env
AUTH_ENABLED=true
API_KEYS=["key1","key2","key3"]
```

Usar nos requests:

```bash
curl -H "X-API-Key: key1" http://localhost:8000/process ...
```

```python
client = DatalidClient("http://localhost:8000", api_key="key1")
```

### JWT (Avançado)

```python
import requests

# Login (implementar endpoint de login)
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "user",
    "password": "pass"
})
token = response.json()['access_token']

# Usar token
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/process",
    headers=headers,
    files={"file": open("produto.jpg", "rb")}
)
```

## ⚙️ Configuração

### Variáveis de Ambiente

Criar arquivo `.env`:

```env
# Servidor
API_NAME=Datalid API
API_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=false

# CORS
CORS_ENABLED=true
CORS_ORIGINS=["*"]

# Limites
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=50
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60

# Autenticação
AUTH_ENABLED=false
API_KEYS=[]

# Modelos
DEFAULT_YOLO_MODEL=experiments/yolov8m_seg_best/weights/best.pt
MODEL_DEVICE=0
DEFAULT_OCR_ENGINE=openocr
DEFAULT_CONFIDENCE=0.25

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Debug
DEBUG=false
ENVIRONMENT=production
```

### Configuração Programática

```python
from src.api.config import APISettings

settings = APISettings(
    host="0.0.0.0",
    port=8080,
    max_file_size_mb=20,
    default_ocr_engine="parseq"
)
```

## 🐳 Docker

### Build

```bash
docker build -t datalid-api .
```

### Run

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  datalid-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./outputs:/app/outputs
    environment:
      - LOG_LEVEL=INFO
      - DEFAULT_OCR_ENGINE=openocr
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📊 Monitoramento

### Prometheus

Scrape configuration:

```yaml
scrape_configs:
  - job_name: 'datalid-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/v2/metrics/prometheus'
```

### Grafana

Importar dashboard com métricas:
- Uptime
- Requisições por minuto
- Taxa de erro
- Tempo médio de processamento
- Imagens processadas
- Datas encontradas

## 🔧 Desenvolvimento

### Rodar em modo debug

```bash
python -m src.api.main --reload --debug
```

### Testes

```bash
pytest tests/api/
```

### Formato de código

```bash
black src/api/
isort src/api/
flake8 src/api/
```

## 📚 Recursos Adicionais

- [Documentação Swagger](http://localhost:8000/docs)
- [Documentação ReDoc](http://localhost:8000/redoc)
- [OpenAPI Schema](http://localhost:8000/openapi.json)

## 🆘 Troubleshooting

### API não inicia

- Verificar se porta 8000 está livre
- Verificar logs em `logs/api.log`
- Verificar se modelos estão disponíveis

### Erro ao processar imagem

- Verificar formato da imagem (JPEG, PNG, BMP)
- Verificar tamanho da imagem (máx 10MB por padrão)
- Verificar logs para detalhes

### Performance lenta

- Usar GPU se disponível (`MODEL_DEVICE=0`)
- Reduzir `detection_confidence` se necessário
- Considerar usar jobs assíncronos para múltiplas imagens

## 📄 Licença

[Sua licença aqui]
