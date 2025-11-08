# 🚀 Datalid API

API REST completa e modular para detecção e extração de datas de validade usando Deep Learning (YOLO) e OCR.

## ✨ Características

### Core Features
- ✅ **Detecção de Datas** - YOLO segmentation para localizar regiões de validade
- ✅ **Múltiplos Engines OCR** - Suporte para OpenOCR, PARSeq, TrOCR, Tesseract, EasyOCR, PaddleOCR
- ✅ **Parsing Inteligente** - Extração e validação automática de datas
- ✅ **GPU Accelerated** - Suporte completo para CUDA
- ✅ **API REST** - FastAPI com documentação automática
- ✅ **Cliente Python** - SDK fácil de usar

### Recursos Avançados
- 🔄 **Processamento Assíncrono** - Sistema de jobs para processar em background
- 🔌 **WebSocket** - Processamento em tempo real com feedback de progresso
- 📊 **Métricas** - Monitoramento completo (Prometheus-compatible)
- 🔐 **Autenticação** - Suporte para API Keys e JWT
- ⚡ **Batch Processing** - Processar múltiplas imagens de uma vez
- 🌐 **CORS** - Configurável para uso em webapps
- 📝 **Logging** - Sistema robusto com Loguru
- 🐳 **Docker** - Deploy fácil com Docker/Docker Compose

## 🚀 Quick Start

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Iniciar Servidor

```bash
# Modo simples
python -m src.api.main

# Ou com o script
python scripts/api/start_server.py --dev

# Com configurações personalizadas
python scripts/api/start_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --device cuda:0 \
  --ocr-engine openocr
```

### 3. Acessar Documentação

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 4. Testar API

```bash
# Via script de teste
python scripts/api/test_api.py

# Via curl
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg"
```

## 📖 Uso Básico

### Via Cliente Python

```python
from scripts.api.client import DatalidClient

# Criar cliente
client = DatalidClient("http://localhost:8000")

# Processar imagem
result = client.process_image("produto.jpg")

# Acessar resultados
if result['best_date']:
    print(f"Data: {result['best_date']['date']}")
    print(f"Confiança: {result['best_date']['confidence']}")
    print(f"Dias até expirar: {result['best_date']['days_until_expiry']}")
    
    if result['best_date']['is_expired']:
        print("⚠️ Produto expirado!")
```

### Via HTTP/cURL

```bash
# Processar imagem
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@produto.jpg" \
  -F "detection_confidence=0.3" \
  -F "ocr_engine=openocr" \
  -F "return_visualization=true"

# Health check
curl http://localhost:8000/health

# Informações
curl http://localhost:8000/info

# Métricas
curl http://localhost:8000/v2/metrics
```

## 🔌 Endpoints Principais

### API v1

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/info` | GET | Informações da API |
| `/process` | POST | Processar uma imagem |
| `/process/batch` | POST | Processar múltiplas imagens |
| `/process/url` | POST | Processar imagem de URL |

### API v2 (Recursos Avançados)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/v2/ws` | WS | WebSocket para tempo real |
| `/v2/jobs` | POST | Criar job assíncrono |
| `/v2/jobs/{id}` | GET | Status do job |
| `/v2/jobs` | GET | Listar jobs |
| `/v2/jobs/{id}` | DELETE | Cancelar job |
| `/v2/metrics` | GET | Métricas do sistema |
| `/v2/metrics/endpoints` | GET | Métricas por endpoint |
| `/v2/metrics/prometheus` | GET | Formato Prometheus |

## 📦 Exemplos Avançados

### Processamento em Batch

```python
client = DatalidClient("http://localhost:8000")

# Processar múltiplas imagens
results = client.process_batch([
    "produto1.jpg",
    "produto2.jpg",
    "produto3.jpg"
])

print(f"Sucesso: {results['successful']}/{results['total_images']}")

for res in results['results']:
    if res['success'] and res['result']['best_date']:
        print(f"{res['filename']}: {res['result']['best_date']['date']}")
```

### Jobs Assíncronos

```python
# Criar job
job = client.create_job("produto.jpg")
print(f"Job ID: {job['job_id']}")

# Aguardar conclusão
result = client.wait_for_job(job['job_id'])

# Ou de forma mais simples
result = client.process_image_async("produto.jpg")
```

### WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/v2/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`Progresso: ${data.progress * 100}%`);
  }
  
  if (data.type === 'result') {
    console.log('Data encontrada:', data.data.best_date.date);
  }
};

// Enviar imagem
ws.send(JSON.stringify({
  type: 'process',
  image: imageBase64
}));
```

## ⚙️ Configuração

### Arquivo .env

Copie `.env.example` para `.env` e ajuste:

```env
# Servidor
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Autenticação
AUTH_ENABLED=false
API_KEYS=["key1","key2"]

# Modelos
MODEL_DEVICE=0  # 0 para GPU, cpu para CPU
DEFAULT_OCR_ENGINE=openocr
DEFAULT_CONFIDENCE=0.25

# Limites
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=50
RATE_LIMIT_REQUESTS=60

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

### Configuração Programática

```python
from src.api.config import APISettings

settings = APISettings(
    host="0.0.0.0",
    port=8080,
    default_ocr_engine="parseq",
    max_file_size_mb=20
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
  -e MODEL_DEVICE=0 \
  --gpus all \
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
    environment:
      - LOG_LEVEL=INFO
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

```yaml
scrape_configs:
  - job_name: 'datalid-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/v2/metrics/prometheus'
```

### Métricas Disponíveis

- `datalid_uptime_seconds` - Uptime da API
- `datalid_requests_total` - Total de requisições
- `datalid_errors_total` - Total de erros
- `datalid_images_processed_total` - Imagens processadas
- `datalid_dates_found_total` - Datas encontradas
- `datalid_processing_time_seconds` - Tempo médio de processamento
- `datalid_requests_per_minute` - Taxa de requisições
- `datalid_error_rate_percent` - Taxa de erro

## 🔐 Autenticação

### API Key

```python
# No .env
AUTH_ENABLED=true
API_KEYS=["chave-secreta-1","chave-secreta-2"]

# No cliente
client = DatalidClient(
    "http://localhost:8000",
    api_key="chave-secreta-1"
)
```

```bash
# Via curl
curl -H "X-API-Key: chave-secreta-1" \
  http://localhost:8000/process ...
```

## 📚 Documentação Completa

- **[Guia Completo](docs/API_COMPLETE_GUIDE.md)** - Documentação detalhada
- **[API Reference](docs/API.md)** - Referência de endpoints
- **[Quick Start](docs/API_QUICK_START.md)** - Início rápido
- **[Exemplos](examples/api_usage.py)** - Código de exemplo

## 🛠️ Desenvolvimento

### Rodar em Modo Debug

```bash
python scripts/api/start_server.py --dev
```

### Testes

```bash
# Teste rápido
python scripts/api/test_api.py

# Testes completos
pytest tests/api/

# Com cobertura
pytest tests/api/ --cov=src/api
```

### Formato de Código

```bash
black src/api/
isort src/api/
flake8 src/api/
```

## 🏗️ Arquitetura

```
src/api/
├── __init__.py          # Exports principais
├── main.py              # Aplicação FastAPI
├── config.py            # Configurações (Pydantic Settings)
├── routes.py            # Endpoints API v1
├── routes_v2.py         # Endpoints API v2
├── schemas.py           # Modelos Pydantic
├── service.py           # Lógica de processamento
├── auth.py              # Autenticação
├── middleware.py        # Middlewares
├── websocket.py         # WebSocket handlers
├── jobs.py              # Sistema de jobs
├── metrics.py           # Métricas e monitoramento
└── utils.py             # Utilitários
```

## 📈 Performance

### Benchmarks

- **Detecção**: ~200-500ms (GPU)
- **OCR**: ~300-800ms dependendo do engine
- **Total**: ~0.5-1.5s por imagem

### Otimizações

- Use GPU para melhor performance (`MODEL_DEVICE=0`)
- Configure batch processing para múltiplas imagens
- Use jobs assíncronos para não bloquear
- Ajuste `detection_confidence` conforme necessário
- Cache de modelos automático

## 🆘 Troubleshooting

### API não inicia

```bash
# Verificar porta
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Ver logs
tail -f logs/api.log
```

### Erro de memória GPU

```bash
# Usar CPU
MODEL_DEVICE=cpu python -m src.api.main

# Ou reduzir batch size
MAX_BATCH_SIZE=10
```

### Performance lenta

- Verificar se está usando GPU
- Reduzir resolução das imagens
- Usar OCR engine mais rápido (parseq_tiny)
- Aumentar `detection_confidence`

## 📄 Licença

[Sua licença aqui]

## 🤝 Contribuindo

Contributions são bem-vindas! Por favor leia [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## 📧 Suporte

- 📖 Documentação: [docs/](docs/)
- 🐛 Issues: [GitHub Issues]
- 💬 Discussões: [GitHub Discussions]

---

Feito com ❤️ usando FastAPI, YOLO, e múltiplos engines de OCR.
