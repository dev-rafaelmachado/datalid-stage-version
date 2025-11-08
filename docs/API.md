# 🚀 Datalid API

API REST completa e modular para detecção e extração de datas de validade em imagens de produtos.

## 📋 Índice

- [Recursos](#-recursos)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [Documentação](#-documentação)
- [Endpoints](#-endpoints)
- [Configuração](#️-configuração)
- [Exemplos](#-exemplos)
- [Deploy](#-deploy)

## ✨ Recursos

- ✅ **Pipeline Completo**: YOLO → OCR → Parse de Datas
- ✅ **Múltiplas Engines OCR**: OpenOCR, Tesseract, EasyOCR, PaddleOCR, PARSeq, TrOCR
- ✅ **Processamento em Lote**: Processe múltiplas imagens de uma vez
- ✅ **Documentação Automática**: Swagger UI e ReDoc
- ✅ **Validação Robusta**: Pydantic schemas com validação automática
- ✅ **Rate Limiting**: Proteção contra abuso
- ✅ **CORS Configurável**: Integração fácil com frontends
- ✅ **Logging Completo**: Rastreamento de todas as requisições
- ✅ **Autenticação Opcional**: API Key based
- ✅ **Visualizações**: Retorne imagens com detecções desenhadas
- ✅ **Cache Inteligente**: Otimização de performance

## 🔧 Instalação

### 1. Requisitos

```bash
Python 3.8+
pip
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar Ambiente

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp .env.example .env
```

Edite `.env` com suas configurações.

## 🚀 Uso Rápido

### Iniciar o Servidor

```bash
# Método 1: Usando Python
python -m src.api.main

# Método 2: Usando Uvicorn diretamente
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Método 3: Com auto-reload (desenvolvimento)
uvicorn src.api.main:app --reload
```

### Testar

```bash
# Health check
curl http://localhost:8000/health

# Informações da API
curl http://localhost:8000/info
```

## 📚 Documentação

Após iniciar o servidor, acesse:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🛣️ Endpoints

### Health & Info

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações básicas da API |
| GET | `/health` | Health check com status dos componentes |
| GET | `/info` | Informações detalhadas (modelos, engines, limites) |

### Processamento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/process` | Processa uma única imagem |
| POST | `/process/batch` | Processa múltiplas imagens em lote |
| POST | `/process/url` | Processa imagem de uma URL |

## ⚙️ Configuração

### Variáveis de Ambiente

Principais configurações (veja `.env.example` para todas):

```bash
# Servidor
PORT=8000
WORKERS=1

# Limites
MAX_FILE_SIZE_MB=10.0
MAX_BATCH_SIZE=50

# Modelos
DEFAULT_YOLO_MODEL=experiments/yolov8m_seg_best/weights/best.pt
DEFAULT_OCR_ENGINE=openocr

# Segurança
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60

# Autenticação (opcional)
AUTH_ENABLED=false
API_KEYS=["sua-chave-secreta-aqui"]
```

## 💡 Exemplos

### Python

```python
import requests

# Processar uma imagem
with open('produto.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/process',
        files=files
    )

result = response.json()
print(f"Status: {result['status']}")
print(f"Data encontrada: {result['best_date']['date']}")
print(f"Confiança: {result['best_date']['confidence']}")
```

### cURL

```bash
# Processar imagem
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg" \
  -F "detection_confidence=0.3" \
  -F "ocr_engine=openocr" \
  -F "return_visualization=true"

# Processar de URL
curl -X POST "http://localhost:8000/process/url" \
  -F "url=https://exemplo.com/produto.jpg"

# Batch
curl -X POST "http://localhost:8000/process/batch" \
  -F "files=@produto1.jpg" \
  -F "files=@produto2.jpg" \
  -F "files=@produto3.jpg"
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('produto.jpg'));

axios.post('http://localhost:8000/process', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Data encontrada:', response.data.best_date.date);
})
.catch(error => {
  console.error('Erro:', error.response.data);
});
```

### Response Example

```json
{
  "status": "success",
  "message": "1 data(s) de validade encontrada(s)",
  "detections": [
    {
      "bbox": {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 350.7,
        "y2": 120.9,
        "width": 230.2,
        "height": 40.6
      },
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "exp_date",
      "has_mask": true
    }
  ],
  "dates": [
    {
      "date": "2025-12-31",
      "confidence": 0.92,
      "format": "DD/MM/YYYY",
      "is_valid": true,
      "is_expired": false,
      "days_until_expiry": 421
    }
  ],
  "best_date": {
    "date": "2025-12-31",
    "confidence": 0.92,
    "format": "DD/MM/YYYY",
    "is_valid": true,
    "is_expired": false,
    "days_until_expiry": 421
  },
  "metrics": {
    "total_time": 1.234,
    "detection_time": 0.456,
    "ocr_time": 0.678,
    "parsing_time": 0.100,
    "num_detections": 1,
    "num_dates_found": 1
  },
  "processed_at": "2025-11-05T10:30:00",
  "request_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

## 🚢 Deploy

### Docker (Recomendado)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Expor porta
EXPOSE 8000

# Comando
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t datalid-api .

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models datalid-api
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
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - MODEL_DEVICE=cpu
    restart: unless-stopped
```

### Produção

Para produção, considere:

1. **Usar HTTPS**: Reverse proxy (Nginx/Traefik)
2. **Múltiplos Workers**: Ajustar `WORKERS` conforme CPU
3. **Autenticação**: Ativar `AUTH_ENABLED=true`
4. **Rate Limiting**: Ajustar conforme carga esperada
5. **Monitoring**: Integrar com Prometheus/Grafana
6. **Load Balancing**: Se precisar escalar horizontalmente

## 🔒 Segurança

### Autenticação

Ative autenticação via API Key:

```bash
# .env
AUTH_ENABLED=true
API_KEY_HEADER=X-API-Key
API_KEYS=["chave-secreta-1", "chave-secreta-2"]
```

Uso:

```bash
curl -X POST "http://localhost:8000/process" \
  -H "X-API-Key: chave-secreta-1" \
  -F "file=@produto.jpg"
```

### Rate Limiting

Limite de requisições por IP:

```bash
# .env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60  # 60 requisições por minuto
```

## 📊 Monitoramento

### Logs

Logs são salvos em `logs/api.log` com rotação automática.

### Métricas

Acesse `/health` para verificar status dos componentes.

## 🛠️ Desenvolvimento

### Estrutura

```
src/api/
├── __init__.py         # Exports principais
├── main.py            # Aplicação FastAPI
├── config.py          # Configurações (Pydantic Settings)
├── schemas.py         # Modelos de dados (Pydantic)
├── routes.py          # Endpoints da API
├── service.py         # Lógica de processamento
├── middleware.py      # Middleware (logging, CORS, etc)
└── utils.py           # Utilitários
```

### Adicionar Novo Endpoint

1. Adicionar rota em `routes.py`
2. Criar schemas em `schemas.py` (se necessário)
3. Implementar lógica em `service.py` (se necessário)
4. Documentar com docstrings

### Testes

```bash
# TODO: Implementar testes
pytest tests/api/
```

## 🐛 Troubleshooting

### Porta já em uso

```bash
# Mudar porta
PORT=8001 uvicorn src.api.main:app
```

### GPU não disponível

```bash
# Usar CPU
MODEL_DEVICE=cpu uvicorn src.api.main:app
```

### Erro ao carregar modelo

Verifique se o caminho do modelo está correto em `.env`:

```bash
DEFAULT_YOLO_MODEL=experiments/yolov8m_seg_best/weights/best.pt
```

## 📝 Licença

CC BY 4.0

## 👥 Contribuindo

Contribuições são bem-vindas! Abra uma issue ou pull request.

## 📧 Suporte

Para questões ou problemas, abra uma issue no repositório.

---

**Feito com ❤️ pela equipe Datalid**
