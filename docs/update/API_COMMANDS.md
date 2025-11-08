# 🛠️ Comandos Úteis - Datalid API

Referência rápida de comandos para trabalhar com a API.

## 🚀 Iniciar Servidor

```bash
# Modo desenvolvimento (auto-reload, debug)
python scripts/api/start_server.py --dev

# Modo produção
python scripts/api/start_server.py --host 0.0.0.0 --port 8000 --workers 4

# Com GPU
python scripts/api/start_server.py --device cuda:0

# Com OCR específico
python scripts/api/start_server.py --ocr-engine openocr

# Com autenticação
python scripts/api/start_server.py --auth --api-key "minha-chave"

# Via Makefile
make api-dev      # desenvolvimento
make api-start    # produção
```

## 🧪 Testar API

```bash
# Teste rápido
python scripts/api/test_api.py

# Health check
curl http://localhost:8000/health

# Info da API
curl http://localhost:8000/info

# Via Makefile
make api-test
make api-health
```

## 📸 Processar Imagens

### Via cURL

```bash
# Processar uma imagem
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg" \
  -F "detection_confidence=0.3" \
  -F "ocr_engine=openocr" \
  -F "return_visualization=true"

# Com autenticação
curl -X POST "http://localhost:8000/process" \
  -H "X-API-Key: sua-chave" \
  -F "file=@produto.jpg"

# Processar de URL
curl -X POST "http://localhost:8000/process/url" \
  -F "url=https://example.com/produto.jpg"

# Batch (múltiplas imagens)
curl -X POST "http://localhost:8000/process/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"
```

### Via Python

```python
from scripts.api.client import DatalidClient

client = DatalidClient("http://localhost:8000")

# Simples
result = client.process_image("produto.jpg")

# Com opções
result = client.process_image(
    "produto.jpg",
    detection_confidence=0.3,
    ocr_engine="openocr",
    return_visualization=True
)

# Batch
results = client.process_batch([
    "img1.jpg",
    "img2.jpg",
    "img3.jpg"
])

# De URL
result = client.process_url("https://example.com/produto.jpg")

# Helpers
date = client.get_expiry_date("produto.jpg")
expired = client.is_expired("produto.jpg")
days = client.days_until_expiry("produto.jpg")
```

## ⏱️ Jobs Assíncronos

### Via cURL

```bash
# Criar job
JOB_ID=$(curl -X POST "http://localhost:8000/v2/jobs" \
  -F "file=@produto.jpg" | jq -r '.job_id')

# Consultar status
curl "http://localhost:8000/v2/jobs/$JOB_ID"

# Listar jobs
curl "http://localhost:8000/v2/jobs"

# Cancelar job
curl -X DELETE "http://localhost:8000/v2/jobs/$JOB_ID"
```

### Via Python

```python
# Criar job
job = client.create_job("produto.jpg")
print(f"Job ID: {job['job_id']}")

# Aguardar conclusão
result = client.wait_for_job(job['job_id'])

# Ou de forma mais simples
result = client.process_image_async("produto.jpg")

# Sem aguardar
job = client.process_image_async("produto.jpg", wait=False)
# ... fazer outras coisas ...
result = client.wait_for_job(job['job_id'])

# Listar jobs
jobs = client.list_jobs(status='completed')

# Cancelar job
client.cancel_job(job['job_id'])
```

## 🔌 WebSocket

### JavaScript

```javascript
const ws = new WebSocket('ws://localhost:8000/v2/ws');

ws.onopen = () => {
  console.log('Conectado!');
  
  // Enviar imagem
  ws.send(JSON.stringify({
    type: 'process',
    image: imageBase64
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`${data.step}: ${data.progress * 100}%`);
  }
  
  if (data.type === 'result') {
    console.log('Resultado:', data.data);
    ws.close();
  }
};

// Ping
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```

### Python

```python
import asyncio
import websockets
import json
import base64

async def process():
    uri = "ws://localhost:8000/v2/ws"
    
    async with websockets.connect(uri) as ws:
        # Enviar imagem
        with open("produto.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        await ws.send(json.dumps({
            "type": "process",
            "image": image_b64
        }))
        
        # Receber mensagens
        async for msg in ws:
            data = json.loads(msg)
            
            if data['type'] == 'progress':
                print(f"Progresso: {data['progress']*100:.0f}%")
            
            if data['type'] == 'result':
                print("Resultado:", data['data'])
                break

asyncio.run(process())
```

## 📊 Métricas

```bash
# Métricas do sistema
curl http://localhost:8000/v2/metrics

# Métricas por endpoint
curl http://localhost:8000/v2/metrics/endpoints

# Formato Prometheus
curl http://localhost:8000/v2/metrics/prometheus

# Via Makefile
make api-metrics

# Via Python
metrics = client.get_metrics()
print(f"Imagens: {metrics['total_images_processed']}")
print(f"Tempo médio: {metrics['average_processing_time']:.2f}s")
```

## 🐳 Docker

```bash
# Build
docker build -t datalid-api .

# Run (CPU)
docker run -d -p 8000:8000 \
  --name datalid-api \
  -v $(pwd)/models:/app/models:ro \
  datalid-api

# Run (GPU)
docker run -d -p 8000:8000 \
  --name datalid-api \
  --gpus all \
  -v $(pwd)/models:/app/models:ro \
  datalid-api

# Docker Compose
docker-compose up -d

# Logs
docker-compose logs -f api

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## 🔍 Monitoramento

```bash
# Health check
curl http://localhost:8000/health | jq

# Status dos componentes
curl http://localhost:8000/health | jq '.components'

# Logs (Docker)
docker-compose logs -f api

# Logs (local)
tail -f logs/api.log

# Métricas em tempo real
watch -n 5 'curl -s http://localhost:8000/v2/metrics | jq'
```

## 🧹 Manutenção

```bash
# Limpar cache
curl -X POST http://localhost:8000/v2/admin/clear-cache

# Recarregar modelos
curl -X POST http://localhost:8000/v2/admin/reload

# Limpar uploads temporários
rm -rf uploads/*

# Limpar logs antigos
rm logs/*.log.gz
```

## 📝 Desenvolvimento

```bash
# Instalar dependências
pip install -r requirements.txt

# Modo desenvolvimento
python scripts/api/start_server.py --dev

# Com reload automático
python scripts/api/start_server.py --reload

# Com debug
python scripts/api/start_server.py --debug

# Testes
pytest tests/api/

# Cobertura
pytest tests/api/ --cov=src/api

# Lint
flake8 src/api/
black src/api/
isort src/api/

# Type check
mypy src/api/
```

## 🔐 Segurança

```bash
# Gerar API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Gerar JWT secret
openssl rand -hex 32

# Testar com autenticação
curl -H "X-API-Key: sua-chave" \
  http://localhost:8000/process ...

# Verificar rate limit
for i in {1..100}; do
  curl -s http://localhost:8000/health > /dev/null
done
```

## 🧪 Testes de Carga

```bash
# Instalar hey
go install github.com/rakyll/hey@latest

# Teste simples (health)
hey -n 1000 -c 10 http://localhost:8000/health

# Teste de processamento
hey -n 100 -c 5 \
  -m POST \
  -H "Content-Type: multipart/form-data" \
  -D image.jpg \
  http://localhost:8000/process

# Monitorar durante teste
watch -n 1 'curl -s http://localhost:8000/v2/metrics | jq .requests_per_minute'
```

## 📦 Backup e Restore

```bash
# Backup dos modelos
tar -czf models_backup.tar.gz models/

# Backup das configurações
tar -czf config_backup.tar.gz config/ .env

# Backup dos resultados
tar -czf results_backup.tar.gz outputs/api_results/

# Restore
tar -xzf models_backup.tar.gz
tar -xzf config_backup.tar.gz
tar -xzf results_backup.tar.gz
```

## 🔄 Atualização

```bash
# Pull latest
git pull origin main

# Atualizar dependências
pip install -r requirements.txt --upgrade

# Restart
docker-compose restart api

# Ou
pkill -f "uvicorn"
python scripts/api/start_server.py --dev
```

## 📊 Grafana Dashboard

```bash
# Queries úteis para Grafana

# Taxa de requisições
rate(datalid_requests_total[5m])

# Taxa de erro
rate(datalid_errors_total[5m]) / rate(datalid_requests_total[5m]) * 100

# Tempo médio de processamento
avg(datalid_processing_time_seconds)

# Imagens processadas por hora
rate(datalid_images_processed_total[1h]) * 3600

# Taxa de sucesso de datas
rate(datalid_dates_found_total[5m]) / rate(datalid_images_processed_total[5m]) * 100
```

## 🎯 Comandos Makefile

```bash
# API
make api-dev          # Iniciar em modo dev
make api-start        # Iniciar em produção
make api-test         # Testar API
make api-examples     # Rodar exemplos
make api-docs         # Ver documentação
make api-client       # Testar cliente
make api-metrics      # Ver métricas
make api-health       # Health check

# Outros (se disponíveis)
make install          # Instalar dependências
make clean            # Limpar arquivos temporários
make test             # Todos os testes
```

## 🆘 Troubleshooting

```bash
# Verificar se porta está livre
lsof -i :8000                    # Linux/Mac
netstat -ano | findstr :8000     # Windows

# Matar processo na porta
kill -9 $(lsof -t -i:8000)       # Linux/Mac
taskkill /PID <pid> /F           # Windows

# Verificar GPU
nvidia-smi

# Verificar logs de erro
grep ERROR logs/api.log

# Testar conexão
telnet localhost 8000

# Verificar DNS/network
curl -v http://localhost:8000/health
```

## 🔗 Links Úteis

```bash
# Documentação
http://localhost:8000/docs        # Swagger UI
http://localhost:8000/redoc       # ReDoc
http://localhost:8000/openapi.json # OpenAPI Schema

# Endpoints
http://localhost:8000/            # Root
http://localhost:8000/health      # Health
http://localhost:8000/info        # Info
http://localhost:8000/v2/metrics  # Métricas

# WebSocket
ws://localhost:8000/v2/ws         # WebSocket

# Admin (se habilitado)
http://localhost:8000/v2/admin/*  # Admin endpoints
```

---

**💡 Dica:** Adicione estes comandos aos seus favoritos ou crie aliases no shell!

```bash
# Aliases úteis (.bashrc ou .zshrc)
alias api-start="python scripts/api/start_server.py --dev"
alias api-test="python scripts/api/test_api.py"
alias api-logs="tail -f logs/api.log"
alias api-metrics="curl -s http://localhost:8000/v2/metrics | jq"
```
