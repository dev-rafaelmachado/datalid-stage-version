# 🚀 Guia Rápido - API Datalid

## Instalação e Configuração (5 minutos)

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Ambiente

```bash
# Copiar arquivo de configuração
cp .env.example .env

# Editar .env conforme necessário
# (opcional - valores padrão funcionam para teste local)
```

### 3. Iniciar a API

```bash
# Método 1: Usando make
make api-run

# Método 2: Python direto
python scripts/api/run_api.py

# Método 3: Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Método 4: Desenvolvimento (com auto-reload)
make api-dev
```

A API estará disponível em: **http://localhost:8000**

## Testar a API (2 minutos)

### 1. Health Check

```bash
# Via curl
curl http://localhost:8000/health

# Ou via make
make api-health
```

### 2. Processar uma Imagem

```bash
# Via curl
curl -X POST "http://localhost:8000/process" \
  -F "file=@data/sample.jpg"

# Ou via make
make api-process IMAGE=data/sample.jpg
```

### 3. Testar Todos os Endpoints

```bash
python scripts/api/test_api.py
```

## Usar a API

### Python

```python
from scripts.api.client import DatalidClient

# Criar cliente
client = DatalidClient("http://localhost:8000")

# Processar imagem
result = client.process_image("produto.jpg")
print(f"Data: {result['best_date']['date']}")

# Verificar se expirado
if client.is_expired("produto.jpg"):
    print("⚠️ Produto expirado!")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/process', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Data encontrada:', data.best_date.date);
});
```

### cURL

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg" \
  -F "detection_confidence=0.3" \
  -F "ocr_engine=openocr"
```

## Documentação Interativa

Após iniciar a API, acesse:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker (Opcional)

```bash
# Build
docker build -t datalid-api .

# Run
docker run -p 8000:8000 datalid-api

# Ou com Docker Compose
docker-compose up -d
```

## Frontend Demo

Abra o arquivo `examples/frontend_demo.html` no navegador para uma interface web interativa.

## Endpoints Principais

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações básicas |
| `/health` | GET | Status de saúde |
| `/info` | GET | Informações detalhadas |
| `/process` | POST | Processar uma imagem |
| `/process/batch` | POST | Processar múltiplas imagens |
| `/process/url` | POST | Processar de URL |

## Parâmetros Comuns

- `detection_confidence` (float, 0-1): Confiança mínima para detecções
- `detection_iou` (float, 0-1): IoU threshold para NMS
- `ocr_engine` (string): Engine de OCR (openocr, tesseract, easyocr, etc)
- `return_visualization` (bool): Retornar imagem com visualizações
- `return_crops` (bool): Retornar crops das detecções
- `return_full_ocr` (bool): Retornar todos os resultados OCR

## Response Exemplo

```json
{
  "status": "success",
  "message": "1 data(s) de validade encontrada(s)",
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
  }
}
```

## Configurações Importantes

### Rate Limiting

Padrão: 60 requisições/minuto por IP
Configurar em `.env`:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
```

### Limites de Arquivo

Padrão: 10MB por imagem, 50 imagens por batch
Configurar em `.env`:

```bash
MAX_FILE_SIZE_MB=10.0
MAX_BATCH_SIZE=50
```

### Autenticação (Opcional)

```bash
# .env
AUTH_ENABLED=true
API_KEYS=["chave-secreta-1", "chave-secreta-2"]
```

Uso:

```bash
curl -X POST "http://localhost:8000/process" \
  -H "X-API-Key: chave-secreta-1" \
  -F "file=@produto.jpg"
```

## Troubleshooting

### Porta já em uso

```bash
# Mudar porta no .env
PORT=8001
```

### GPU não disponível

```bash
# Usar CPU
MODEL_DEVICE=cpu
```

### Modelo não encontrado

Verifique o caminho em `.env`:

```bash
DEFAULT_YOLO_MODEL=experiments/yolov8m_seg_best/weights/best.pt
```

## Comandos Make Úteis

```bash
make api-run          # Iniciar API
make api-dev          # Iniciar com auto-reload
make api-test         # Testar endpoints
make api-health       # Health check
make api-info         # Informações
make api-docker-build # Build Docker
make api-compose-up   # Docker Compose up
make api-clean        # Limpar uploads
```

## Próximos Passos

1. Ler documentação completa em `docs/API.md`
2. Explorar endpoints na Swagger UI
3. Testar com suas próprias imagens
4. Integrar com seu frontend/aplicação
5. Configurar para produção (HTTPS, workers, etc)

## Suporte

- Documentação: `docs/API.md`
- Exemplos: `examples/`
- Scripts: `scripts/api/`
- Issues: GitHub repository

---

**Pronto para começar!** 🚀

A API está configurada e pronta para uso. Acesse http://localhost:8000/docs para explorar todos os endpoints interativamente.
