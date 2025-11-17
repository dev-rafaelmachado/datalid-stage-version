# ⚙️ Configuração com Variáveis de Ambiente

Guia de como configurar a API usando variáveis de ambiente.

## 📋 Visão Geral

A API agora carrega **todas** as configurações de variáveis de ambiente, permitindo fácil ajuste entre ambientes (desenvolvimento, produção, Docker, etc).

## 🎯 Prioridade de Configuração

A API segue esta ordem de prioridade:

1. **Variáveis de Ambiente** (.env ou exportadas)
2. Arquivo YAML de configuração
3. Valores padrão no código

## 📁 Arquivos de Configuração

### `.env` (desenvolvimento local)
```bash
cp .env.example .env
# Edite .env com suas configurações
```

### `.env.production` (produção/Docker)
Já configurado para ambiente Docker/AWS.

## 🔧 Principais Configurações

### 🎯 Modelo YOLO (Importante!)

```bash
# Local (desenvolvimento)
API_YOLO_MODEL_PATH=models/yolov8m-seg.pt

# Docker
API_YOLO_MODEL_PATH=/app/models/yolov8m-seg.pt

# Caminho absoluto
API_YOLO_MODEL_PATH=/home/user/models/yolov8m-seg.pt
```

### 🖥️ GPU vs CPU

```bash
# Usar CPU (padrão)
API_MODEL_DEVICE=cpu
API_USE_GPU=false

# Usar GPU (se disponível)
API_MODEL_DEVICE=cuda:0
API_USE_GPU=true
```

### 🌐 Servidor

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
```

### 📝 OCR

```bash
API_DEFAULT_OCR_ENGINE=openocr
API_DEFAULT_OCR_CONFIG=config/ocr/openocr.yaml
API_DEFAULT_PREPROCESSING_CONFIG=config/preprocessing/ppro-openocr.yaml
```

## 🚀 Uso

### 1. Desenvolvimento Local

```bash
# 1. Copiar arquivo de exemplo
cp .env.example .env

# 2. Editar .env
nano .env

# 3. Ajustar caminho do modelo
API_YOLO_MODEL_PATH=models/yolov8m-seg.pt

# 4. Iniciar API
python -m uvicorn src.api.main:app --reload
```

### 2. Docker

```bash
# Usar .env.production
docker run --env-file .env.production datalid-api

# Ou sobrescrever variáveis
docker run \
  -e API_YOLO_MODEL_PATH=/app/models/yolov8m-seg.pt \
  -e API_USE_GPU=false \
  datalid-api
```

### 3. Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    image: datalid-api
    env_file:
      - .env.production
    environment:
      # Sobrescrever específicas
      API_PORT: 8080
```

### 4. Exportar Manualmente (Linux/Mac)

```bash
export API_YOLO_MODEL_PATH=models/yolov8m-seg.pt
export API_USE_GPU=false
export API_LOG_LEVEL=DEBUG

python -m uvicorn src.api.main:app
```

### 5. PowerShell (Windows)

```powershell
$env:API_YOLO_MODEL_PATH = "models\yolov8m-seg.pt"
$env:API_USE_GPU = "false"
$env:API_LOG_LEVEL = "DEBUG"

python -m uvicorn src.api.main:app
```

## 🧪 Testar Configurações

Use o script de teste:

```bash
# Testar se configurações estão OK
python test_config.py
```

O script verifica:
- ✅ Variáveis de ambiente carregadas
- ✅ Modelo YOLO existe
- ✅ Diretórios criados
- ✅ Arquivos de configuração existem

**Exemplo de saída:**
```
🧪 Testando Configurações da API
═══════════════════════════════════════════════════════════════════

🖥️  SERVIDOR:
   Host: 0.0.0.0
   Port: 8000
   Workers: 1

🎯 MODELO YOLO:
   Caminho configurado: models/yolov8m-seg.pt
   Caminho resolvido: C:\...\models\yolov8m-seg.pt
   Existe: ✅ SIM
   Tamanho: 51.9 MB
   Device: cpu

✅ Tudo OK! Configurações válidas.
```

## 📦 Ambientes Diferentes

### Desenvolvimento

```bash
# .env
API_YOLO_MODEL_PATH=models/yolov8m-seg.pt
API_DEBUG=true
API_LOG_LEVEL=DEBUG
API_RELOAD=true
API_SAVE_RESULTS_BY_DEFAULT=true
```

### Produção Local

```bash
# .env.production
API_YOLO_MODEL_PATH=models/yolov8m-seg.pt
API_DEBUG=false
API_LOG_LEVEL=INFO
API_RELOAD=false
API_CLEANUP_UPLOADS=true
```

### Docker/AWS

```bash
# .env.production (Docker)
API_YOLO_MODEL_PATH=/app/models/yolov8m-seg.pt
API_DEBUG=false
API_LOG_LEVEL=INFO
API_CORS_ORIGINS=["https://seu-frontend.com"]
```

## 🔍 Resolução de Caminhos

A API tenta encontrar o modelo YOLO automaticamente em:

1. Caminho especificado (absoluto ou relativo)
2. `models/<nome-do-modelo>`
3. `./<caminho-especificado>`
4. `/app/models/<nome-do-modelo>` (Docker)
5. `../models/<nome-do-modelo>`

**Exemplo:**
```bash
# Você especifica:
API_YOLO_MODEL_PATH=yolov8m-seg.pt

# API tenta encontrar em:
# 1. yolov8m-seg.pt
# 2. models/yolov8m-seg.pt
# 3. ./yolov8m-seg.pt
# 4. /app/models/yolov8m-seg.pt
# 5. ../models/yolov8m-seg.pt
```

## 📚 Todas as Variáveis

### Servidor
- `API_HOST` - Host (padrão: 0.0.0.0)
- `API_PORT` - Porta (padrão: 8000)
- `API_WORKERS` - Workers (padrão: 1)
- `API_RELOAD` - Auto-reload (padrão: false)

### Modelo YOLO
- `API_YOLO_MODEL_PATH` - Caminho do modelo ⭐ **IMPORTANTE**
- `API_MODEL_DEVICE` - Device (cpu, cuda, cuda:0)
- `API_USE_GPU` - Usar GPU (true/false)

### Detecção
- `API_DEFAULT_CONFIDENCE` - Confiança mínima (0-1)
- `API_DEFAULT_IOU` - IoU threshold (0-1)

### OCR
- `API_DEFAULT_OCR_ENGINE` - Engine (openocr, easyocr, etc)
- `API_DEFAULT_OCR_CONFIG` - Config OCR
- `API_DEFAULT_PREPROCESSING_CONFIG` - Config pré-processamento

### Limites
- `API_MAX_FILE_SIZE_MB` - Tamanho máximo (MB)
- `API_MAX_BATCH_SIZE` - Batch máximo
- `API_ALLOWED_EXTENSIONS` - Extensões permitidas
- `API_REQUEST_TIMEOUT` - Timeout (segundos)

### CORS
- `API_CORS_ENABLED` - Habilitar (true/false)
- `API_CORS_ORIGINS` - Origins permitidas (JSON array)

### Storage
- `API_UPLOAD_DIR` - Diretório uploads
- `API_RESULTS_DIR` - Diretório resultados
- `API_CLEANUP_UPLOADS` - Limpar após processar

### Logging
- `API_LOG_LEVEL` - Level (DEBUG, INFO, WARNING, ERROR)
- `API_LOG_REQUESTS` - Log requests (true/false)
- `API_LOG_FILE` - Arquivo de log

### Auth (Opcional)
- `API_AUTH_ENABLED` - Habilitar auth
- `API_API_KEYS` - Chaves válidas (JSON array)

## 🐛 Troubleshooting

### Modelo não encontrado

```bash
❌ Erro: Modelo YOLO não encontrado: models/yolov8m-seg.pt
```

**Solução:**
1. Verifique se o arquivo existe
2. Use caminho absoluto
3. Verifique permissões

```bash
# Verificar
ls -la models/yolov8m-seg.pt

# Usar caminho absoluto
API_YOLO_MODEL_PATH=/home/user/projeto/models/yolov8m-seg.pt
```

### Variáveis não carregadas

```bash
⚠️  ATENÇÃO: Nenhuma variável de ambiente encontrada!
```

**Solução:**
1. Criar arquivo `.env`
2. Verificar localização (deve estar na raiz)
3. Reiniciar a aplicação

### GPU não funciona

```bash
# Verificar se CUDA está disponível
python -c "import torch; print(torch.cuda.is_available())"

# Se False:
API_USE_GPU=false
API_MODEL_DEVICE=cpu
```

## 💡 Dicas

### 1. Use .env para desenvolvimento
```bash
cp .env.example .env
# Edite livremente
```

### 2. Use .env.production para deploy
```bash
# Já configurado para Docker/AWS
docker run --env-file .env.production datalid-api
```

### 3. Teste antes de fazer deploy
```bash
python test_config.py
```

### 4. Use caminhos absolutos em produção
```bash
API_YOLO_MODEL_PATH=/app/models/yolov8m-seg.pt
```

### 5. Ajuste CORS em produção
```bash
# Desenvolvimento (permissivo)
API_CORS_ORIGINS=["*"]

# Produção (restritivo)
API_CORS_ORIGINS=["https://seu-frontend.com","https://www.seu-frontend.com"]
```

## 📖 Próximos Passos

1. ✅ Copiar `.env.example` para `.env`
2. ✅ Ajustar `API_YOLO_MODEL_PATH`
3. ✅ Testar com `python test_config.py`
4. ✅ Iniciar API
5. ✅ Fazer deploy

---

**Documentação relacionada:**
- [Deploy AWS](24-AWS-DEPLOY.md)
- [API REST](16-API-REST.md)
- [Deploy Rápido](../DEPLOY-QUICK.md)
