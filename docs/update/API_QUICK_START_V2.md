# 🚀 Quick Start - Datalid API

Inicie em 5 minutos!

## 1️⃣ Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd datalid3.0

# Instale as dependências
pip install -r requirements.txt
```

## 2️⃣ Iniciar Servidor

```bash
# Modo simples
python -m src.api.main

# Modo desenvolvimento (com reload)
python scripts/api/start_server.py --dev

# Personalizado
python scripts/api/start_server.py \
  --port 8080 \
  --device cuda:0 \
  --ocr-engine openocr
```

Servidor em: **http://localhost:8000**  
Documentação: **http://localhost:8000/docs**

## 3️⃣ Primeiro Teste

### Via Browser
Acesse: http://localhost:8000/docs

Clique em `POST /process` → `Try it out` → Upload uma imagem → `Execute`

### Via Python

```python
from scripts.api.client import DatalidClient

client = DatalidClient("http://localhost:8000")
result = client.process_image("sua_imagem.jpg")

print(result['best_date']['date'])  # 2025-12-31
```

### Via cURL

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg"
```

## 4️⃣ Exemplos Completos

```bash
# Execute exemplos interativos
python examples/api_usage.py
```

## ✅ Pronto!

Próximos passos:
- 📖 [Documentação Completa](API_COMPLETE_GUIDE.md)
- 🔧 [Configuração](API_COMPLETE_GUIDE.md#configuração)
- 🐳 [Docker](API_COMPLETE_GUIDE.md#docker)
- 📊 [Monitoramento](API_COMPLETE_GUIDE.md#monitoramento)
