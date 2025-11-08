# 📁 Scripts da API

Scripts auxiliares para trabalhar com a API Datalid.

## 🚀 Iniciar a API

```bash
python scripts/api/run_api.py
```

Ou usando uvicorn diretamente:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Com auto-reload (desenvolvimento):

```bash
uvicorn src.api.main:app --reload
```

## 🧪 Testar a API

```bash
python scripts/api/test_api.py
```

Este script testa todos os endpoints e exibe um relatório completo.

## 🔌 Cliente Python

O arquivo `client.py` contém um cliente Python para facilitar o uso da API:

```python
from scripts.api.client import DatalidClient

# Criar cliente
client = DatalidClient("http://localhost:8000")

# Verificar se está pronta
if client.is_ready():
    print("✅ API pronta!")

# Processar imagem
result = client.process_image("produto.jpg")
print(f"Data: {result['best_date']['date']}")

# Verificar se expirado
if client.is_expired("produto.jpg"):
    print("⚠️ Produto expirado!")

# Dias até expiração
days = client.days_until_expiry("produto.jpg")
print(f"Expira em {days} dias")
```

## 📝 Arquivos

- `run_api.py` - Inicia a API
- `test_api.py` - Testa todos os endpoints
- `client.py` - Cliente Python para a API
