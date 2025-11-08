# 📤 Guia de Upload de Arquivos

## Como Enviar Arquivos para a API

A API Datalid aceita arquivos de imagem via **multipart/form-data**, não JSON body.

---

## 🔧 Exemplos de Uso

### 1. cURL (Windows PowerShell)

```powershell
# Upload simples
curl -X POST http://localhost:8000/process `
  -F "file=@data/sample.jpg"

# Com parâmetros opcionais
curl -X POST http://localhost:8000/process `
  -F "file=@data/sample.jpg" `
  -F "detection_confidence=0.5" `
  -F "ocr_engine=openocr" `
  -F "return_visualization=true"
```

### 2. Python (requests)

```python
import requests

# Upload simples
with open('data/sample.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f}
    )
print(response.json())

# Com parâmetros opcionais
with open('data/sample.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f},
        data={
            'detection_confidence': 0.5,
            'ocr_engine': 'openocr',
            'return_visualization': True
        }
    )
print(response.json())
```

### 3. HTTPie

```bash
# Upload simples
http -f POST http://localhost:8000/process file@data/sample.jpg

# Com parâmetros opcionais
http -f POST http://localhost:8000/process \
  file@data/sample.jpg \
  detection_confidence=0.5 \
  ocr_engine=openocr \
  return_visualization=true
```

### 4. Postman

1. Selecione método **POST**
2. URL: `http://localhost:8000/process`
3. Vá para aba **Body**
4. Selecione **form-data**
5. Adicione campos:
   - `file` (tipo: File) → Selecione sua imagem
   - `detection_confidence` (tipo: Text) → `0.5` (opcional)
   - `ocr_engine` (tipo: Text) → `openocr` (opcional)
   - `return_visualization` (tipo: Text) → `true` (opcional)

### 5. JavaScript (Fetch API)

```javascript
// Upload com FormData
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('detection_confidence', '0.5');
formData.append('ocr_engine', 'openocr');

fetch('http://localhost:8000/process', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### 6. JavaScript (Axios)

```javascript
import axios from 'axios';

const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('detection_confidence', '0.5');

const response = await axios.post('http://localhost:8000/process', formData, {
    headers: {
        'Content-Type': 'multipart/form-data'
    }
});
console.log(response.data);
```

---

## 📋 Parâmetros Disponíveis

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `file` | File | ✅ Sim | Imagem (JPEG, PNG, BMP) |
| `detection_confidence` | float | ❌ Não | Confiança mínima (0.0-1.0) |
| `detection_iou` | float | ❌ Não | Threshold IoU (0.0-1.0) |
| `ocr_engine` | string | ❌ Não | Engine OCR (tesseract, openocr, etc) |
| `return_visualization` | bool | ❌ Não | Retornar imagem visualizada |
| `return_crops` | bool | ❌ Não | Retornar crops das detecções |
| `return_full_ocr` | bool | ❌ Não | Retornar todos resultados OCR |
| `save_results` | bool | ❌ Não | Salvar no servidor |

---

## ⚠️ Erros Comuns

### ❌ ERRO: Field required

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Erro de validação dos dados de entrada",
    "details": {
      "errors": [{
        "type": "missing",
        "loc": ["body", "file"],
        "msg": "Field required"
      }]
    }
  }
}
```

**Causa:** Arquivo não foi enviado ou foi enviado como JSON.

**Solução:** Use `multipart/form-data` (veja exemplos acima).

---

### ❌ ERRO: Formato não suportado

```json
{
  "detail": "Formato não suportado. Use: ['.jpg', '.jpeg', '.png', '.bmp']"
}
```

**Causa:** Arquivo não é uma imagem válida.

**Solução:** Use apenas JPEG, PNG ou BMP.

---

### ❌ ERRO: Arquivo muito grande

```json
{
  "detail": "Arquivo muito grande. Máximo: 10MB"
}
```

**Causa:** Arquivo excede o limite configurado.

**Solução:** Comprima a imagem ou aumente o limite no `.env`:
```env
API_MAX_FILE_SIZE_MB=20
```

---

## 🎯 Teste Rápido

Use este comando para testar se a API está funcionando:

```bash
# PowerShell
curl -X POST http://localhost:8000/process `
  -F "file=@data/sample.jpg"

# Bash/Linux
curl -X POST http://localhost:8000/process \
  -F "file=@data/sample.jpg"
```

---

## 📚 Mais Informações

- **Documentação da API:** http://localhost:8000/docs
- **HTTPie Guide:** [HTTPIE_GUIDE.md](./HTTPIE_GUIDE.md)
- **Variáveis de Ambiente:** [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md)
- **Python Client:** [src/api/client.py](../src/api/client.py)

---

## 🔄 Batch Processing

Para processar múltiplas imagens, use o endpoint `/batch`:

```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(
    'http://localhost:8000/batch',
    files=files
)
print(response.json())
```

---

## 💡 Dicas

1. **Use o Python Client** para facilitar:
   ```python
   from src.api.client import DatalidClient
   
   client = DatalidClient("http://localhost:8000")
   result = client.process_image("data/sample.jpg")
   ```

2. **Documentação interativa**: Acesse http://localhost:8000/docs para testar na interface do Swagger.

3. **Verificar formatos aceitos**:
   ```bash
   curl http://localhost:8000/info
   ```
